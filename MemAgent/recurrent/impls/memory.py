import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, now, unpad
from verl.protocol import DataProto

# ANSI color codes
PINK = '\033[95m'
RESET = '\033[0m'

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens3
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information.
    For implicit_persona dataset, the context is extracted from the prompt messages.
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        
        # Calculate and log max_prompt_length
        calculated_max_prompt_length = recurrent_config.max_chunks * recurrent_config.chunk_size
        data_config.max_prompt_length = calculated_max_prompt_length
        self.context_key = recurrent_config.context_key
        
        logger.info(f"[DATASET INIT] max_chunks={recurrent_config.max_chunks}, chunk_size={recurrent_config.chunk_size}, max_prompt_length={calculated_max_prompt_length}")
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        
        # Parse chat from JSON string if needed (since it's stored as JSON in parquet)
        if isinstance(chat, str):
            try:
                import json
                chat = json.loads(chat)
            except (json.JSONDecodeError, TypeError):
                chat = []
        
        # Keep the entire OpenAI message format intact - pass directly to chat template
        # Apply chat template to the full conversation messages
        context_text = self.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=False
        )

        model_inputs = self.tokenizer(context_text, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        
        # Store original length before truncation for chunk calculation
        original_length = attention_mask.sum(dim=-1).item()
        row_dict["original_context_length"] = original_length

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        processed_length = lengths[0].item()
        row_dict["context_length"] = lengths[0]
        
        # Extract just the last user message content for prompt_ids
        user_query_content = ""
        if isinstance(chat, list) and len(chat) > 0:
            user_query_content = chat[-1].get("content", "")
        
        row_dict["prompt_ids"] = self.tokenizer.encode(
            user_query_content, add_special_tokens=False
        )
        
        # Parse extra_info from JSON string if needed
        extra_info = row_dict.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                import json
                extra_info = json.loads(extra_info)
            except (json.JSONDecodeError, TypeError):
                extra_info = {}
        
        index = extra_info.get("index", 0) if isinstance(extra_info, dict) else 0
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
         # Note: original_context_length is optional and may not be present in all datasets
        return ["context_ids", "context_length"], ["prompt_ids"]

TEMPLATE = """You are presented with a conversation history between the current user and the chatbot. Your task is to carefully analyze this conversation history and extract information about the user's persona and preferences that are revealed or indicated explicitly or implicitly through their interactions, responses, and dialogue patterns.

Focus on identifying personas and preferences of the current user, either explicitly revealed or implicitly mentioned in the user-chatbot conversation histories.
Try to make each information point atomic. Do NOT record any multiple choice options or test questions.

Update the memory by retaining all relevant details from the previous memory while adding any new, useful persona and preference information discovered in the current conversation section.
You are writing a memory section. so do NOT include other texts in your response.

<user_query> 
{prompt}
</user_query>

<previous_memory>
{memory}
</previous_memory>

<conversation_section>
{chunk}
</conversation_section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a user query and previous memory about the user's persona and preferences. Please answer the user query based on the previous memory and provide a personalized response. Put your final answer in \\boxed{{}}.

<user_query> 
{prompt}
</user_query>

<user_memory>
{memory}
</user_memory>

Your answer:
"""


class MemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # A trick to get a simple chat_template for any tokenizer
        # the output text looks like:
        # '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n'
        # This is a format string itself, '{message}' will be replaced by the actual message.
        self.chat_template = chat_template(tokenizer)
        
        try:
            # Debug the template creation process
            formatted_template = self.chat_template.format(message=TEMPLATE)
            logger.info(f"[TEMPLATE DEBUG] Creating TokenTemplate with chat_template length: {len(formatted_template)}")
            logger.info(f"[TEMPLATE DEBUG] Chat template preview: {formatted_template[:200]}...")
            
            self.token_message_template = TokenTemplate(formatted_template, tokenizer)
            self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
        except Exception as e:
            logger.error(f"[TEMPLATE ERROR] Failed to create TokenTemplate: {e}")
            logger.error(f"[TEMPLATE ERROR] Chat template: {self.chat_template}")
            logger.error(f"[TEMPLATE ERROR] TEMPLATE content: {TEMPLATE}")
            raise
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template.length 
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {self.token_message_template.length}(message_template) = {self.max_input_length}\n')
        self.NO_MEMORY_TOKENS = tokenizer.encode("No previous memory", add_special_tokens=False)
    
    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        logger.info(f"[MEMORY AGENT] start() called, gen_batch keys: {list(gen_batch.batch.keys()) if hasattr(gen_batch, 'batch') else 'No batch'}")
        
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        
        logger.info(f"[MEMORY AGENT] Accessing context_length...")
        self.ctx_length = gen_batch.batch['context_length'] # processed/truncated lengths for actual processing
        
        # Check if original_context_length is available (might not be present in validation data)
        logger.info(f"[MEMORY AGENT] Checking for original_context_length...")
        if 'original_context_length' in gen_batch.batch:
            logger.info(f"[MEMORY AGENT] Found original_context_length in batch")
            self.original_ctx_length = gen_batch.batch['original_context_length'] # original lengths for chunk calculation
        else:
            # Fallback: use processed context length (validation might not have original lengths)
            logger.info(f"[MEMORY AGENT] original_context_length not found, using processed lengths as fallback")
            self.original_ctx_length = self.ctx_length
            
        logger.info(f"[MEMORY AGENT] Setting up batch size and memory...")
        self.bsz = len(self.ctx_length)
        self.memory = np.empty(self.bsz, dtype=object)
        self.is_final = False
        
        logger.info(f"[MEMORY AGENT] Calculating chunk statistics...")
        # Calculate chunks based on ORIGINAL context lengths (before truncation)
        original_chunks_per_sample = [(length + self.config.chunk_size - 1) // self.config.chunk_size for length in self.original_ctx_length]
        processed_chunks_per_sample = [(length + self.config.chunk_size - 1) // self.config.chunk_size for length in self.ctx_length]
        total_original_chunks = sum(original_chunks_per_sample)
        total_processed_chunks = sum(processed_chunks_per_sample)
        
        # Check if we have original context lengths or just processed ones
        has_original_lengths = 'original_context_length' in gen_batch.batch
        using_original = has_original_lengths and not torch.equal(self.original_ctx_length, self.ctx_length)

        logger.info(f"\n{'='*25}[RECURRENT] MEMORY PROCESSING INIT{'='*25}")
        logger.info(f"[BATCH] Total samples in batch: {self.bsz}")
        logger.info(f"[BATCH] Chunk size: {self.config.chunk_size} tokens")
        logger.info(f"[BATCH] Context length source: {'ORIGINAL (before truncation)' if using_original else 'PROCESSED (after truncation/validation data)'}")
        
        if using_original:
            logger.info(f"[BATCH] ORIGINAL CONTEXT (before truncation):")
            logger.info(f"  - Original lengths sample - first 10: {self.original_ctx_length[:10].tolist()}")
            logger.info(f"  - Original lengths - Min: {min(self.original_ctx_length)}, Max: {max(self.original_ctx_length)}, Avg: {sum(self.original_ctx_length)/len(self.original_ctx_length):.1f}")
            logger.info(f"  - Original chunks per sample - Max: {max(original_chunks_per_sample)}, Min: {min(original_chunks_per_sample)}, Avg: {sum(original_chunks_per_sample)/len(original_chunks_per_sample):.2f}")
            logger.info(f"  - Total original chunks needed: {total_original_chunks}")
            logger.info(f"[BATCH] PROCESSED CONTEXT (after truncation for processing):")
            logger.info(f"  - Processed lengths sample - first 10: {self.ctx_length[:10].tolist()}")
            logger.info(f"  - Processed lengths - Min: {min(self.ctx_length)}, Max: {max(self.ctx_length)}, Avg: {sum(self.ctx_length)/len(self.ctx_length):.1f}")
            logger.info(f"  - Processed chunks per sample - Max: {max(processed_chunks_per_sample)}, Min: {min(processed_chunks_per_sample)}, Avg: {sum(processed_chunks_per_sample)/len(processed_chunks_per_sample):.2f}")
            logger.info(f"  - Total processed chunks to execute: {total_processed_chunks}")
        else:
            logger.info(f"[BATCH] CONTEXT LENGTHS (using {'processed' if not has_original_lengths else 'same as original'}):")
            logger.info(f"  - Context lengths sample - first 10: {self.ctx_length[:10].tolist()}")
            logger.info(f"  - Context lengths - Min: {min(self.ctx_length)}, Max: {max(self.ctx_length)}, Avg: {sum(self.ctx_length)/len(self.ctx_length):.1f}")
            logger.info(f"  - Chunks per sample - Max: {max(processed_chunks_per_sample)}, Min: {min(processed_chunks_per_sample)}, Avg: {sum(processed_chunks_per_sample)/len(processed_chunks_per_sample):.2f}")
            logger.info(f"  - Total chunks to process: {total_processed_chunks}")
        
        logger.info(f"{'='*50}")
    
    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        # suppose 0 is pad_token_id
        # max_chunks = 3, chunk_sieze = 2
        # pi is token in prompt, ti is token in chat template, 
        # [1,2] [3,4] [5,0] | p0 string
        # [1,2] [3,0] [0,0] | p1,p1 string
        # [1,0] [0,0] [0,0] | p2,p2,p2 string
        # -------- round 1 ---------
        # [1,2]            [t0,p0,t1, m,t2, 1, 2,t3]                           [ 0, 0, 0,t0,p0,t1, m,t2, 1, 2,t3]
        # [1,2]  -format-> [t0,p1,p1,t1, m,t2, 1, 2,t3] -pad2Dlist2Tendors->   [ 0, 0,t0,p1,p1,t1, m,t2, 1, 2,t3]
        # [1,0]            [t0,p2,p2,p3,t1, m,t2, 1,t3]                        [ 0, 0,t0,p2,p2,p3,t1, m,t2, 1,t3]
        # get mask & positionids - use ORIGINAL context lengths to determine activity
        # This ensures we process all chunks based on the original conversation length
        # Also enforce max_chunks limit to prevent infinite processing
        context_based_active = self.original_ctx_length > self.step * self.config.chunk_size
        max_chunks_based_active = self.step < self.config.max_chunks
        active_mask = context_based_active & max_chunks_based_active
        self.active_mask = active_mask
        gen_batch = self.gen_batch
        
        # Log turn information with detailed debugging
        active_samples = active_mask.sum().item()
        context_active_samples = context_based_active.sum().item()
        current_token_threshold = self.step * self.config.chunk_size
        
        logger.info(f"\n{'='*30}[RECURRENT] STEP {self.step}{'='*30}")
        logger.info(f"[MEMORY] Active samples in this turn: {active_samples} / {self.bsz}")
        logger.info(f"[MEMORY] Context-based active samples: {context_active_samples}")
        logger.info(f"[MEMORY] Max chunks limit: {self.config.max_chunks}, current step: {self.step}")
        logger.info(f"[MEMORY] Processing chunk {self.step + 1} (tokens {self.step * self.config.chunk_size} to {(self.step + 1) * self.config.chunk_size})")
        logger.info(f"[MEMORY] Current token threshold for activity: {current_token_threshold}")
        logger.info(f"[MEMORY] Context lengths - First 10 samples: {self.ctx_length[:10].tolist()}")
        logger.info(f"[MEMORY] Original context lengths - First 10 samples: {self.original_ctx_length[:10].tolist()}")
        logger.info(f"[MEMORY] Active mask - First 10 samples: {active_mask[:10].tolist()}")
        logger.info(f"[MEMORY] Context-based active mask - First 10 samples: {context_based_active[:10].tolist()}")
        
        # Debug why samples become inactive
        if active_samples < self.bsz:
            inactive_samples = self.bsz - active_samples
            logger.info(f"[MEMORY] {inactive_samples} samples became inactive (context_length <= {current_token_threshold} OR step >= max_chunks)")
            # Show some examples of inactive samples
            inactive_indices = torch.where(~active_mask)[0][:5]  # First 5 inactive samples
            for idx in inactive_indices:
                context_active = context_based_active[idx].item()
                step_within_limit = self.step < self.config.max_chunks
                logger.info(f"[MEMORY]   Inactive sample {idx}: context_length={self.ctx_length[idx]}, original_length={self.original_ctx_length[idx]}, context_active={context_active}, step_within_limit={step_within_limit}")
        
        # Expected chunks calculation for debugging
        logger.info(f"[MEMORY] Expected chunks based on original lengths - First 10: {[(length + self.config.chunk_size - 1) // self.config.chunk_size for length in self.original_ctx_length[:10]]}")
        
        # if all context is used, and its not done, then it will be the final turn for this batch
        if active_samples == 0:
            self.is_final = True
            self.messages = [
                self.token_final_message_template.format(
                    prompt=prompt,
                    memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                )
                for prompt, memory in zip(gen_batch.non_tensor_batch['prompt_ids'], self.memory)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'[MEMORY] FINAL TURN: All samples completed chunking, moving to final response generation')
            logger.info(f'[MEMORY] Final turn will process all {self.bsz} samples for response generation')
        else:
            # 1. no need to pad prompt
            # 2. context padded for 2D indexing, elegant engineering
            # 3. no need to pad memory
            prompt_i = gen_batch.non_tensor_batch['prompt_ids'][active_mask]
            
            # Extract chunks from context_ids, but handle the case where we might be asking for chunks
            # beyond the truncated context length
            start_token = self.config.chunk_size * self.step
            end_token = self.config.chunk_size * (self.step + 1)
            
            # For samples where the chunk extends beyond the available context, we'll get padding
            # This is expected behavior when original context is longer than processed context
            chunk_i = gen_batch.batch['context_ids'][active_mask, start_token:end_token]
            
            # Log chunk extraction details
            chunk_sizes = []
            for i, sample_idx in enumerate(torch.where(active_mask)[0]):
                available_tokens = self.ctx_length[sample_idx]
                chunk_end_in_available = min(end_token, available_tokens)
                actual_chunk_size = max(0, chunk_end_in_available - start_token)
                chunk_sizes.append(actual_chunk_size)
            
            logger.info(f"[MEMORY] Chunk extraction: tokens {start_token}-{end_token}")
            logger.info(f"[MEMORY] Actual chunk sizes (non-padded tokens): {chunk_sizes[:10]}")
            
            memory_i = self.memory[active_mask]
            
            # format: we use our token_template to avoid decoding & formatting with str function & encoding back.
            self.messages = [
                self.token_message_template.format(
                        prompt=prompt,
                        memory=memory if memory is not None else self.NO_MEMORY_TOKENS, # use pre-tokenized "No previous memory" for first round
                        chunk=chunk[chunk != self.tokenizer.pad_token_id], # unpadding needed here
                )
                for prompt, memory, chunk in zip(prompt_i, memory_i, chunk_i)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'MemoryAgent.action() done')
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        if not self.is_final:
            # Update memory with new responses
            new_memories = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            self.memory[self.active_mask] = new_memories

        self.log_step(gen_output)
        self.step += 1
        return gen_output
    
    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=2000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]
        
        def colorize_memory_responses(text):
            """Add pink color to memory generation responses (responses to TEMPLATE prompt)."""
            # For memory turns (non-final), highlight the entire response
            if not self.is_final:
                # Apply pink color to each line individually to ensure proper coloring across multi-line text
                lines = text.split('\n')
                colored_lines = [f"{PINK}{line}{RESET}" for line in lines]
                return '\n'.join(colored_lines)
            else:
                # For final turns, don't highlight since it's the final answer, not memory content
                return text

        # Batch information
        batch_size = gen_output.batch['responses'].shape[0] if gen_output.batch['responses'].dim() > 1 else 1
        is_final = self.is_final
        
        logger.info(f"[MEMORY] Step {self.step} completed")
        logger.info(f"[MEMORY] Batch size for this turn: {batch_size}")
        logger.info(f"[MEMORY] Turn type: {'FINAL (response generation)' if is_final else 'MEMORY (chunk processing)'}")
        
        if is_final:
            logger.info(f"[MEMORY] FINAL TURN SUMMARY:")
            logger.info(f"  - Samples processed in final turn: {batch_size} (all samples generate final responses)")
            logger.info(f"  - Total memory processing steps completed: {self.step}")
        else:
            active_count = self.active_mask.sum().item() if hasattr(self, 'active_mask') else batch_size
            logger.info(f"[MEMORY] MEMORY TURN SUMMARY:")
            logger.info(f"  - Active samples in this chunk: {active_count}")
            logger.info(f"  - Chunk range: tokens {self.step * self.config.chunk_size} to {(self.step + 1) * self.config.chunk_size}")

        # Message and Response section (show only first sample to avoid clutter)
        if self.is_final:
            # In final turn, show first sample (index 0) since all samples are processed
            if len(self.messages) > 0:
                decoded_message = self.tokenizer.decode(self.messages[0])
                rsp = gen_output.batch['responses'][0]
                decoded_response = self.tokenizer.decode(rsp[rsp!=self.tokenizer.pad_token_id])
                
                # Apply color formatting to memory responses
                colored_message = clip_long_string(decoded_message)  # Don't color the prompt
                colored_response = colorize_memory_responses(decoded_response)  # Color memory responses
                
                logger.info(f"[MESSAGE SAMPLE 0] {colored_message}")
                logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
                logger.info(f"[RESPONSE SAMPLE 0] {colored_response}")
                logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
            else:
                logger.info("MESSAGE and RESPONSE is empty - no messages available.")
        elif hasattr(self, 'active_mask') and self.active_mask.any():
            # Find first active sample
            active_idx = torch.where(self.active_mask)[0][0].item()
            
            # Check bounds to prevent IndexError
            if active_idx < len(self.messages):
                decoded_message = self.tokenizer.decode(self.messages[active_idx])
                rsp = gen_output.batch['responses'][active_idx]
                decoded_response = self.tokenizer.decode(rsp[rsp!=self.tokenizer.pad_token_id])
                
                # Apply color formatting to memory responses
                colored_message = clip_long_string(decoded_message)  # Don't color the prompt
                colored_response = colorize_memory_responses(decoded_response)  # Color memory responses
                
                logger.info(f"[MESSAGE SAMPLE {active_idx}] {colored_message}")
                logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
                logger.info(f"[RESPONSE SAMPLE {active_idx}] {colored_response}")
                logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
            else:
                logger.info(f"[WARNING] Active sample index {active_idx} is out of bounds for messages list (length: {len(self.messages)})")
                logger.info("Skipping message/response logging for this sample.")
        else:
            logger.info("MESSAGE and RESPONSE is empty since no samples are active.")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
