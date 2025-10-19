
# # Original path in verl: verl/workers/actor/dp_actor.py
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty, grad_acc_mode
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ANSI color codes for colored text
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        # Validate input micro_batch before processing
        if "input_ids" not in micro_batch:
            raise ValueError("micro_batch must contain 'input_ids'")
        
        input_ids = micro_batch["input_ids"]
        if input_ids.numel() == 0 or input_ids.shape[0] == 0:
            raise ValueError(f"Empty input_ids tensor in micro_batch: shape={input_ids.shape}")
        
        response_length = micro_batch["responses"].size(-1)
        if response_length == 0:
            raise ValueError(f"Empty responses tensor in micro_batch: shape={micro_batch['responses'].shape}")
        
        # Enhanced debugging: Print response and attention mask details for problematic cases
        if 'responses' in micro_batch and 'attention_mask' in micro_batch and 'response_mask' in micro_batch:
            response_mask = micro_batch['response_mask']
            attention_mask = micro_batch['attention_mask']
            batch_size = micro_batch['responses'].shape[0]
            
            # Quick check for critical issues only
            for i in range(batch_size):
                resp_sum = response_mask[i].sum().item()
                attn_sum = attention_mask[i].sum().item()
                
                if resp_sum > 0 and attn_sum == 0:
                    print(f'[FORWARD] 🚨 CRITICAL: Sample {i} has {resp_sum} response tokens but ZERO attention!')
                elif resp_sum == 0:
                    print(f'[FORWARD] ℹ️  Sample {i}: padded sample (no response tokens), attention_sum={attn_sum}')
        
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch_size, seqlen = input_ids.shape
            if batch_size == 0 or seqlen == 0:
                raise ValueError(f"Invalid tensor dimensions: batch_size={batch_size}, seqlen={seqlen}")
            
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            
            # Validate attention mask before remove padding operations
            # For padded micro-batches with minimal attention masks (only first token), handle gracefully
            if self.use_remove_padding and attention_mask.sum() <= batch_size:
                # Check if this looks like a minimally padded sample (only first tokens are valid)
                first_token_sum = attention_mask[:, 0].sum()
                if first_token_sum == attention_mask.sum() and first_token_sum == batch_size:
                    print(f"[FORWARD] Detected minimally padded micro-batch (only first tokens valid) - returning dummy results. attention_mask shape: {attention_mask.shape}")
                    # Return dummy tensors with appropriate shapes for minimally padded micro-batches
                    dummy_entropy = torch.zeros((batch_size, response_length), device=attention_mask.device, dtype=torch.float32) if calculate_entropy else None
                    dummy_log_probs = torch.zeros((batch_size, response_length), device=attention_mask.device, dtype=torch.float32)
                    return dummy_entropy, dummy_log_probs
                elif attention_mask.sum() == 0:
                    print(f"[FORWARD] Warning: Attention mask is all zeros (completely padded micro-batch) - returning dummy results. attention_mask shape: {attention_mask.shape}")
                    # Return dummy tensors with appropriate shapes for completely padded micro-batches
                    dummy_entropy = torch.zeros((batch_size, response_length), device=attention_mask.device, dtype=torch.float32) if calculate_entropy else None
                    dummy_log_probs = torch.zeros((batch_size, response_length), device=attention_mask.device, dtype=torch.float32)
                    return dummy_entropy, dummy_log_probs
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                
                # Check if unpadding resulted in empty tensor (all tokens were padding)
                # Return dummy results for padded micro-batches instead of raising exceptions
                if input_ids_rmpad.numel() == 0:
                    print(f"[FORWARD] Warning: All tokens were padding after unpad_input (padded micro-batch) - returning dummy results. Original input_ids shape: {input_ids.shape}, attention_mask sum: {attention_mask.sum()}")
                    # Return dummy tensors with appropriate shapes for padded micro-batches
                    dummy_entropy = torch.zeros((batch_size, response_length), device=attention_mask.device, dtype=torch.float32) if calculate_entropy else None
                    dummy_log_probs = torch.zeros((batch_size, response_length), device=attention_mask.device, dtype=torch.float32)
                    return dummy_entropy, dummy_log_probs
                
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # Validate that we still have valid data after all preprocessing
                # Return dummy results for padded micro-batches instead of raising exceptions
                if input_ids_rmpad.shape[1] == 0:
                    print(f"[FORWARD] Warning: No valid tokens remaining after remove padding preprocessing (padded micro-batch) - returning dummy results. input_ids_rmpad shape: {input_ids_rmpad.shape}")
                    # Return dummy tensors with appropriate shapes for padded micro-batches
                    dummy_entropy = torch.zeros((batch_size, response_length), device=attention_mask.device, dtype=torch.float32) if calculate_entropy else None
                    dummy_log_probs = torch.zeros((batch_size, response_length), device=attention_mask.device, dtype=torch.float32)
                    return dummy_entropy, dummy_log_probs

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length:]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length:]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length:, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _safe_distributed_operation(self, operation_name, operation_func, timeout_seconds=300):
        """
        Safely execute a distributed operation with timeout and error handling.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute the operation
            timeout_seconds: Timeout in seconds (default 5 minutes)
            
        Returns:
            Result of operation_func or None if failed
        """
        if not (isinstance(self.actor_module, FSDP) and torch.distributed.is_initialized()):
            return operation_func()
            
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Distributed operation '{operation_name}' timed out after {timeout_seconds} seconds")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            result = operation_func()
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError as e:
            rank = torch.distributed.get_rank()
            print(f'[POLICY Worker {rank}] TIMEOUT: {e}')
            return None
        except Exception as e:
            rank = torch.distributed.get_rank()
            print(f'[POLICY Worker {rank}] ERROR in {operation_name}: {e}')
            return None
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
            
    # Note: cross-rank sync barrier removed from here; we synchronize once per mini-batch uniformly
                
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
            print(f'{BLUE}[COMPUTE_LOG_PROB] MICRO-BATCH INFO: Created {len(micro_batches)} micro-batch(es) from original batch size={data.batch.batch_size[0]} (multi-modal){RESET}')
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            print(f'{BLUE}[COMPUTE_LOG_PROB] MICRO-BATCH INFO: Created {len(micro_batches)} micro-batch(es) from original batch size={batch.batch_size[0] if hasattr(batch, "batch_size") else len(batch)} (dynamic){RESET}')
        else:
            micro_batches = batch.split(micro_batch_size)
            print(f'{BLUE}[COMPUTE_LOG_PROB] MICRO-BATCH INFO: Created {len(micro_batches)} micro-batch(es) from original batch size={batch.batch_size[0] if hasattr(batch, "batch_size") else len(batch)} (fixed size){RESET}')

        log_probs_lst = []
        entropy_lst = []
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                try:
                    entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
                    log_probs_lst.append(log_probs)
                    if calculate_entropy:
                        entropy_lst.append(entropy)
                except Exception as e:
                    print(f'[COMPUTE_LOG_PROB] Skipping failed micro-batch {micro_batch_idx}: {type(e).__name__}: {e}')
                    # Continue to next micro-batch without adding anything to the lists
                    continue

        # Check if we have any successful micro-batches
        if len(log_probs_lst) == 0:
            raise RuntimeError("All micro-batches failed during log probability computation")
            
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            # Note: If some micro-batches failed, the indices length might not match log_probs
            # This is a limitation when using dynamic batching with failed micro-batches
            if len(indices) != log_probs.size(0):
                print(f'[COMPUTE_LOG_PROB] WARNING: indices length ({len(indices)}) != log_probs size ({log_probs.size(0)}) - some micro-batches may have failed')
                # Truncate indices to match actual results
                indices = indices[:log_probs.size(0)]
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        rank = torch.distributed.get_rank()

        # make sure we are in training mode
        self.actor_module.train()
        print(f'[POLICY Worker {rank}] data in update_policy', len(data))
        print(f'[POLICY Worker {rank}] update_policy: This worker received {len(data)} samples (distributed from trainer)')
        print(f'{BLUE}[POLICY Worker {rank}] ORIGINAL BATCH INFO: Starting update_policy with total data size={len(data)} samples{RESET}')

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        ######
        # ADD: loss mask
        ######
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')

        ######
        # ADD: check multirun padding mask
        ######
        padded = 'no_padding_mask' in data.batch
        print(f'[POLICY Worker {rank}] update_policy: padded={padded}')
        if padded:
            # Include no_padding_mask in selection for later filtering
            select_keys.append('no_padding_mask')
            batch = data.select(batch_keys=select_keys).batch
        else:
            batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

    # Note: cross-rank dummy padding removed; we rely on per-mini-batch padding below

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
            raise NotImplementedError("need to be fixed for multi-turn code")
        ######
        # ADD: Splits `proto` into `self.config.update_steps_per_batch` chunks.
        #     proto_split is similar to `np.array_split`/`torch.tensor_split`, support inequally-sized chunks.
        #     note that self.config.train_batch_size has been injected in verl/workers/fsdp_workers.py
        ######
        if padded:
            # For padded data, bypass td_split and create a simple single-batch dataloader
            # since td_split seems to have issues with padded TensorDict structures
            num_mini_batches = max(1, self.config.train_batch_size // self.config.ppo_mini_batch_size)
            print(f'[POLICY Worker {rank}] update_policy: calculated num_mini_batches={num_mini_batches}')
            
            if self.config.ppo_mini_batch_size == 1:
                # Single batch case - just use the whole batch
                dataloader = [batch]
                print(f'[POLICY Worker {rank}] update_policy: using single batch approach, batch len={len(batch)}')
            else:
                # Multiple batches - use regular split
                dataloader = batch.split(self.config.ppo_mini_batch_size)
                print(f'[POLICY Worker {rank}] update_policy: using regular split with {len(dataloader)} batches')
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)
            print(f'[POLICY Worker {rank}] update_policy: created normal dataloader with {len(dataloader)} batches')

        metrics = {}
        total_valid_mini_batches = 0
        total_skipped_mini_batches = 0
        
        for epoch in range(self.config.ppo_epochs):
            print(f'[POLICY Worker {rank}] Starting epoch {epoch}/{self.config.ppo_epochs}')

            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                print(f'[POLICY Worker {rank}] mini_batch', len(mini_batch), 'batch_idx', batch_idx, 'len(dataloader)', len(dataloader))

                # Skip empty mini-batches early
                if len(mini_batch) == 0:
                    print(f'[POLICY Worker {rank}] Skipping completely empty mini-batch {batch_idx}')
                    total_skipped_mini_batches += 1
                    continue

                # Do NOT filter out padded samples at mini-batch level when padded=True.
                # Keeping dummy rows ensures identical micro-batch counts across ranks and avoids collective hangs.
                if padded and 'no_padding_mask' in mini_batch:
                    no_padding_mask = mini_batch['no_padding_mask']
                    print(f'[POLICY Worker {rank}] Padded mode: keeping all {len(mini_batch)} samples (real={int(no_padding_mask.sum().item())}) for cross-rank alignment')
                
                # Early validation: Check attention masks for all samples before any processing
                # In distributed padded mode, trust the distribution logic and use response_mask for validation
                # The ray_trainer ensures padded samples have minimal attention_mask (first token only) to prevent tensor errors
                if 'response_mask' in mini_batch and not padded:
                    # Only do response mask validation if we're NOT in distributed padded mode
                    # In padded mode, some samples intentionally have zero response masks as padding
                    response_mask = mini_batch['response_mask']
                    valid_samples = []
                    
                    # Use actual tensor size to avoid index errors
                    actual_batch_size = response_mask.shape[0]
                    for i in range(actual_batch_size):
                        sample_response_sum = response_mask[i].sum()
                        if sample_response_sum > 0:
                            valid_samples.append(i)
                        else:
                            print(f'[POLICY Worker {rank}] Sample {i} has no valid response tokens (response_mask sum = 0), skipping')

                    if len(valid_samples) == 0:
                        print(f'[POLICY Worker {rank}] All samples in mini-batch have invalid response masks, skipping mini-batch {batch_idx}')
                        total_skipped_mini_batches += 1
                        continue
                    elif len(valid_samples) < actual_batch_size:
                        print(f'[POLICY Worker {rank}] Early filtering mini-batch from {actual_batch_size} to {len(valid_samples)} valid samples')
                        # Create a new mini_batch with only valid samples
                        valid_indices = torch.tensor(valid_samples, device=response_mask.device)
                        mini_batch = mini_batch[valid_indices]
                        print(f'[POLICY Worker {rank}] After early response mask filtering: new len={len(mini_batch)}')
                    else:
                        print(f'[POLICY Worker {rank}] All {len(valid_samples)} samples in mini-batch are valid')
                elif padded:
                    # In distributed padded mode, samples were already carefully distributed by ray_trainer
                    # Some samples may have zero response masks as intentional padding - this is expected
                    # The attention masks are kept minimal (first token only) to prevent tensor errors
                    # Trust the distribution and process all samples
                    print(f'[POLICY Worker {rank}] Distributed padded mode: processing all {len(mini_batch)} samples (some may be intentionally padded)')

                    # Quick validation: Check for any samples with zero attention but non-zero response
                    if 'attention_mask' in mini_batch and 'response_mask' in mini_batch:
                        attention_mask = mini_batch['attention_mask']
                        response_mask = mini_batch['response_mask']
                        
                        # Use actual tensor size instead of len(mini_batch) to avoid index errors
                        actual_batch_size = attention_mask.shape[0]
                        for i in range(actual_batch_size):
                            attn_sum = attention_mask[i].sum().item()
                            resp_sum = response_mask[i].sum().item()
                            
                            # print('attention_mask[i]', attention_mask[i][-32:])
                            # print('response_mask[i]', response_mask[i][-32:])
                            # if resp_sum > 0 and attn_sum == 0:
                            #     print(f'[POLICY Worker {rank}] 🚨 CRITICAL: Sample {i} has {resp_sum} response tokens but ZERO attention mask!')
                            # elif resp_sum == 0 and attn_sum == 0:
                            #     print(f'[POLICY Worker {rank}] ✓ Sample {i}: properly padded (both masks zero)')
                            # elif resp_sum > 0 and attn_sum > 0:
                            #     print(f'[POLICY Worker {rank}] ✓ Sample {i}: valid sample ({resp_sum} response, {attn_sum} attention tokens)')
                else:
                    print(f'[POLICY Worker {rank}] No response mask validation needed for {len(mini_batch)} samples')

                # Count this as a valid mini-batch that will be processed
                total_valid_mini_batches += 1
                
                # For DP actor policy updates, always use the configured micro-batch size
                # Dynamic batching should only apply to compute operations (log prob), not training batch sizes
                # This ensures we respect the user's configured ppo_micro_batch_size for gradient computation
                use_fixed_micro_batch_size = True
                
                # For distributed training stability, prefer larger micro-batches to reduce collective operations
                if isinstance(self.actor_module, FSDP) and torch.distributed.is_initialized():
                    # Increase micro-batch size to reduce NCCL collective operation frequency
                    effective_micro_batch_size = max(self.config.ppo_micro_batch_size_per_gpu, 2)
                    print(f'[POLICY Worker {rank}] FSDP detected: using effective_micro_batch_size={effective_micro_batch_size} to reduce collective ops')
                else:
                    effective_micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
                
                if use_fixed_micro_batch_size:
                    if effective_micro_batch_size == 1:
                        print(f'{GREEN}[POLICY Worker {rank}] effective_micro_batch_size=1, processing entire mini_batch directly (len={len(mini_batch)}){RESET}')
                        micro_batches = [mini_batch]
                        print(f'{BLUE}[POLICY Worker {rank}] MICRO-BATCH INFO: Created {len(micro_batches)} micro-batch(es) from original mini-batch size={len(mini_batch)}{RESET}')
                    else:
                        # Use fixed micro-batch sizes
                        if has_multi_modal_inputs:
                            self.gradient_accumulation = max(1, self.config.ppo_mini_batch_size // effective_micro_batch_size)
                            num_micro_batches = mini_batch.batch.batch_size[0] // effective_micro_batch_size
                            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                            print(f'{BLUE}[POLICY Worker {rank}] MICRO-BATCH INFO: Created {len(micro_batches)} micro-batch(es) from original mini-batch size={mini_batch.batch.batch_size[0]} (multi-modal){RESET}')
                        elif padded:
                            # For padded batches, ensure mini_batch length is a multiple of effective_micro_batch_size
                            # by padding with dummy samples that yield zero loss. This prevents a smaller last micro-batch
                            # which can cause collective op shape mismatches and hangs.
                            remainder = len(mini_batch) % effective_micro_batch_size
                            if remainder != 0:
                                pad_count = effective_micro_batch_size - remainder
                                pad_token_id = getattr(self.config, 'pad_token_id', 0)
                                print(f'{GREEN}[POLICY Worker {rank}] Padding mini-batch from {len(mini_batch)} to {len(mini_batch)+pad_count} to align micro-batch size {effective_micro_batch_size}{RESET}')

                                # Build padded tensors per key using first row as template
                                new_tensors = {}
                                try:
                                    # Works for TensorDict-like objects
                                    items_iter = mini_batch.items()
                                except Exception:
                                    # Fallback assume dict-like
                                    items_iter = mini_batch.items()
                                for k, v in items_iter:
                                    if not torch.is_tensor(v):
                                        new_tensors[k] = v
                                        continue
                                    tmpl = v[0:1].clone()
                                    if k in ('response_mask', 'no_padding_mask'):
                                        tmpl.zero_()
                                    elif k in ('advantages', 'old_log_probs', 'ref_log_prob'):
                                        tmpl.zero_()
                                    elif k == 'attention_mask':
                                        tmpl.zero_()
                                        if tmpl.numel() > 0:
                                            tmpl[..., 0] = 1
                                    elif k in ('input_ids', 'responses'):
                                        tmpl.fill_(pad_token_id)
                                    elif k == 'position_ids':
                                        if tmpl.ndim >= 2:
                                            tmpl[..., 1:] = 0
                                    reps = [pad_count] + [1] * (tmpl.dim() - 1)
                                    pad_block = tmpl.repeat(*reps)
                                    new_tensors[k] = torch.cat([v, pad_block], dim=0)

                                # Maintain or create real-sample mask
                                device = mini_batch['input_ids'].device if 'input_ids' in mini_batch else torch.device('cuda')
                                if '__real_sample_mask__' in mini_batch:
                                    rmask = mini_batch['__real_sample_mask__']
                                else:
                                    rmask = torch.ones(len(mini_batch), dtype=torch.bool, device=device)
                                rmask = torch.cat([rmask, torch.zeros(pad_count, dtype=torch.bool, device=device)], dim=0)
                                new_tensors['__real_sample_mask__'] = rmask

                                # Rewrap as TensorDict if available, else keep dict-like
                                try:
                                    if hasattr(mini_batch, 'batch_size') and hasattr(mini_batch, 'keys'):
                                        from tensordict import TensorDict  # type: ignore
                                        mini_batch = TensorDict(new_tensors, batch_size=[len(rmask)])
                                    else:
                                        mini_batch = new_tensors
                                except Exception as rewrap_err:
                                    print(f'[POLICY Worker {rank}] Warning: Failed to rewrap padded mini-batch ({rewrap_err}); using dict')
                                    mini_batch = new_tensors

                            # Now split into equal-sized micro-batches
                            micro_batches = [mini_batch[i:i+effective_micro_batch_size] for i in range(0, len(mini_batch), effective_micro_batch_size)]
                            print(f'{GREEN}[POLICY Worker {rank}] Using regular split for padded batch: {len(mini_batch)} samples -> {len(micro_batches)} micro-batches of size {effective_micro_batch_size} (no samples ignored; last padded to full){RESET}')
                            print(f'{BLUE}[POLICY Worker {rank}] MICRO-BATCH INFO: Created {len(micro_batches)} micro-batch(es) from original mini-batch size={len(mini_batch)} (padded){RESET}')
                        else:   
                            self.gradient_accumulation = max(1, self.config.ppo_mini_batch_size // effective_micro_batch_size)
                            # split batch into micro_batches
                            micro_batches = mini_batch.split(effective_micro_batch_size)
                            print(f'{GREEN}[POLICY Worker {rank}] Using fixed micro-batch size with regular split: {len(mini_batch)} samples -> {len(micro_batches)} micro-batches of size {effective_micro_batch_size}{RESET}')
                            print(f'{BLUE}[POLICY Worker {rank}] MICRO-BATCH INFO: Created {len(micro_batches)} micro-batch(es) from original mini-batch size={len(mini_batch)} (regular){RESET}')
                # Note: Dynamic batching removed for actor updates to ensure consistent micro-batch sizes
                # Dynamic batching is still used in compute_log_prob for memory efficiency during inference

                self.actor_optimizer.zero_grad()

                #######
                # ADD: For unbias grad_gcc, see MODIFY in below for more info.
                #######
                if not self.config.use_dynamic_bsz:
                    from warnings import warn
                    warn("Using dynamic bsz is highly recomended for multiturn since there will be padding samples")
                mini_batch_token_nums = mini_batch['response_mask'].sum()
                
                # Track if any micro-batch was successfully processed to ensure proper FSDP state
                successful_micro_batches = 0
                failed_micro_batches = 0

                for data_idx, data in enumerate(micro_batches):
                    print(f'[POLICY Worker {rank}] micro_batch {data_idx}/{len(micro_batches)} with micro batch size {len(data)}')

                    try:
                        # Early validation of micro-batch data
                        required_keys = ['response_mask', 'old_log_probs', 'advantages', 'input_ids', 'responses', 'attention_mask']
                        missing_keys = [key for key in required_keys if key not in data]
                        if missing_keys:
                            raise ValueError(f"Micro-batch missing required keys: {missing_keys}")
                        
                        # Check for empty tensors that would cause issues
                        if data['input_ids'].numel() == 0:
                            raise ValueError("Empty input_ids in micro-batch")
                        if data['responses'].numel() == 0:
                            raise ValueError("Empty responses in micro-batch")
                            
                        # Check for tensor shape mismatches
                        batch_size = data['input_ids'].shape[0]
                        if data['responses'].shape[0] != batch_size:
                            raise ValueError(f"Batch size mismatch: input_ids {data['input_ids'].shape[0]} vs responses {data['responses'].shape[0]}")
                        if data['attention_mask'].shape[0] != batch_size:
                            raise ValueError(f"Batch size mismatch: input_ids {data['input_ids'].shape[0]} vs attention_mask {data['attention_mask'].shape[0]}")
                            
                        #######
                        # MODIFIED: use loss_mask directly
                        #######
                        response_mask = data['response_mask']
                        old_log_prob = data["old_log_probs"]
                        advantages = data["advantages"]
                        
                        # Check if this micro-batch has any valid response tokens
                        # In distributed training, we still need to do forward pass even for padded micro-batches
                        micro_batch_has_tokens = response_mask.sum().item() > 0
                        if not micro_batch_has_tokens:
                            print(f'[POLICY Worker {rank}] Micro-batch has zero response tokens - will do forward pass but skip loss computation')

                        clip_ratio = self.config.clip_ratio
                        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        # Debug tensor shapes before policy loss computation
                        print(f'[POLICY Worker {rank}] Tensor shapes before policy loss:')
                        print(f'[POLICY Worker {rank}]   old_log_prob shape: {old_log_prob.shape}')
                        print(f'[POLICY Worker {rank}]   advantages shape: {advantages.shape}')
                        print(f'[POLICY Worker {rank}]   response_mask shape: {response_mask.shape}')
                        print(f'[POLICY Worker {rank}]   responses shape: {data["responses"].shape if "responses" in data else "N/A"}')
                        print(f'[POLICY Worker {rank}]   input_ids shape: {data["input_ids"].shape if "input_ids" in data else "N/A"}')
                        
                        # all return: (bsz, response_length)
                        calculate_entropy = False
                        if entropy_coeff != 0:
                            calculate_entropy = True
                    
                        # Forward pass - this may raise exceptions for problematic micro-batches
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                        
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                        
                        # Check for non-finite policy loss components early
                        if not torch.isfinite(pg_loss).all():
                            print(f'[POLICY Worker {rank}] WARNING: Non-finite policy loss detected in micro-batch {data_idx}: pg_loss={pg_loss}')
                            print(f'[POLICY Worker {rank}] log_prob stats: min={log_prob.min()}, max={log_prob.max()}, mean={log_prob.mean()}')
                            print(f'[POLICY Worker {rank}] old_log_prob stats: min={old_log_prob.min()}, max={old_log_prob.max()}, mean={old_log_prob.mean()}')
                            print(f'[POLICY Worker {rank}] advantages stats: min={advantages.min()}, max={advantages.max()}, mean={advantages.mean()}')
                            # Clip the policy loss to a reasonable range
                            max_policy_loss = self.config.get('max_policy_loss', 5.0)
                            pg_loss = torch.clamp(pg_loss, min=-max_policy_loss, max=max_policy_loss)
                            print(f'[POLICY Worker {rank}] Clipped policy loss: {pg_loss}')
                        
                        print(f'[POLICY Worker {rank}] Done computing policy loss')

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            
                            # Check for non-finite entropy loss
                            if not torch.isfinite(entropy_loss).all():
                                print(f'[POLICY Worker {rank}] WARNING: Non-finite entropy loss detected in micro-batch {data_idx}: entropy_loss={entropy_loss}')
                                print(f'[POLICY Worker {rank}] entropy stats: min={entropy.min()}, max={entropy.max()}, mean={entropy.mean()}')
                                # Clip the entropy loss to a reasonable range
                                max_entropy_loss = self.config.get('max_entropy_loss', 5.0)
                                entropy_loss = torch.clamp(entropy_loss, min=-max_entropy_loss, max=max_entropy_loss)
                                print(f'[POLICY Worker {rank}] Clipped entropy loss: {entropy_loss}')

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                            print(f'[POLICY Worker {rank}] Done computing entropy loss')
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = data["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                            
                            # Check for non-finite KL loss
                            if not torch.isfinite(kl_loss).all():
                                print(f'[POLICY Worker {rank}] WARNING: Non-finite KL loss detected in micro-batch {data_idx}: kl_loss={kl_loss}')
                                print(f'[POLICY Worker {rank}] kld stats: min={kld.min()}, max={kld.max()}, mean={kld.mean()}')
                                print(f'[POLICY Worker {rank}] ref_log_prob stats: min={ref_log_prob.min()}, max={ref_log_prob.max()}, mean={ref_log_prob.mean()}')
                                # Clip the KL loss to a reasonable range
                                max_kl_loss = self.config.get('max_kl_loss', 5.0)
                                kl_loss = torch.clamp(kl_loss, min=-max_kl_loss, max=max_kl_loss)
                                print(f'[POLICY Worker {rank}] Clipped KL loss: {kl_loss}')

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef
                            print(f'[POLICY Worker {rank}] Done computing KL loss')

                        acc_grad_mode = grad_acc_mode(loss_agg_mode)
                        if acc_grad_mode == "seq":
                            # Use real-sample counts when available to avoid dilution from dummy padding
                            if ('__real_sample_mask__' in data) and ('__real_sample_mask__' in mini_batch):
                                real_micro = int(data['__real_sample_mask__'].sum().item())
                                real_mini = int(mini_batch['__real_sample_mask__'].sum().item())
                                if real_mini == 0:
                                    # Zero-loss backward to keep collectives aligned across ranks
                                    print(f'[POLICY Worker {rank}] Zero real samples in micro-batch; performing zero-loss backward for seq acc mode')
                                    (policy_loss.sum() * 0.0).backward()
                                    successful_micro_batches += 1
                                    continue
                                scale = real_micro / real_mini
                            else:
                                scale = len(data) / len(mini_batch)
                            loss = policy_loss * scale  # self.gradient_accumulation
                        elif acc_grad_mode == "token":
                            # weights by token nums, note that we want to apply a simple scalar, or the compute-graph will be extremely large.
                            # Add safety check to prevent division by zero
                            mini_batch_tokens = mini_batch_token_nums.item()
                            if mini_batch_tokens == 0:
                                # Zero-loss backward to keep collectives aligned across ranks
                                print(f'[POLICY Worker {rank}] Zero tokens in mini-batch; performing zero-loss backward for token acc mode')
                                (policy_loss.sum() * 0.0).backward()
                                successful_micro_batches += 1
                                continue
                            loss = policy_loss * (response_mask.sum().item() / mini_batch_tokens)
                        else:
                            raise NotImplementedError(f"Unsupported acc_grad_mode: {acc_grad_mode}")

                        # Apply loss clipping to prevent NaN and extreme values
                        max_loss_value = self.config.get('max_loss_value', 10.0)  # Default max loss value
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            print(f'[POLICY Worker {rank}] WARNING: Non-finite loss detected in micro-batch {data_idx}, clipping to finite values. Original loss: {loss}')
                            # Replace NaN and inf with the max loss value
                            loss = torch.where(torch.isfinite(loss), loss, torch.tensor(max_loss_value, device=loss.device, dtype=loss.dtype))
                            print(f'[POLICY Worker {rank}] Clipped loss: {loss}')
                        
                        # Apply additional loss magnitude clipping
                        loss = torch.clamp(loss, min=-max_loss_value, max=max_loss_value)
                        
                        # Final check - if still non-finite after clipping, skip
                        if not torch.isfinite(loss).all():
                            print(f'[POLICY Worker {rank}] WARNING: Loss still non-finite after clipping in micro-batch {data_idx}, skipping backward pass. Loss: {loss}')
                            failed_micro_batches += 1
                            continue

                        loss.backward()
                        print(f'[POLICY Worker {rank}] Done loss back propagation')
                        successful_micro_batches += 1

                        data_metrics = {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                        append_to_dict(metrics, data_metrics)

                    except Exception as e:
                        print(f'[POLICY Worker {rank}] ERROR in micro-batch {data_idx}/{len(micro_batches)}: {type(e).__name__}: {e}')
                        print(f'[POLICY Worker {rank}] Skipping problematic micro-batch {data_idx} and continuing with next one...')
                        
                        # Clear any accumulated gradients from failed micro-batch
                        try:
                            for param in self.actor_module.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
                        except Exception as cleanup_error:
                            print(f'[POLICY Worker {rank}] Failed to cleanup gradients after micro-batch error: {cleanup_error}')
                        
                        failed_micro_batches += 1
                        
                        # Try to free GPU memory after error
                        try:
                            torch.cuda.empty_cache()
                        except Exception as cuda_error:
                            print(f'[POLICY Worker {rank}] Failed to clear CUDA cache: {cuda_error}')
                        
                        continue

                print(f'[POLICY Worker {rank}] Micro-batch processing summary: {successful_micro_batches} successful, {failed_micro_batches} failed out of {len(micro_batches)} total')

                # Coordinate optimizer steps across all workers to prevent NCCL hanging
                # All workers must participate in the same collective operations
                if isinstance(self.actor_module, FSDP) and torch.distributed.is_initialized():
                    try:
                        # Step 1: Coordinate whether any worker has successful micro-batches
                        has_successful = torch.tensor(1.0 if successful_micro_batches > 0 else 0.0, device='cuda')
                        
                        def coordinate_successful():
                            torch.distributed.all_reduce(has_successful, op=torch.distributed.ReduceOp.MAX)
                            return has_successful.item()
                        
                        result = self._safe_distributed_operation("coordinate_successful", coordinate_successful, timeout_seconds=60)
                        if result is None:
                            print(f'[POLICY Worker {rank}] Failed to coordinate successful micro-batches - assuming local decision')
                            any_worker_has_successful = successful_micro_batches > 0
                        else:
                            any_worker_has_successful = result > 0.5
                        
                        # Step 2: If any worker has successful batches, check for NaN gradients across all workers
                        has_nan_grad = False
                        if any_worker_has_successful and successful_micro_batches > 0:
                            for name, param in self.actor_module.named_parameters():
                                if param.grad is not None and torch.isnan(param.grad).any():
                                    print(f'[POLICY] Worker {rank} WARNING: NaN gradient detected in {name}')
                                    has_nan_grad = True
                                    break
                        
                        # Step 3: Coordinate NaN gradient status across all workers
                        nan_grad_tensor = torch.tensor(1.0 if has_nan_grad else 0.0, device='cuda')
                        
                        def coordinate_nan_gradients():
                            torch.distributed.all_reduce(nan_grad_tensor, op=torch.distributed.ReduceOp.MAX)
                            return nan_grad_tensor.item()
                        
                        result = self._safe_distributed_operation("coordinate_nan_gradients", coordinate_nan_gradients, timeout_seconds=60)
                        if result is None:
                            print(f'[POLICY Worker {rank}] Failed to coordinate NaN gradients - assuming local decision')
                            any_worker_has_nan = has_nan_grad
                        else:
                            any_worker_has_nan = result > 0.5
                        
                        print(f'[POLICY Worker {rank}] Coordination: local_successful={successful_micro_batches > 0}, any_worker_successful={any_worker_has_successful}, local_nan={has_nan_grad}, any_worker_nan={any_worker_has_nan}')
                        
                        # Step 4: All workers make the same decision
                        if any_worker_has_successful and not any_worker_has_nan:
                            # Safe to proceed with optimizer step - all workers participate
                            if successful_micro_batches > 0:
                                print(f'[POLICY Worker {rank}] Performing optimizer step after {successful_micro_batches} successful micro-batches')
                                grad_norm = self._optimizer_step()
                            else:
                                print(f'[POLICY Worker {rank}] Participating in optimizer step coordination (no local successes but others do)')
                                # Zero gradients but still participate in collective operations
                                self.actor_optimizer.zero_grad()
                                grad_norm = self._optimizer_step()
                            data = {"actor/grad_norm": grad_norm.detach().item() if hasattr(grad_norm, 'detach') else grad_norm}
                        else:
                            # Skip optimizer step - either no successes anywhere or NaN gradients detected
                            if not any_worker_has_successful:
                                print(f'[POLICY Worker {rank}] All workers skipping optimizer step (no successful micro-batches anywhere)')
                            else:
                                print(f'[POLICY Worker {rank}] All workers skipping optimizer step (NaN gradients detected)')
                            self.actor_optimizer.zero_grad()
                            data = {"actor/grad_norm": 0.0}
                            
                    except Exception as coordination_error:
                        print(f'[POLICY Worker {rank}] CRITICAL: Distributed coordination failed: {coordination_error}')
                        print(f'[POLICY Worker {rank}] Falling back to local decision to avoid hanging')
                        # Fallback: make local decision to avoid hanging the entire job
                        if successful_micro_batches > 0:
                            try:
                                self.actor_optimizer.zero_grad()  # Clear any accumulated gradients
                                data = {"actor/grad_norm": 0.0}
                                print(f'[POLICY Worker {rank}] Fallback: cleared gradients due to coordination failure')
                            except Exception as fallback_error:
                                print(f'[POLICY Worker {rank}] Fallback failed: {fallback_error}')
                                data = {"actor/grad_norm": 0.0}
                        else:
                            data = {"actor/grad_norm": 0.0}
                        
                else:
                    # Non-distributed case - original logic
                    if successful_micro_batches > 0:
                        try:
                            grad_norm = self._optimizer_step()
                            data = {"actor/grad_norm": grad_norm.detach().item() if hasattr(grad_norm, 'detach') else grad_norm}
                        except Exception as optimizer_error:
                            print(f'[POLICY Worker {rank}] ERROR during optimizer step: {type(optimizer_error).__name__}: {optimizer_error}')
                            print(f'[POLICY Worker {rank}] Clearing gradients and continuing...')
                            try:
                                self.actor_optimizer.zero_grad()
                            except Exception as cleanup_error:
                                print(f'[POLICY Worker {rank}] Failed to clear gradients: {cleanup_error}')
                            data = {"actor/grad_norm": 0.0}
                    else:
                        print(f'[POLICY Worker {rank}] Skipping optimizer step (no successful micro-batches)')
                        try:
                            self.actor_optimizer.zero_grad()
                        except Exception as cleanup_error:
                            print(f'[POLICY Worker {rank}] Failed to clear gradients: {cleanup_error}')
                        data = {"actor/grad_norm": 0.0}
                    
                append_to_dict(metrics, data)
                
                # Uniform cross-rank synchronization once per mini-batch to keep ranks aligned
                # Use a timeout to prevent infinite hanging
                if isinstance(self.actor_module, FSDP) and torch.distributed.is_initialized():
                    def barrier_sync():
                        torch.distributed.barrier()
                        return True
                    
                    result = self._safe_distributed_operation("barrier_sync", barrier_sync, timeout_seconds=120)
                    if result is not None:
                        print(f'[POLICY Worker {rank}] Successfully synchronized after mini-batch {batch_idx}')
                    else:
                        print(f'[POLICY Worker {rank}] WARNING: Failed to synchronize after mini-batch {batch_idx} - continuing anyway')
                        # Don't try to recover here as it's complex - let subsequent operations handle the state
                
        print(f'[POLICY Worker {rank}] Processing summary: {total_valid_mini_batches} valid mini-batches, {total_skipped_mini_batches} skipped mini-batches')

        # Ensure we always return valid metrics even if no processing occurred
        if total_valid_mini_batches == 0:
            print(f'[POLICY Worker {rank}] Warning: No mini-batches were successfully processed - this worker had no valid data')
            # Add default metrics to prevent downstream issues
            if "actor/pg_loss" not in metrics:
                metrics["actor/pg_loss"] = 0.0
            if "actor/grad_norm" not in metrics:
                metrics["actor/grad_norm"] = 0.0
        
        # Ensure all gradients are cleared for distributed consistency
        try:
            self.actor_optimizer.zero_grad()
        except Exception as cleanup_error:
            print(f'[POLICY Worker {rank}] Failed to clear gradients at end: {cleanup_error}')
            
        # Final cleanup to prevent memory leaks
        try:
            torch.cuda.empty_cache()
        except Exception as cuda_error:
            print(f'[POLICY Worker {rank}] Failed to clear CUDA cache at end: {cuda_error}')
            
        print(f'[POLICY Worker {rank}] Done update policy')
        return metrics