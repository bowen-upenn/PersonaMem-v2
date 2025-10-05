# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        def parse_messages(item):
            item = series_to_item(item)
            # If messages are stored as JSON strings, parse them back to Python objects
            if isinstance(item, str):
                import json
                return json.loads(item)
            return item
        
        self.messages = self.dataframe[self.messages_key].apply(parse_messages).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None

    def __len__(self):
        return len(self.messages)

    def _process_message_tokens(
        self,
        messages: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        if start_idx > 0:
            prev_applied_text = self._apply_chat_template_with_thinking_removal(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self._apply_chat_template_with_thinking_removal(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                )

        else:
            prev_applied_text = ""

        cur_applied_text = self._apply_chat_template_with_thinking_removal(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )
        # Get tokens for the current message only - optimized to avoid redundant tokenization
        if is_assistant:
            # Fast encode only the incremental parts
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            message_only_text = cur_applied_text[len(prev_applied_text_w_generation_prompt) :]
            
            # Use faster encoding without special tokens
            generation_prompt_tokens = self.tokenizer(
                generation_prompt_text, 
                add_special_tokens=False, 
                return_attention_mask=False,
                return_token_type_ids=False
            )['input_ids']
            
            _message_tokens = self.tokenizer(
                message_only_text, 
                add_special_tokens=False,
                return_attention_mask=False, 
                return_token_type_ids=False
            )['input_ids']
            
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * len(generation_prompt_tokens) + [1] * len(_message_tokens)
        else:
            # For non-assistant messages, use fast encoding
            message_only_text = cur_applied_text[len(prev_applied_text) :]
            message_tokens = self.tokenizer(
                message_only_text, 
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )['input_ids']
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: List[int],
        concat_loss_mask: List[int],
        concat_attention_mask: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list)
        ):
            # Decode both versions to see the actual text differences
            full_text = self.tokenizer.decode(full_tokens_list)
            concat_text = self.tokenizer.decode(concat_tokens)
            
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
                f"length: {len(concat_tokens)}. Using concatenated version."
            )
            
            # Enable this debug logging if needed to see exact differences
            # logging.debug(f"Full text: {repr(full_text[:500])}...")
            # logging.debug(f"Concat text: {repr(concat_text[:500])}...")
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )

        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def _remove_blank_thinking_tags(self, text: str) -> str:
        """
        Remove blank thinking tags from Qwen3 chat template output for SFT.
        This prevents token mismatch issues when no thinking content is present.
        """
        import re
        # Qwen3 adds <think>\n\n</think>\n even when enable_thinking=False
        # Remove these blank thinking blocks for SFT training
        return re.sub(r'<think>\s*</think>\s*', '', text, flags=re.MULTILINE | re.DOTALL)

    def _apply_chat_template_with_thinking_removal(self, messages, tools=None, **kwargs):
        """
        Apply chat template and remove blank thinking tags for SFT training.
        """
        # For tokenized output, get text first, clean it, then tokenize
        if kwargs.get('tokenize', False):
            text_kwargs = kwargs.copy()
            text_kwargs['tokenize'] = False
            text_kwargs.pop('return_tensors', None)
            
            text_result = self.tokenizer.apply_chat_template(messages, tools=tools, **text_kwargs)
            text_result = self._remove_blank_thinking_tags(text_result)
            
            if kwargs.get('return_tensors') == 'pt':
                return torch.tensor(self.tokenizer.encode(text_result, add_special_tokens=False)).unsqueeze(0)
            else:
                return self.tokenizer.encode(text_result, add_special_tokens=False)
        else:
            # For text output, apply template and clean
            result = self.tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
            return self._remove_blank_thinking_tags(result)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None

        messages[1]['content'] = messages[0]['content'] + ' ' + messages[1]['content']
        messages = messages[1:]

        if self.tools is not None:
            tools = json.loads(self.tools[item])
        else:
            tools = None
        # First, get the full conversation tokens
        try:
            full_tokens = self._apply_chat_template_with_thinking_removal(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}"
            )
            raise

        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
