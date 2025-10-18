# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

# ANSI color codes for green text
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def _ensure_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        expr = value.replace(" ", "")
        if expr.isdigit():
            return int(expr)
        if "*" in expr:
            result = 1
            for part in expr.split("*"):
                part = part.strip()
                if not part:
                    continue
                if not part.isdigit():
                    raise ValueError(f"Cannot convert segment {part!r} to int")
                result *= int(part)
            return result
    raise TypeError(f"Cannot convert {value!r} to int")


def _flatten_leading_dims(tensor: torch.Tensor, keep_last_dims: int = 1) -> torch.Tensor:
    """Collapse any leading batch dimensions while keeping the trailing dims intact."""
    if tensor is None:
        return None
    if tensor.dim() <= keep_last_dims:
        return tensor
    return tensor.reshape(-1, *tensor.shape[-keep_last_dims:])


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
        input_ids_raw = micro_batch["input_ids"]
        responses = micro_batch["responses"]
        
        # Log correct batch information before any tensor manipulation
        if responses.numel() == 0:
            print(f"{YELLOW}[ACTOR] Skip empty micro-batch (responses) on device {next(self.actor_module.parameters()).device}{RESET}")
            return None, None
        
        responses_flat = _flatten_leading_dims(responses, keep_last_dims=1)
        response_length = responses_flat.size(-1)
        batch_size_from_responses = responses_flat.size(0)
        
        # Log the actual batch size and sequence length correctly
        print(f'[ACTOR] Forwarding micro-batch of size torch.Size([{batch_size_from_responses}, {input_ids_raw.shape[-1]}]) on device {next(self.actor_module.parameters()).device}')
        print(f'[ACTOR] Micro-batch details: {batch_size_from_responses} samples, seq_len={input_ids_raw.shape[-1]}, original_shape={input_ids_raw.shape}')
        
        responses = responses_flat
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            if input_ids.numel() == 0:
                print(f"{YELLOW}[ACTOR] Skip empty micro-batch (input_ids) on device {next(self.actor_module.parameters()).device}{RESET}")
                return None, None
            input_ids = _flatten_leading_dims(input_ids, keep_last_dims=1)
            print('[ACTOR] input_ids shape:', input_ids.shape)
            try:
                input_ids = input_ids.reshape(batch_size_from_responses, -1)
            except RuntimeError as err:
                raise RuntimeError(f"Failed to reshape input_ids from shape {input_ids.shape} to batch size {batch_size_from_responses}") from err
            batch_size, seqlen = input_ids.shape
            attention_mask = _flatten_leading_dims(micro_batch["attention_mask"], keep_last_dims=1)
            attention_mask = attention_mask.reshape(batch_size, -1)
            position_ids = micro_batch["position_ids"]
            position_ids = position_ids.squeeze()
            if position_ids.dim() > 3:  # Allow 3D for qwen2vl mrope, but squeeze higher dims
                position_ids = position_ids.squeeze()
            keep_last_dims = position_ids.dim() - 1 if position_ids.dim() > 1 else 1
            position_ids = _flatten_leading_dims(position_ids, keep_last_dims=keep_last_dims)
            try:
                if position_ids.dim() > 1:
                    position_ids = position_ids.reshape(batch_size, *position_ids.shape[1:])
                else:
                    position_ids = position_ids.reshape(batch_size, -1)
            except RuntimeError as err:
                raise RuntimeError(f"Failed to reshape position_ids from shape {position_ids.shape} to batch size {batch_size}") from err
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

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
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

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
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, responses)
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

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
            micro_batch_indices = [None] * len(micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = _ensure_int(data.meta_info["max_token_len"]) * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            micro_batch_indices = indices
        else:
            micro_batches = batch.split(micro_batch_size)
            micro_batch_indices = [None] * len(micro_batches)

        log_probs_lst = []
        entropy_lst = []
        gathered_indices = []
        for micro_batch, micro_idx in zip(micro_batches, micro_batch_indices):
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            if log_probs is None:
                continue
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if use_dynamic_bsz and micro_idx is not None:
                gathered_indices.extend(micro_idx)

        if not log_probs_lst:
            raise RuntimeError("No valid micro-batches available to compute log probabilities.")

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            assert len(gathered_indices) == log_probs.size(0), f"{len(gathered_indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(gathered_indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Log initial batch information with better details
        if hasattr(data, 'batch') and hasattr(data.batch, 'batch_size'):
            initial_batch_size = data.batch.batch_size[0] if len(data.batch.batch_size) > 0 else 0
            print(f"{GREEN}[POLICY Worker {rank}] Initial batch size: {initial_batch_size}{RESET}")
            
            # Try to determine if this is a final turn or memory turn
            if hasattr(data, 'non_tensor_batch') and 'is_final' in data.non_tensor_batch:
                is_final_turn = data.non_tensor_batch['is_final']
                turn_type = "FINAL (response generation)" if is_final_turn else "MEMORY (chunk processing)"
                print(f"{GREEN}[POLICY Worker {rank}] Turn type: {turn_type}{RESET}")
        
        # Gather total batch size across all GPUs
        if torch.distributed.is_initialized():
            local_batch_size = len(data.batch) if hasattr(data, 'batch') else 0
            total_samples = torch.tensor([local_batch_size], device='cuda', dtype=torch.int64)
            torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)
            if rank == 0:
                print(f"{GREEN}[POLICY] Total samples across all {torch.distributed.get_world_size()} GPUs: {total_samples.item()}{RESET}")
                print(f"{GREEN}[POLICY] Note: This represents ALL TURNS (memory + final) combined across all samples{RESET}")

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
        if padded:
            from recurrent.utils import indexing_proto
            # batch is a TensorDict here, we need a DataProto for code reusing.
            proto = data.select(batch_keys=select_keys)
            # we need to drop empty samples, since they will implact sequence-level averaging loss
            batch = indexing_proto(proto, data.batch['no_padding_mask']).batch
        else:
            batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

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
            from recurrent.utils import td_split
            dataloader = td_split(batch, self.config.train_batch_size // self.config.ppo_mini_batch_size)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        
        # Log total batch distribution across all GPUs at the start
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # Gather sample counts from all GPUs for the first mini-batch
            if hasattr(batch, "batch_size"):
                total_samples = int(batch.batch_size[0]) if len(batch.batch_size) > 0 else 1
            else:
                try:
                    total_samples = len(batch)
                except TypeError:
                    total_samples = 1
            
            # Gather all sample counts
            sample_counts = torch.tensor([total_samples], device='cuda')
            all_sample_counts = [torch.zeros_like(sample_counts) for _ in range(world_size)]
            torch.distributed.all_gather(all_sample_counts, sample_counts)
            
            if rank == 0:
                all_counts = [int(x.item()) for x in all_sample_counts]
                total_across_gpus = sum(all_counts)
                print(f"{GREEN}[TOTAL BATCH INFO] Total samples across {world_size} GPUs: {total_across_gpus}")
                for i, count in enumerate(all_counts):
                    print(f"  GPU {i}: {count} samples")
                if len(set(all_counts)) > 1:
                    print(f"{YELLOW}WARNING: Uneven distribution detected! Samples per GPU: {all_counts}{RESET}")
                print(f"{RESET}")
        
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                # split batch into micro_batches
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                if hasattr(mini_batch, "batch_size"):
                    samples_per_gpu = int(mini_batch.batch_size[0]) if len(mini_batch.batch_size) > 0 else 1
                else:
                    try:
                        samples_per_gpu = len(mini_batch)
                    except TypeError:
                        samples_per_gpu = 1
                print(f"{GREEN}[POLICY Worker {rank}] Mini-batch contains {samples_per_gpu} samples before micro batching{RESET}")
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len_cfg = _ensure_int(self.config.ppo_max_token_len_per_gpu)
                    max_token_len = max_token_len_cfg * self.ulysses_sequence_parallel_size

                    if hasattr(mini_batch, "batch_dims") and mini_batch.batch_dims == 0:
                        mini_batch = mini_batch.unsqueeze(0)
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    print(f"{GREEN}[POLICY Worker {rank}] Using dynamic micro-batching: {samples_per_gpu} samples -> {len(micro_batches)} micro-batches (token-based){RESET}")
                    
                    # Log micro-batch sizes for better understanding
                    micro_batch_sizes = [mb.batch_size[0] if hasattr(mb, 'batch_size') and len(mb.batch_size) > 0 else len(mb) for mb in micro_batches]
                    print(f"{GREEN}[POLICY Worker {rank}] Micro-batch sizes: {micro_batch_sizes} (total samples: {sum(micro_batch_sizes)}){RESET}")
                    ###### NOTE: rearrange_micro_batches will generate max(num_micro for num_micro in all_dp_workers) and torch.distributed.all_reduce is called
                    ###### When debugging, set a breakpoint after here, or code will be stuck here.
                else:
                    ######
                    # ADD: I will not disable dynamic_bsz, just in case, use proto_split to get num_micro_batches
                    ######
                    if padded:
                        from recurrent.utils import td_split
                        num_micro_batches = -(-len(mini_batch) // self.config.ppo_micro_batch_size_per_gpu)
                        micro_batches = td_split(mini_batch, num_micro_batches)
                    else:   
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        # split batch into micro_batches
                        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                #######
                # ADD: For unbias grad_gcc, see MODIFY in below for more info.
                #######
                if not self.config.use_dynamic_bsz:
                    from warnings import warn
                    warn("Using dynamic bsz is highly recomended for multiturn since there will be padding samples")
                mini_batch_response_mask = mini_batch['response_mask']
                mini_batch_token_nums = mini_batch_response_mask.sum()
                mini_batch_seq_count = _flatten_leading_dims(mini_batch_response_mask, keep_last_dims=1).shape[0]
                mini_batch_token_denom = max(mini_batch_token_nums.item(), 1e-8)

                for micro_batch in micro_batches:
                    # Support all hardwares
                    if isinstance(micro_batch, DataProto):
                        micro_batch = {**micro_batch.batch.to(torch.cuda.current_device()), **micro_batch.non_tensor_batch}
                    else:
                        micro_batch = micro_batch.to(torch.cuda.current_device())  # actor device is cpu when using offload

                    #######
                    # MODIFIED: use loss_mask directly
                    #######
                    calculate_entropy = self.config.entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(micro_batch=micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)

                    if log_prob is None or log_prob.numel() == 0:
                        print(f"{YELLOW}[POLICY Worker {rank}] Skipping empty micro-batch during policy update{RESET}")
                        continue

                    response_mask = micro_batch['response_mask']
                    try:
                        response_mask = _flatten_leading_dims(response_mask, keep_last_dims=1).reshape(log_prob.shape)
                    except RuntimeError as err:
                        raise RuntimeError(f"Failed to reshape response_mask from shape {response_mask.shape} to {log_prob.shape}") from err

                    token_count = response_mask.sum()
                    if token_count.item() == 0:
                        print(f"{YELLOW}[POLICY Worker {rank}] Response mask empty after reshaping; skipping micro-batch{RESET}")
                        continue

                    old_log_prob = micro_batch["old_log_probs"]
                    try:
                        old_log_prob = _flatten_leading_dims(old_log_prob, keep_last_dims=1).reshape(log_prob.shape)
                    except RuntimeError as err:
                        raise RuntimeError(f"Failed to reshape old_log_probs from shape {old_log_prob.shape} to {log_prob.shape}") from err

                    advantages = micro_batch["advantages"]
                    try:
                        advantages = _flatten_leading_dims(advantages, keep_last_dims=1).reshape(log_prob.shape)
                    except RuntimeError as err:
                        raise RuntimeError(f"Failed to reshape advantages from shape {advantages.shape} to {log_prob.shape}") from err

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

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

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = micro_batch["ref_log_prob"]
                        try:
                            ref_log_prob = _flatten_leading_dims(ref_log_prob, keep_last_dims=1).reshape(log_prob.shape)
                        except RuntimeError as err:
                            raise RuntimeError(f"Failed to reshape ref_log_prob from shape {ref_log_prob.shape} to {log_prob.shape}") from err
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    ######
                    # MODIFY: we have to fix grad_acc computation: weighted averaging by token num in stead of len(data)
                    #         See 
                    #         If we use Dr. GRPO algorithm（unbias_length_enable）, then this fix is no 
                    #           more needed since policy averaging there is sequence-level.
                    #         Since we have a variant of batchsize, we also remove self.gradient_accumulation
                    ######
                    acc_grad_mode = grad_acc_mode(loss_agg_mode)
                    if acc_grad_mode == "seq":
                        seq_denom = max(mini_batch_seq_count, 1)
                        loss = policy_loss * (log_prob.size(0) / seq_denom)
                    elif acc_grad_mode == "token":
                        # weights by token nums, note that we want to apply a simple scalar, or the compute-graph will be extremely large.
                        loss = policy_loss * (token_count.item() / mini_batch_token_denom)
                    else:
                        raise NotImplementedError(f"Unsupported acc_grad_mode: {acc_grad_mode}")


                    loss.backward()

                    metric_data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, metric_data)

                grad_norm = self._optimizer_step()
                metric_data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, metric_data)
        self.actor_optimizer.zero_grad()
        return metrics
