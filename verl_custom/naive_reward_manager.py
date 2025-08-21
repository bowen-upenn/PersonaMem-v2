# Original path in verl: verl/verl/workers/reward_manager/naive.py

import os
import sys

# Add the VERL package to Python path so we can import from verl modules
current_dir = os.path.dirname(os.path.abspath(__file__))
verl_path = os.path.join(current_dir, "..", "verl")
if verl_path not in sys.path:
    sys.path.insert(0, verl_path)

# Add the parent directory to Python path so we can import from verl_custom
parent_dir = os.path.join(current_dir, "..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from collections import defaultdict
import random

import torch

from verl import DataProto
from verl_custom import default_compute_score
from verl.workers.reward_manager import register


@register("custom_naive")
class CustomNaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", eval_method='embed') -> None:
        """
        Initialize the CustomNaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.eval_method = eval_method  # Default evaluation method


    # In naive_reward_manager.py, modify the __call__ method
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        # Track total count and randomly selected indices for each data source
        data_source_counts = defaultdict(int)
        data_source_selected_indices = defaultdict(set)
        
        # First pass: count data sources and randomly select indices to print
        for i in range(len(data)):
            data_item = data[i]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            data_source_counts[data_source] += 1
        
        # For each data source, randomly select which indices to print
        for data_source, count in data_source_counts.items():
            if count > 0:
                # Randomly select min(num_examine, count) indices
                num_to_select = min(self.num_examine, count)
                selected_indices = random.sample(range(count), num_to_select)
                data_source_selected_indices[data_source] = set(selected_indices)
        
        # Reset counters for actual processing
        data_source_current_index = defaultdict(int)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            groundtruth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            groundtruth_preference = groundtruth.get("groundtruth_preference", "")
            correct_answer = groundtruth.get("correct_answer", "")

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # Extract ImplicitPersona specific information
            persona_id = extra_info.get('persona_id', '')
            question = extra_info.get('question', '')
            persona = extra_info.get('persona', {})
            preference_type = extra_info.get('preference_type', '')

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=groundtruth,
                extra_info=extra_info,
                eval_method=self.eval_method,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # Add ImplicitPersona specific information to reward_extra_info
            reward_extra_info["persona_ids"].append(persona_id)
            reward_extra_info["groundtruth_preferences"].append(groundtruth_preference)
            reward_extra_info["correct_answers"].append(correct_answer)
            reward_extra_info["preference_types"].append(preference_type)

            reward_tensor[i, valid_response_length - 1] = reward

            # Check if this example should be printed (randomly selected)
            current_index = data_source_current_index[data_source]
            should_print = current_index in data_source_selected_indices[data_source]
            data_source_current_index[data_source] += 1

            if should_print:
                print("[response]", response_str)
                print("[persona]", persona)
                print("[persona_id]", persona_id)
                print("[question]", question)
                print("[correct_answer]", correct_answer)
                print("[groundtruth_preference]", groundtruth_preference)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                print('\n' + '-' * 50 + '\n')

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor