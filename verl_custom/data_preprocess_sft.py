#!/usr/bin/env python3
"""
Create multi-turn SFT-formatted data that works with the MultiTurnSFTDataset.
Converts RL data to full conversation format with assistant responses.
"""

import pandas as pd
import json
import argparse
import os
import tqdm
from typing import Optional, Dict, Any, List, Union
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoTokenizer, AutoProcessor


def filter_overlong_sft_data(
    dataset: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 37048,
    messages_key: str = "messages"
) -> pd.DataFrame:
    """
    Filter out SFT samples where full conversation exceeds max_length tokens.
    Similar to filter_overlong_prompts_in_dataset but for SFT format.
    """
    print(f"Original dataset length: {len(dataset)}")
    
    def compute_sft_length(row_data: Dict[str, Any]) -> int:
        """Compute the token length of full conversation for SFT training."""
        try:
            # Get the full messages (stored as JSON strings in parquet)
            full_messages = row_data[messages_key]
            
            # Parse JSON string back to Python objects
            if isinstance(full_messages, str):
                full_messages = json.loads(full_messages)
            
            if not isinstance(full_messages, list):
                print(f"Warning: Messages is not a list: {type(full_messages)}")
                return max_length + 1
            
            # Use the tokenizer's chat template
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                try:
                    # Apply chat template without tokenization first
                    formatted_text = tokenizer.apply_chat_template(
                        full_messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    # Then tokenize
                    tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                    return len(tokens)
                except Exception as e:
                    print(f"Warning: Chat template failed, using simple concatenation: {e}")
            
            # Fallback: simple concatenation
            all_text = ""
            for msg in full_messages:
                all_text += f"{msg['role']}: {msg['content']}\n"
            
            tokens = tokenizer.encode(all_text, add_special_tokens=True)
            return len(tokens)
            
        except Exception as e:
            print(f"Warning: Could not compute length for row, using max length + 1: {e}")
            return max_length + 1  # Mark for removal
    
    # Apply filtering
    filtered_indices = []
    for idx, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset), desc="Filtering overlong SFT samples"):
        try:
            seq_length = compute_sft_length(row)
            if seq_length <= max_length:
                filtered_indices.append(idx)
            else:
                print(f"Filtering out row {idx}: length {seq_length} > {max_length}")
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    filtered_dataset = dataset.loc[filtered_indices].reset_index(drop=True)
    removed_count = len(dataset) - len(filtered_dataset)
    print(f"Filtered dataset length: {len(filtered_dataset)} (removed {removed_count} overlong samples)")
    
    return filtered_dataset


def convert_rl_to_multiturn_sft(
    rl_parquet_paths: Union[str, List[str]], 
    output_path: str, 
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 37048,
    filter_overlong: bool = True
):
    """
    Convert RL-formatted parquet files to multi-turn SFT-formatted parquet compatible with MultiTurnSFTDataset.
    
    This creates full conversation data including assistant responses for multi-turn training.
    Can handle multiple input files to combine data from different sources.
    """
    # Handle both single file and multiple files
    if isinstance(rl_parquet_paths, str):
        rl_parquet_paths = [rl_parquet_paths]
    
    all_dfs = []
    for rl_parquet_path in rl_parquet_paths:
        if os.path.exists(rl_parquet_path):
            print(f"Loading RL data from {rl_parquet_path}")
            df = pd.read_parquet(rl_parquet_path)
            print(f"Loaded {len(df)} rows from {os.path.basename(rl_parquet_path)}")
            all_dfs.append(df)
        else:
            print(f"Warning: {rl_parquet_path} not found, skipping")
    
    if not all_dfs:
        print("No valid input files found, creating empty SFT file")
        sft_df = pd.DataFrame({'messages': []})
        sft_df.to_parquet(output_path)
        return
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined total: {len(df)} rows")
    
    if len(df) == 0:
        print("Empty dataframe, creating empty SFT file")
        sft_df = pd.DataFrame({'messages': []})
        sft_df.to_parquet(output_path)
        return
    
    sft_data = []
    mcq_count = 0
    regular_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse the JSON-encoded prompt (list of messages)
            messages = json.loads(row['prompt'])
            
            # Parse the reward_model to get the correct answer and MCQ status
            reward_model = json.loads(row['reward_model'])
            
            # Validate messages is a list of message dicts
            if not isinstance(messages, list):
                print(f"Warning: Row {idx} prompt is not a list, skipping")
                continue
            
            # Extract the correct answer from reward_model
            try:
                correct_answer = reward_model['ground_truth']['correct_answer']
                if not correct_answer or correct_answer.strip() == '':
                    print(f"Warning: Row {idx} has empty correct_answer, skipping")
                    continue
            except (KeyError, TypeError):
                print(f"Warning: Row {idx} missing correct_answer in reward_model, skipping")
                continue
            
            # Check if this is MCQ format
            is_mcq = reward_model.get('ground_truth', {}).get('is_mcq', False)
            
            # For MCQ format, fix the correct_answer format from (x) to \boxed{x} while keeping content
            if is_mcq and correct_answer:
                # Check if answer starts with (a), (b), (c), or (d) and convert to \boxed{} format
                import re
                mcq_pattern = r'^\([abcd]\)'
                if re.match(mcq_pattern, correct_answer.strip()):
                    # Extract the letter and convert format, but keep the original content
                    letter = correct_answer.strip()[1]  # Get 'a', 'b', 'c', or 'd'
                    # Keep everything after the "(x) " part
                    content_after_letter = correct_answer.strip()[3:]  # Skip "(x) "
                    # Format as: \boxed{letter} + original content
                    correct_answer = f"\\boxed{{{letter}}} {content_after_letter}"
            
            # Ensure all messages have required keys
            valid_messages = []
            for msg in messages:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    valid_messages.append(msg)
            
            if not valid_messages:
                print(f"Warning: Row {idx} has no valid messages, skipping")
                continue
            
            # Prepare conversation for training - should end with user message
            # If it ends with assistant message, remove it (we'll use correct_answer as target)
            if valid_messages[-1]['role'] == 'assistant':
                conversation_messages = valid_messages[:-1]
            else:
                conversation_messages = valid_messages
                
            # Make sure we have at least one message and it ends with user
            if not conversation_messages or conversation_messages[-1]['role'] != 'user':
                print(f"Warning: Row {idx} doesn't end with user message, skipping")
                continue
            
            # Clean up the last user message by removing the thinking instruction
            # This instruction always appears in the final user query (second-to-last message)
            if conversation_messages and conversation_messages[-1]['role'] == 'user':
                last_user_content = conversation_messages[-1]['content']
                # Remove the thinking instruction that appears at the end
                thinking_instruction = " Always perform your reasoning inside <think> and </think> tags before your final answer."
                if last_user_content.endswith(thinking_instruction):
                    conversation_messages[-1]['content'] = last_user_content[:-len(thinking_instruction)]
            
            # Create full conversation including assistant response for multi-turn training
            # This includes the entire conversation history + the target assistant response
            full_conversation = conversation_messages + [{"role": "assistant", "content": correct_answer.strip()}]
            
            sft_data.append({
                'messages': full_conversation,  # Full conversation in OpenAI format
                'original_idx': idx,
                'is_mcq': is_mcq
            })
            
            # Count MCQ vs regular
            if is_mcq:
                mcq_count += 1
            else:
                regular_count += 1
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"Converted {len(sft_data)} valid conversations for SFT")
    
    # Show MCQ statistics
    if sft_data:
        total = len(sft_data)
        mcq_percentage = (mcq_count / total * 100) if total > 0 else 0
        regular_percentage = (regular_count / total * 100) if total > 0 else 0
        print(f"  Regular format: {regular_count} samples ({regular_percentage:.1f}%)")
        print(f"  MCQ format: {mcq_count} samples ({mcq_percentage:.1f}%)")
    
    # Create SFT dataframe with JSON-encoded messages
    # PyArrow can't handle nested Python objects directly, so we store as JSON strings
    # The MultiTurnSFTDataset will need to parse these JSON strings
    if sft_data:
        # Convert messages to JSON strings for storage in parquet
        for item in sft_data:
            item['messages'] = json.dumps(item['messages'], ensure_ascii=False)
        sft_df = pd.DataFrame(sft_data)
    else:
        sft_df = pd.DataFrame({'messages': [], 'original_idx': [], 'is_mcq': []})
    
    # Apply length filtering if tokenizer is provided and filtering is enabled
    if filter_overlong and tokenizer is not None and len(sft_df) > 0:
        print(f"Filtering overlong SFT samples (max_length={max_length})...")
        sft_df = filter_overlong_sft_data(
            dataset=sft_df,
            tokenizer=tokenizer,
            max_length=max_length,
            messages_key="messages"
        )
    
    # Save to parquet
    print(f"Saving SFT data to {output_path}")
    sft_df.to_parquet(output_path, engine='pyarrow')


def main():
    parser = argparse.ArgumentParser(description="Convert RL parquet files to multi-turn SFT format with length filtering")
    parser.add_argument("--input_dir", default="verl_custom/data/implicit_persona", 
                       help="Directory containing RL parquet files")
    parser.add_argument("--output_dir", default="verl_custom/data/implicit_persona_sft",
                       help="Directory to save multi-turn SFT parquet files")
    parser.add_argument("--model_path", 
                       default="verl_custom/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796",
                       help="Model path for tokenizer loading")
    parser.add_argument("--max_length", type=int, default=37048,
                       help="Maximum sequence length (prompt + response)")
    parser.add_argument("--filter_overlong", action="store_true",
                       help="Enable filtering of overlong sequences (disabled by default since data is pre-filtered)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                       help="Trust remote code when loading tokenizer")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer for length filtering (only if explicitly requested)
    tokenizer = None
    if args.filter_overlong and args.model_path:
        print(f"Loading tokenizer from {args.model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, 
                trust_remote_code=args.trust_remote_code
            )
            print(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Proceeding without length filtering...")
    else:
        print("Skipping tokenizer loading - length filtering disabled (data is pre-filtered)")
    
    print("=" * 80)
    print("MULTI-TURN SFT DATA CREATION CONFIGURATION")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Max length: {args.max_length} tokens")
    print(f"Filter overlong: {args.filter_overlong}")
    print(f"Trust remote code: {args.trust_remote_code}")
    print("NOTE: Data is pre-filtered in data_preprocess.py, so length filtering is disabled by default")
    print("=" * 80)
    
    # Convert train data (single file)
    train_files = ["train_text_32k.parquet"]
    train_paths = [os.path.join(args.input_dir, f) for f in train_files]
    train_output = os.path.join(args.output_dir, "train.parquet")
    
    existing_train_paths = [p for p in train_paths if os.path.exists(p)]
    if existing_train_paths:
        print(f"\n=== Converting train files to train.parquet ===")
        convert_rl_to_multiturn_sft(
            existing_train_paths,
            train_output,
            tokenizer=tokenizer,
            max_length=args.max_length,
            filter_overlong=args.filter_overlong
        )
    else:
        print("Warning: No train files found, skipping")
    
    # Convert validation data (combine regular and MCQ files to maintain ratio)
    val_files = ["val_text_32k.parquet", "val_text_32k_mcq.parquet"]
    val_paths = [os.path.join(args.input_dir, f) for f in val_files]
    val_output = os.path.join(args.output_dir, "val.parquet")
    
    existing_val_paths = [p for p in val_paths if os.path.exists(p)]
    if existing_val_paths:
        print(f"\n=== Converting validation files to val.parquet ===")
        print(f"Combining files: {[os.path.basename(p) for p in existing_val_paths]}")
        convert_rl_to_multiturn_sft(
            existing_val_paths,
            val_output,
            tokenizer=tokenizer,
            max_length=args.max_length,
            filter_overlong=args.filter_overlong
        )
    else:
        print("Warning: No validation files found, skipping")
    
    print(f"\nMulti-turn SFT data saved to {args.output_dir}")


if __name__ == "__main__":
    main()