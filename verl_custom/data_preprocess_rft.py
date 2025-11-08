#!/usr/bin/env python3
"""
Preprocess the ImplicitPersona dataset (new benchmark schema) to VERL-readable parquet format.

This version:
- Reads fixed dataset paths:
    Text:
        data/benchmark/text/train.csv
        data/benchmark/text/val.csv
    Multimodal:
        data/benchmark/multimodal/train.csv
        data/benchmark/multimodal/val.csv
- Emits 8 parquet files (split × dtype × window):
    train_text_32k.parquet,    val_text_32k.parquet
    train_text_128k.parquet,   val_text_128k.parquet
    train_multimodal_32k.parquet,  val_multimodal_32k.parquet
    train_multimodal_128k.parquet, val_multimodal_128k.parquet
- Keeps original VERL structure and functionality.
- IMPORTANT: extra_info contains ONLY the active window's fields (32k OR 128k).
"""

# Original path in verl: verl/examples/data_preprocess/gsm8k.py

import argparse
import json
import os
from typing import List, Dict, Any, Optional
import ast
import tqdm
import pandas as pd
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.hdfs_io import copy, makedirs
import random

# ----------------------------
# General I/O helpers
# ----------------------------

def load_persona_data(csv_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(csv_path)


def load_conversation_context(context_file_path: str) -> List[Dict[str, Any]]:
    """
    Load conversation context from JSON file.

    Supports both:
      - {"messages": [...]}
      - [...] (list of messages directly)
    """
    try:
        with open(context_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'chat_history' in data:
            return data['chat_history']
        elif isinstance(data, list):
            return data
        else:
            print(f"Warning: Unexpected JSON structure in {context_file_path}")
            return []

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load context file {context_file_path}: {e}")
        return []


# ----------------------------
# Parsing helpers
# ----------------------------

def safe_eval_list(value: Any) -> List[str]:
    """
    Safely evaluate string representation of list, returning List[str].
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == '':
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            else:
                return [str(parsed)]
        else:
            return [str(value)]
    except (ValueError, SyntaxError):
        return [str(value)]


def safe_eval_dict(value: Any) -> Dict[str, Any]:
    """
    Safely evaluate string representation of a dict. Returns {} on failure.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == '':
        return {}
    if isinstance(value, dict):
        return value
    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
    except (ValueError, SyntaxError):
        pass
    return {}


# ----------------------------
# Configuration parsing utilities
# ----------------------------

def parse_script_overrides(script_file: str) -> dict:
    """
    Parse parameter overrides from shell script file.
    Returns a dictionary of parameter paths to values.
    """
    overrides = {}
    
    if not os.path.exists(script_file):
        print(f"Warning: Script file not found: {script_file}")
        return overrides
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Extract parameter overrides (e.g., data.max_prompt_length=32000)
        import re
        # Match patterns like "parameter.path=value \"
        pattern = r'(\w+(?:\.\w+)+)=([^\s\\]+)'
        matches = re.findall(pattern, content)
        
        for param_path, value in matches:
            # Convert string values to appropriate types
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]  # Remove quotes
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]  # Remove quotes
            else:
                # Try to convert to number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string if not a number
                    pass
            
            overrides[param_path] = value
            
        print(f"Parsed {len(overrides)} parameter overrides from {script_file}")
        for param, val in overrides.items():
            print(f"  {param} = {val}")
            
    except Exception as e:
        print(f"Warning: Could not parse script file {script_file}: {e}")
    
    return overrides


def apply_overrides_to_config(config, overrides: dict):
    """
    Apply parameter overrides to OmegaConf config object.
    """
    from omegaconf import OmegaConf
    
    for param_path, value in overrides.items():
        try:
            # Split the parameter path (e.g., "data.max_prompt_length" -> ["data", "max_prompt_length"])
            keys = param_path.split('.')
            
            # Navigate to the parent container
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = OmegaConf.create({})
                current = current[key]
            
            # Set the final value
            final_key = keys[-1]
            current[final_key] = value
            
            print(f"Applied override: {param_path} = {value}")
        except Exception as e:
            print(f"Warning: Could not apply override {param_path}={value}: {e}")
    
    return config


# ----------------------------
# Length filtering functionality (moved from rl_dataset.py)
# ----------------------------

def filter_overlong_prompts_in_dataset(
    dataset: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin] = None,
    max_prompt_length: int = 1024,
    prompt_key: str = "prompt",
    image_key: str = "images", 
    video_key: str = "videos",
    enable_thinking: Optional[bool] = None,
    chat_template_func = None,
    num_workers: int = 1
) -> pd.DataFrame:
    """
    Filter out prompts that are longer than max_prompt_length tokens.
    This function replicates the filtering logic from rl_dataset.py.
    """
    print(f"Original dataset length: {len(dataset)}")
    
    def compute_prompt_length(row_data: Dict[str, Any]) -> int:
        """Compute the token length of a prompt, mirroring rl_dataset.py logic."""
        
        def build_messages(example: dict):
            """Replicated _build_messages logic from rl_dataset.py"""
            messages = example.get(prompt_key, [])
            
            # Handle JSON-serialized prompt data
            if isinstance(messages, str):
                messages = json.loads(messages)
            
            # Handle image/video processing (simplified for preprocessing)
            if image_key in example or video_key in example:
                import re
                for message in messages:
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                        content_list = []
                        segments = re.split("(<image>|<video>)", content)
                        segments = [item for item in segments if item != ""]
                        for segment in segments:
                            if segment == "<image>":
                                content_list.append({"type": "image"})
                            elif segment == "<video>":
                                content_list.append({"type": "video"})
                            else:
                                content_list.append({"type": "text", "text": segment})
                        message["content"] = content_list
            
            return messages

        try:
            messages = build_messages(row_data)
            
            if processor is not None:
                # Processor-based length computation
                try:
                    from verl.utils.dataset.vision_utils import process_image, process_video
                except ImportError:
                    # Fallback if vision_utils not available during preprocessing
                    pass
                
                if enable_thinking is None:
                    raw_prompt = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                else:
                    raw_prompt = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
                    )
                
                # For preprocessing, we'll use a simplified approach without actual image/video processing
                # This assumes text-only length computation during preprocessing
                return len(processor.tokenizer.encode(raw_prompt, add_special_tokens=False))
            
            else:
                # Tokenizer-based length computation
                prompt_data = row_data.get(prompt_key, [])
                if chat_template_func is not None:
                    return len(chat_template_func(prompt_data, add_generation_prompt=True))
                else:
                    return len(tokenizer.apply_chat_template(prompt_data, add_generation_prompt=True, enable_thinking=enable_thinking))
        
        except Exception as e:
            print(f"Warning: Could not compute prompt length for row: {e}")
            return 0  # Keep rows where we can't compute length (conservative approach)
    
    # Apply filtering
    filtered_indices = []
    for idx, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset), desc="Filtering overlong prompts"):
        try:
            row_dict = row.to_dict()
            prompt_length = compute_prompt_length(row_dict)
            if prompt_length <= max_prompt_length:
                filtered_indices.append(idx)
        except Exception as e:
            print(f"Warning: Error processing row {idx}: {e}")
            # Keep row if we can't process it (conservative approach)
            filtered_indices.append(idx)
    
    filtered_dataset = dataset.loc[filtered_indices].reset_index(drop=True)
    print(f"Filtered dataset length: {len(filtered_dataset)} (removed {len(dataset) - len(filtered_dataset)} overlong prompts)")
    
    return filtered_dataset


# ----------------------------
# Processing
# ----------------------------

def _select_context_fields(row: pd.Series, window: str) -> Dict[str, Any]:
    """
    Select context-dependent fields for 32k or 128k and return ONLY those.
    Keys returned:
      - context_file_path (str)
      - chat_history_link (str)
      - num_tokens (int)
      - distance (int/float)
      - num_persona_relevant_tokens (int)
      - num_persona_irrelevant_tokens (int)
    """
    window = window.strip().lower()
    if window not in ("32k", "128k"):
        raise ValueError(f"window must be '32k' or '128k', got {window}")

    if window == "32k":
        chat_link = row.get('chat_history_32k_link', '')
        context_file_path = chat_link
        num_tokens = row.get('total_tokens_in_chat_history_32k', 0)
        distance = row.get('distance_from_related_snippet_to_query_32k', 0)
        num_rel = row.get('num_persona_relevant_tokens_32k', 0)
        num_irrel = row.get('num_persona_irrelevant_tokens_32k', 0)
    else:
        chat_link = row.get('chat_history_128k_link', '')
        context_file_path = chat_link
        num_tokens = row.get('total_tokens_in_chat_history_128k', 0)
        distance = row.get('distance_from_related_snippet_to_query_128k', 0)
        num_rel = row.get('num_persona_relevant_tokens_128k', 0)
        num_irrel = row.get('num_persona_irrelevant_tokens_128k', 0)

    def _nz_str(v):
        return '' if (isinstance(v, float) and pd.isna(v)) else v

    return {
        "context_file_path": _nz_str(context_file_path),
        "chat_history_link": _nz_str(chat_link),
        "num_tokens": 0 if pd.isna(num_tokens) else num_tokens,
        "distance": 0 if pd.isna(distance) else distance,
        "num_persona_relevant_tokens": 0 if pd.isna(num_rel) else num_rel,
        "num_persona_irrelevant_tokens": 0 if pd.isna(num_irrel) else num_irrel,
    }


def _extract_question_from_row(row: pd.Series) -> str:
    """
    Build the user question string from new schema.

    'user_query' is typically a JSON-like string with {'role': 'user', 'content': '...'}.
    """
    user_query = row.get('user_query', '')
    question_text = ''

    if isinstance(user_query, dict):
        question_text = str(user_query.get('content', ''))
    elif isinstance(user_query, str):
        uq_dict = safe_eval_dict(user_query)
        if uq_dict:
            question_text = str(uq_dict.get('content', ''))
        else:
            question_text = user_query
    else:
        question_text = str(user_query) if user_query is not None else ''

    return question_text


def convert_to_verl_format(row: pd.Series, idx: int, window: str, is_mcq: bool = False) -> Optional[Dict[str, Any]]:
    """
    Convert a single row to VERL-readable format, adapted to new column names.
    window: '32k' or '128k'
    is_mcq: If True, add MCQ prompt to the user question
    """
    try:
        # Window-specific fields (ONLY the chosen window)
        ctx = _select_context_fields(row, window)
        context_file_path = ctx["context_file_path"]

        # Load conversation context if available
        if context_file_path and os.path.exists(context_file_path):
            conversations = load_conversation_context(context_file_path)
        else:
            if context_file_path:
                print(f"Warning: Context file not found for row {idx}: {context_file_path}")
            conversations = []

        # Ensure conversations is always a list
        if not isinstance(conversations, list):
            conversations = []

        # Build user question
        question = _extract_question_from_row(row)
        
        # Get answer options
        correct_answer_text = row.get('correct_answer', '')
        incorrect_answers = safe_eval_list(row.get('incorrect_answers', '[]'))
        all_answers = []
        if correct_answer_text != '':
            all_answers.append(str(correct_answer_text))
        all_answers.extend(str(x) for x in incorrect_answers)
        
        # Add MCQ format to the question if requested and we have enough options
        if is_mcq and len(all_answers) >= 4:
            # Randomize the order of options so correct answer isn't always 'a'
            shuffled_answers = all_answers[:4].copy()
            random.seed(42 + idx)  # Use deterministic seed based on row index for reproducibility
            random.shuffle(shuffled_answers)
            
            # Find which position the correct answer ended up in
            correct_index = shuffled_answers.index(str(correct_answer_text))
            correct_letter = chr(97 + correct_index)  # Convert to 'a', 'b', 'c', or 'd'
            correct_answer = f"({correct_letter}) {correct_answer_text}"  # Format as "(a) text"
            
            options_text = "\n".join([
                f"({chr(97 + i)}) {option}"  # chr(97) = 'a'
                for i, option in enumerate(shuffled_answers)
            ])
            mcq_prompt = f"\n\nYou are performing a multiple-choice question task. You must choose the best response from the following options to answer the user query above:\n{options_text}\n\nProvide your answer in the format: \\boxed{{a}}, \\boxed{{b}}, \\boxed{{c}}, or \\boxed{{d}}."
            question = question + mcq_prompt
            all_answers = shuffled_answers  # Update all_answers to match MCQ order
        else:
            # Regular format - keep original correct answer
            correct_answer = correct_answer_text
        
        # Add thinking instruction at the end
        question += " Always perform your reasoning inside <think> and </think> tags before your final answer."

        # Compose messages: system + conversation + user
        messages = []
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant that provides personalized responses based on the user's preferences in conversation history."
        })
        
        # Add conversations, ensuring each is a valid message dict
        if conversations:
            for conv in conversations:
                if isinstance(conv, dict) and "role" in conv and "content" in conv:
                    messages.append(conv)
        
        messages.append({
            "role": "user",
            "content": question
        })
        
        pref_type = row.get('pref_type', '')
        groundtruth_preference = row.get('preference', row.get('groundtruth_preference', ''))

        # Build VERL record
        verl_data = {
            "data_source": "implicit_persona",
            "prompt": messages,
            "ability": "personalization",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "groundtruth_preference": str(groundtruth_preference),
                    "correct_answer": str(correct_answer),
                    "all_answers": [str(option) for option in all_answers],
                    "pref_type": str(pref_type),
                    "is_mcq": is_mcq,
                },
            },
            "extra_info": {
                # Base identifiers and Q/A - ensure all are serializable types
                "index": idx if pd.notna(idx) else 0,
                "persona_id": row.get('persona_id'),
                "question": question,
            },
        }

        return verl_data

    except Exception as e:
        print(f"Error converting row {idx} to VERL format (window={window}): {e}")

def process_persona_dataset(csv_path: str, window: str, sample_size: Optional[int] = None, 
                          tokenizer: Optional[PreTrainedTokenizer] = None,
                          processor: Optional[ProcessorMixin] = None,
                          max_prompt_length: int = 1024,
                          enable_thinking: Optional[bool] = None,
                          filter_overlong: bool = True,
                          is_mcq: bool = False) -> pd.DataFrame:
    """
    Process the dataset at csv_path and return a DataFrame for the requested window.
    window: '32k' or '128k'
    sample_size: If provided, randomly sample this many rows from the dataset
    tokenizer: Tokenizer for length filtering (optional)
    processor: Processor for multimodal length filtering (optional) 
    max_prompt_length: Maximum prompt length in tokens
    enable_thinking: Enable thinking tokens for Qwen models
    filter_overlong: Whether to filter overlong prompts during preprocessing
    is_mcq: If True, add MCQ prompts to all questions
    """
    print(f"Loading data from {csv_path}...")
    raw_data = load_persona_data(csv_path)
    print(f"Loaded {len(raw_data)} instances")
    
    # Apply sampling if specified
    if sample_size is not None and len(raw_data) > sample_size:
        print(f"Randomly sampling {sample_size} instances from {len(raw_data)} total instances...")
        raw_data = raw_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled dataset size: {len(raw_data)} instances")

    print(f"Converting to VERL format for window={window} (validation mode: {is_mcq})...")
    verl_data = []
    for idx, row in tqdm.tqdm(raw_data.iterrows(), total=len(raw_data)):
        verl_instance = convert_to_verl_format(row, idx, window, is_mcq=is_mcq)
        if verl_instance is not None:
            verl_data.append(verl_instance)

    dataset = pd.DataFrame(verl_data)
    
    # Apply length filtering during preprocessing if tokenizer is provided and filtering is enabled
    if filter_overlong and tokenizer is not None and len(dataset) > 0:
        print(f"Filtering overlong prompts (max_length={max_prompt_length})...")
        dataset = filter_overlong_prompts_in_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            processor=processor,
            max_prompt_length=max_prompt_length,
            prompt_key="prompt",
            enable_thinking=enable_thinking
        )
    
    # Validate the dataset before returning
    if len(dataset) > 0:
        # Ensure all columns have consistent string types (since we JSON serialized them)
        prompt_types = dataset['prompt'].apply(type).unique()
        reward_model_types = dataset['reward_model'].apply(type).unique()
        extra_info_types = dataset['extra_info'].apply(type).unique()
        
        print(f"Prompt column types: {prompt_types}")
        print(f"Reward model column types: {reward_model_types}")
        print(f"Extra info column types: {extra_info_types}")
    
    print(f"Successfully converted {len(dataset)} instances to VERL format (window={window})")
    return dataset


# ----------------------------
# Main CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess ImplicitPersona dataset to VERL format (new benchmark schema), generating parquet files for text/multimodal × train/val × 32k/128k. Reads configuration parameters (model path, max_prompt_length, enable_thinking, trust_remote_code) from YAML config file to ensure perfect alignment with training configuration. extra_info contains ONLY the active window's info.")
    # Fixed default paths per your request
    parser.add_argument(
        "--text_train_csv",
        default="data/benchmark/text/train.csv",
        help="Path to the TEXT train CSV file."
    )
    parser.add_argument(
        "--text_val_csv",
        default="data/benchmark/text/benchmark.csv",
        help="Path to the TEXT val CSV file."
    )
    parser.add_argument(
        "--multimodal_train_csv",
        default="data/benchmark/multimodal/train.csv",
        help="Path to the MULTIMODAL train CSV file."
    )
    parser.add_argument(
        "--multimodal_val_csv",
        default="data/benchmark/multimodal/val.csv",
        help="Path to the MULTIMODAL val CSV file."
    )
    parser.add_argument(
        "--local_dir",
        default="verl_custom/data/implicit_persona",
        help="Local directory to save processed data"
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory to copy processed data (optional)"
    )
    # New arguments for length filtering - following main_ppo.py pattern and training script
    parser.add_argument(
        "--config_file",
        default="verl_custom/ppo_trainer.yaml",
        help="YAML configuration file to read training parameters from (default: verl_custom/ppo_trainer.yaml)"
    )
    parser.add_argument(
        "--script_file",
        default="verl_custom/scripts/run_qwen3_8b_ppo.sh",
        help="Shell script file to read parameter overrides from (default: verl_custom/scripts/run_qwen3_8b_ppo.sh)"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Model path for tokenizer and processor loading. If not provided, will be read from config/script files."
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=None,
        help="Enable thinking tokens for Qwen models. If not provided, will be read from config/script files."
    )
    parser.add_argument(
        "--no_filter_overlong",
        action="store_true", 
        help="Disable filtering of overlong prompts during preprocessing"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=None,
        help="Trust remote code when loading tokenizer/processor. If not provided, will be read from config/script files."
    )
    parser.add_argument(
        "--use_shm",
        action="store_true",
        help="Use shared memory for faster model loading"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check existing parquet files in local_dir and show data sample counts, then exit"
    )

    args = parser.parse_args()

    # Load configuration from YAML file to ensure perfect alignment with training
    print(f"Loading configuration from: {args.config_file}")
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config_file)
    
    # Parse and apply overrides from shell script
    print(f"Loading parameter overrides from: {args.script_file}")
    script_overrides = parse_script_overrides(args.script_file)
    if script_overrides:
        config = apply_overrides_to_config(config, script_overrides)
    
    # Extract key parameters from config (now with overrides applied), with command line overrides taking precedence
    model_path = args.model_path if args.model_path else config.actor_rollout_ref.model.path
    max_prompt_length = config.data.max_prompt_length
    enable_thinking = args.enable_thinking if args.enable_thinking is not None else config.data.get("enable_thinking", None)
    trust_remote_code = args.trust_remote_code if args.trust_remote_code is not None else config.data.get("trust_remote_code", False)
    
    print("=" * 80)
    print("PREPROCESSING WITH CONFIG")
    print("=" * 80)
    print(f"Config file: {args.config_file}")
    print(f"Script file: {args.script_file}")
    print(f"Model path: {model_path}")
    print(f"Max prompt length: {max_prompt_length} tokens")
    print(f"Enable thinking: {enable_thinking}")
    print(f"Trust remote code: {trust_remote_code}")
    print(f"Filter overlong: {not args.no_filter_overlong}")
    print("=" * 80)

    # Prepare output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Check existing parquet files if --check-only flag is provided
    if args.check_only:
        print("=" * 80)
        print("CHECKING EXISTING PARQUET FILES")
        print("=" * 80)
        print(f"Directory: {local_dir}")
        
        # Find all parquet files in the directory
        parquet_files = []
        if os.path.exists(local_dir):
            for filename in os.listdir(local_dir):
                if filename.endswith('.parquet'):
                    parquet_files.append(filename)
        
        if not parquet_files:
            print("No parquet files found in the directory.")
            return
        
        # Sort files for consistent output
        parquet_files.sort()
        
        print(f"Found {len(parquet_files)} parquet file(s):")
        print()
        
        total_samples = 0
        for filename in parquet_files:
            filepath = os.path.join(local_dir, filename)
            try:
                df = pd.read_parquet(filepath)
                num_samples = len(df)
                total_samples += num_samples
                print(f"{filename:<30} : {num_samples:>8,} samples")
            except Exception as e:
                print(f"{filename:<30} : Error reading file - {e}")
        
        print("-" * 50)
        print(f"{'Total':<30} : {total_samples:>8,} samples")
        print("=" * 80)
        return
    
    # Load tokenizer if provided - following main_ppo.py pattern exactly
    tokenizer = None
    processor = None
    
    if model_path:
        try:
            print(f"Loading model from: {model_path}")
            
            # Download the checkpoint from HDFS to the local machine, following main_ppo.py
            from verl.utils.fs import copy_to_local
            local_path = copy_to_local(model_path, use_shm=args.use_shm)
            
            # Instantiate the tokenizer and processor following main_ppo.py exactly
            from verl.utils import hf_processor, hf_tokenizer
            
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            # Used for multimodal LLM, could be None
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
            
            print(f"Loaded tokenizer using VERL utils from: {local_path}")
            if processor is not None:
                print(f"Loaded processor using VERL utils from: {local_path}")
                
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Proceeding without length filtering...")
            tokenizer = None
            processor = None

    def process_split(csv_path: Optional[str], dtype: str, split: str, window: str):
        """
        Process a single (csv_path, dtype, split, window) combination and save parquet.
        For validation splits, creates both MCQ and embedding similarity versions.
        For training splits, creates both regular version and MCQ version for 20% of data.
        """
        if not csv_path or not os.path.exists(csv_path):
            print(f"Skipping {dtype} {split} ({window}): CSV not found at {csv_path}")
            return

        # For validation splits, create both MCQ and embedding similarity versions
        if split == "val":
            sample_size = None
            
            # Process MCQ version (with MCQ prompts)
            print(f"Processing MCQ validation version for {dtype} {split} ({window})...")
            df_mcq = process_persona_dataset(
                csv_path=csv_path, 
                window=window, 
                sample_size=sample_size,
                tokenizer=tokenizer,
                processor=processor,
                max_prompt_length=max_prompt_length,
                enable_thinking=enable_thinking if enable_thinking is not None else None,
                filter_overlong=not args.no_filter_overlong,
                is_mcq=True  # Add MCQ prompts
            )
            out_path_mcq = os.path.join(local_dir, f"benchmark_{dtype}_{window}_mcq.parquet")
            print(f"Saving {dtype} {split} ({window}) MCQ dataset to {out_path_mcq}...")
            
            # Convert nested structures to JSON strings to avoid PyArrow nested data issues
            df_to_save_mcq = df_mcq.copy()
            df_to_save_mcq['prompt'] = df_to_save_mcq['prompt'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            df_to_save_mcq['reward_model'] = df_to_save_mcq['reward_model'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            df_to_save_mcq['extra_info'] = df_to_save_mcq['extra_info'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            
            df_to_save_mcq.to_parquet(out_path_mcq, engine='pyarrow')
            
            # Process embedding similarity version (without MCQ prompts)
            print(f"Processing embedding similarity validation version for {dtype} {split} ({window})...")
            df_embed = process_persona_dataset(
                csv_path=csv_path, 
                window=window, 
                sample_size=sample_size,
                tokenizer=tokenizer,
                processor=processor,
                max_prompt_length=max_prompt_length,
                enable_thinking=enable_thinking if enable_thinking is not None else None,
                filter_overlong=not args.no_filter_overlong,
                is_mcq=False  # No MCQ prompts, just regular questions
            )
            out_path_embed = os.path.join(local_dir, f"benchmark_{dtype}_{window}.parquet")
            print(f"Saving {dtype} {split} ({window}) embedding similarity dataset to {out_path_embed}...")
            
            # Convert nested structures to JSON strings to avoid PyArrow nested data issues
            df_to_save_embed = df_embed.copy()
            df_to_save_embed['prompt'] = df_to_save_embed['prompt'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            df_to_save_embed['reward_model'] = df_to_save_embed['reward_model'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            df_to_save_embed['extra_info'] = df_to_save_embed['extra_info'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            
            df_to_save_embed.to_parquet(out_path_embed, engine='pyarrow')
            
            # Quick stats/sample for both versions
            print(f"{dtype.upper()} {split.upper()} MCQ ({window}) -> {len(df_mcq)} instances")
            print(f"{dtype.upper()} {split.upper()} EMBED ({window}) -> {len(df_embed)} instances")
            
            if len(df_mcq) > 0:
                sample_row = df_mcq.iloc[0]
                print(f"  MCQ Data source: {sample_row.get('data_source', 'N/A')}")
                print(f"  MCQ Ability: {sample_row.get('ability', 'N/A')}")
                print(f"  MCQ Sample prompt length: {len(str(sample_row.get('prompt', '')))} characters")
                
            if len(df_embed) > 0:
                sample_row = df_embed.iloc[0]
                print(f"  Embed Data source: {sample_row.get('data_source', 'N/A')}")
                print(f"  Embed Ability: {sample_row.get('ability', 'N/A')}")
                print(f"  Embed Sample prompt length: {len(str(sample_row.get('prompt', '')))} characters")
        else:
            # For training splits, create mixed dataset: 80% regular + 20% MCQ in single file
            print(f"Processing training data for {dtype} {split} ({window})...")
            
            # Load raw data to determine MCQ vs regular split
            raw_data = load_persona_data(csv_path)
            print(f"Loaded {len(raw_data)} instances")
            
            if len(raw_data) == 0:
                print(f"Warning: No data found for {dtype} {split} ({window})")
                return

            # Randomly select 80% for MCQ format using deterministic random seed
            random.seed(42)
            indices = list(range(len(raw_data)))
            random.shuffle(indices)
            
            mcq_size = int(0.8 * len(raw_data))
            mcq_indices = set(indices[:mcq_size])
            
            print(f"  Processing {len(raw_data)} samples: {len(raw_data) - mcq_size} regular + {mcq_size} MCQ")
            
            # Process all samples, using is_mcq flag based on random selection
            verl_data = []
            for idx, row in tqdm.tqdm(raw_data.iterrows(), total=len(raw_data)):
                is_mcq = idx in mcq_indices
                verl_instance = convert_to_verl_format(row, idx, window, is_mcq=is_mcq)
                if verl_instance is not None:
                    verl_data.append(verl_instance)
            
            df_full = pd.DataFrame(verl_data)
            
            # Apply length filtering if enabled
            if not args.no_filter_overlong and tokenizer is not None and len(df_full) > 0:
                print(f"Filtering overlong prompts (max_length={max_prompt_length})...")
                df_full = filter_overlong_prompts_in_dataset(
                    dataset=df_full,
                    tokenizer=tokenizer,
                    processor=processor,
                    max_prompt_length=max_prompt_length,
                    prompt_key="prompt",
                    enable_thinking=enable_thinking
                )
            
            # Save mixed training dataset
            out_path = os.path.join(local_dir, f"{split}_{dtype}_{window}.parquet")
            print(f"Saving {dtype} {split} ({window}) mixed dataset to {out_path}...")
            
            df_to_save = df_full.copy()
            df_to_save['prompt'] = df_to_save['prompt'].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
            df_to_save['reward_model'] = df_to_save['reward_model'].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
            df_to_save['extra_info'] = df_to_save['extra_info'].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
            
            df_to_save.to_parquet(out_path, engine='pyarrow')
            
            # Count actual MCQ vs regular samples for stats (after filtering)
            reward_models = df_full['reward_model'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            mcq_count = sum(1 for rm in reward_models if rm.get('ground_truth', {}).get('is_mcq', False))
            regular_count = len(df_full) - mcq_count
            mcq_percentage = (mcq_count / len(df_full) * 100) if len(df_full) > 0 else 0
            print(f"{dtype.upper()} {split.upper()} ({window}) -> {len(df_full)} instances")
            print(f"  Regular format: {regular_count} samples ({100-mcq_percentage:.1f}%)")
            print(f"  MCQ format: {mcq_count} samples ({mcq_percentage:.1f}%)")

    # TEXT: train/val × 32k/128k
    print("\n=== Processing TEXT splits ===")
    # for split, csv_path in (("train", args.text_train_csv), ("val", args.text_val_csv)):
        # for window in ("32k", "128k"):
    # process_split(csv_path, dtype="text", split=split, window="32k")
    process_split(args.text_val_csv, dtype="text", split="val", window="32k")
    process_split(args.text_val_csv, dtype="text", split="val", window="128k")

    # # MULTIMODAL: train/val × 32k/128k
    # print("\n=== Processing MULTIMODAL splits ===")
    # for split, csv_path in (("train", args.multimodal_train_csv), ("val", args.multimodal_val_csv)):
    #     # for window in ("32k", "128k"):
    #     process_split(csv_path, dtype="multimodal", split=split, window='32k')

    # Copy to HDFS if specified (copy the entire output directory with all outputs)
    if args.hdfs_dir is not None:
        print(f"\nCopying all outputs to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print("\nAll requested parquet files generated successfully.")


if __name__ == "__main__":
    main()
