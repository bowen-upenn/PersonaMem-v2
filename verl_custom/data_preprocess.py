#!/usr/bin/env python3
"""
Preprocess the ImplicitPersona dataset to VERL-readable parquet format
"""

# Original path in verl: verl/examples/data_preprocess/gsm8k.py

import argparse
import json
import os
import random
import glob
import re
from typing import List, Dict, Any
import ast

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def find_latest_csv(data_dir: str = "data") -> str:
    """
    Find the latest ImplicitPersona CSV file based on timestamp in filename.
    
    Args:
        data_dir (str): Directory to search for CSV files
        
    Returns:
        str: Path to the latest CSV file
        
    Raises:
        FileNotFoundError: If no matching CSV files are found
    """
    # Pattern to match ImplicitPersona data files with timestamp
    pattern = os.path.join(data_dir, "implicit_persona_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        # Fallback to any CSV file in the directory
        pattern = os.path.join(data_dir, "*.csv")
        csv_files = glob.glob(pattern)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")
    
    # Extract timestamps and find the latest one
    latest_file = None
    latest_timestamp = ""
    
    timestamp_pattern = r"implicit_persona_(\d{8}_\d{6})\.csv"
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        match = re.search(timestamp_pattern, filename)
        if match:
            timestamp = match.group(1)
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_file = file_path
    
    # If no timestamped files found, use the first available CSV
    if latest_file is None:
        latest_file = csv_files[0]
    
    return latest_file


def load_persona_data(csv_path: str) -> pd.DataFrame:
    """
    Load ImplicitPersona data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing persona data
        
    Returns:
        pd.DataFrame: DataFrame containing the persona data
    """
    return pd.read_csv(csv_path)


def load_conversation_context(context_file_path: str) -> List[Dict[str, Any]]:
    """
    Load conversation context from JSON file.
    
    Args:
        context_file_path (str): Path to the context JSON file
        
    Returns:
        List[Dict[str, Any]]: List of conversation messages
    """
    try:
        with open(context_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle new JSON structure with metadata and messages
        if isinstance(data, dict) and 'messages' in data:
            return data['messages']
        # Handle old format (direct array of messages)
        elif isinstance(data, list):
            return data
        else:
            print(f"Warning: Unexpected JSON structure in {context_file_path}")
            return []
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load context file {context_file_path}: {e}")
        return []


def safe_eval_list(value: str) -> List[str]:
    """
    Safely evaluate string representation of list.
    
    Args:
        value (str): String representation of a list
        
    Returns:
        List[str]: Parsed list or empty list if parsing fails
    """
    if pd.isna(value) or value == '':
        return []
    
    try:
        # Handle string representation of lists
        if isinstance(value, str):
            # Try to parse as literal
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            else:
                return [str(parsed)]
        else:
            return [str(value)]
    except (ValueError, SyntaxError):
        # If parsing fails, return as single item list
        return [str(value)]


def format_conversation_as_prompt(conversations: List[Dict[str, Any]]) -> str:
    """
    Format conversation list as a multi-turn prompt string.
    
    Args:
        conversations (List[Dict[str, Any]]): List of conversation messages
        
    Returns:
        str: Formatted conversation string
    """
    if not conversations:
        return ""
    
    formatted_messages = []
    for i, msg in enumerate(conversations):
        # Handle new format (just content strings) and old format (role/content dicts)
        if isinstance(msg, dict):
            if 'content' in msg:
                # Old format with role and content
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                formatted_messages.append(f"{role.capitalize()}: {content}")
            else:
                # New format - assume it's just content, alternating user/assistant
                content = str(msg)
                role = 'user' if i % 2 == 0 else 'assistant'
                formatted_messages.append(f"{role.capitalize()}: {content}")
        elif isinstance(msg, str):
            # Direct string content - assume alternating user/assistant
            role = 'user' if i % 2 == 0 else 'assistant'
            formatted_messages.append(f"{role.capitalize()}: {msg}")
        else:
            # Fallback for any other format
            content = str(msg)
            role = 'user' if i % 2 == 0 else 'assistant'
            formatted_messages.append(f"{role.capitalize()}: {content}")
    
    return "\n".join(formatted_messages)


def create_train_test_split(data: List[Dict[str, Any]], test_ratio: float = 0.2) -> tuple:
    """
    Randomly split data into train and test sets.
    
    Args:
        data (List[Dict[str, Any]]): Complete dataset
        test_ratio (float): Ratio of data to use for testing (default: 0.2)
        
    Returns:
        tuple: (train_data, test_data)
    """
    # Shuffle data randomly
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split point
    test_size = int(len(shuffled_data) * test_ratio)
    
    # Split data
    test_data = shuffled_data[:test_size]
    train_data = shuffled_data[test_size:]
    
    return train_data, test_data


def process_persona_dataset(csv_path: str, test_ratio: float) -> tuple:
    """
    Process the complete ImplicitPersona dataset and return train/test splits.
    
    Args:
        csv_path (str): Path to the CSV file containing persona data
        test_ratio (float): Ratio of data to use for testing
        
    Returns:
        tuple: (train_dataset, test_dataset) as pandas DataFrames
    """
    print(f"Loading ImplicitPersona data from {csv_path}...")
    raw_data = load_persona_data(csv_path)
    print(f"Loaded {len(raw_data)} persona instances")
    
    # Convert to VERL format
    print("Converting to VERL format...")
    verl_data = []
    for idx, row in raw_data.iterrows():
        verl_instance = convert_to_verl_format(row, idx)
        if verl_instance:  # Only add if conversion was successful
            verl_data.append(verl_instance)
    
    print(f"Successfully converted {len(verl_data)} instances to VERL format")
    
    # Create train/test split
    print("Creating train/test split...")
    train_data, test_data = create_train_test_split(verl_data, test_ratio)
    
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Convert to pandas DataFrames
    train_dataset = pd.DataFrame(train_data)
    test_dataset = pd.DataFrame(test_data)
    
    return train_dataset, test_dataset


def convert_to_verl_format(row: pd.Series, idx: int) -> Dict[str, Any]:
    """
    Convert a single persona data row to VERL-readable format.
    
    Args:
        row (pd.Series): Single row from the persona CSV
        idx (int): Index of the instance
        
    Returns:
        Dict[str, Any]: VERL-formatted data instance or None if conversion fails
    """
    try:
        # Load conversation context from the context file
        context_file_path = row.get('context_file_path', '')
        if context_file_path and os.path.exists(context_file_path):
            conversations = load_conversation_context(context_file_path)
        else:
            print(f"Warning: Context file not found for row {idx}: {context_file_path}")
            conversations = []
        
        # Format conversations as prompt
        conversation_prompt = format_conversation_as_prompt(conversations)
        
        # Get question and add it to the prompt
        question = row.get('question', '')
        
        # Combine conversation context and question
        if conversation_prompt:
            full_prompt = f"{conversation_prompt}\n\nUser: {question}"
        else:
            full_prompt = f"User: {question}"
        
        # Get ground truth data
        correct_answer = row.get('correct_answer', '')
        groundtruth_preference = row.get('groundtruth_preference', '')
        
        # Parse all_answers if it's a string representation of a list
        all_answers = safe_eval_list(row.get('all_answers', '[]'))
        
        # Parse persona data if it's a string representation of a dict
        persona_data = row.get('persona', {})
        if isinstance(persona_data, str):
            try:
                persona_data = ast.literal_eval(persona_data)
            except (ValueError, SyntaxError):
                persona_data = {}
        
        # Create VERL-formatted data
        verl_data = {
            "data_source": "implicit_persona",
            "prompt": [
                {
                    "role": "user", 
                    "content": full_prompt,
                }
            ],
            "ability": "personalization",
            "reward_model": {
                "style": "rule", 
                "ground_truth": {
                    "groundtruth_preference": groundtruth_preference,
                    "correct_answer": correct_answer
                },
            },
            "extra_info": {
                "index": idx,
                "persona_id": row.get('persona_id', ''),
                "question": question,
                "correct_answer": correct_answer,
                "all_answers": all_answers,
                "groundtruth_preference": groundtruth_preference,
                "preference_type": row.get('preference_type', ''),
                "conversation_scenario": row.get('conversation_scenario', ''),
                "preference_topic": row.get('preference_topic', ''),
                "query_topic": row.get('query_topic', ''),
                "who": row.get('who', ''),
                "updated": row.get('updated', ''),
                "previous_preference": row.get('previous_preference', ''),
                "persona": persona_data,
                "context_file_path": context_file_path,
                "num_tokens": row.get('num_tokens', 0),
                "distance_from_query_to_target_conversation": row.get('distance_from_query_to_target_conversation', 0),
            },
        }
        
        return verl_data
    
    except Exception as e:
        print(f"Error converting row {idx} to VERL format: {e}")
        return None
def main():
    parser = argparse.ArgumentParser(description="Preprocess ImplicitPersona dataset to VERL format")
    parser.add_argument(
        "--csv_path", 
        default="data/example_debug.csv",
        help="Path to the CSV file containing persona data. If not specified, will automatically find the latest file in data/ folder"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory to search for CSV files when --csv_path is not specified"
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
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split"
    )
    
    args = parser.parse_args()
    
    # Determine which CSV file to use
    if args.csv_path is None:
        print(f"No CSV path specified, searching for latest file in {args.data_dir}/...")
        csv_path = find_latest_csv(args.data_dir)
        print(f"Found latest file: {csv_path}")
    else:
        csv_path = args.csv_path
        print(f"Using specified CSV file: {csv_path}")
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    # Expand local directory path
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Process the dataset
    train_dataset, test_dataset = process_persona_dataset(csv_path, args.test_ratio)
    
    # Save to parquet files
    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")

    print(f"Saving train dataset to {train_path}...")
    train_dataset.to_parquet(train_path)
    
    print(f"Saving test dataset to {test_path}...")
    test_dataset.to_parquet(test_path)
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"Copying data to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
    
    print("Dataset preprocessing completed successfully!")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total instances: {len(train_dataset) + len(test_dataset)}")
    print(f"Train instances: {len(train_dataset)}")
    print(f"Test instances: {len(test_dataset)}")
    print(f"Test ratio: {len(test_dataset) / (len(train_dataset) + len(test_dataset)):.1%}")
    
    # Print some sample data info
    if len(train_dataset) > 0:
        print("\nSample data structure:")
        sample_row = train_dataset.iloc[0]
        print(f"Data source: {sample_row.get('data_source', 'N/A')}")
        print(f"Ability: {sample_row.get('ability', 'N/A')}")
        print(f"Sample prompt length: {len(str(sample_row.get('prompt', '')))} characters")
        if 'extra_info' in sample_row and isinstance(sample_row['extra_info'], dict):
            extra_info = sample_row['extra_info']
            print(f"Sample persona_id: {extra_info.get('persona_id', 'N/A')}")
            print(f"Sample preference_type: {extra_info.get('preference_type', 'N/A')}")
            print(f"Sample question: {extra_info.get('question', 'N/A')}")
            print(f"Sample correct_answer: {extra_info.get('correct_answer', 'N/A')}")


if __name__ == "__main__":
    main()
