#!/usr/bin/env python3
"""
Simple preprocessing script to convert LongMemEval data to VERL format.
"""

import json
import os
import pandas as pd
import argparse
from typing import List, Dict, Any, Optional


def load_longmemeval_data(json_path: str) -> List[Dict[str, Any]]:
    """Load LongMemEval JSON data."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_haystack_sessions(haystack_sessions: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Flatten nested haystack_sessions into a single conversation list."""
    conversations = []
    for session in haystack_sessions:
        for message in session:
            if isinstance(message, dict) and 'role' in message and 'content' in message:
                conversations.append(message)
    return conversations


def convert_to_verl_format(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Convert LongMemEval item to VERL format.
    
    Maps:
    - question -> user query
    - answer -> correct_answer
    - haystack_sessions -> conversation context
    - question_id -> persona_id
    """
    # Extract basic fields
    question = item.get('question', '')
    answer = item.get('answer', '')
    question_id = item.get('question_id', '')
    
    # Flatten all haystack sessions into conversation history
    haystack_sessions = item.get('haystack_sessions', [])
    conversations = flatten_haystack_sessions(haystack_sessions)
    
    # Build messages: system + conversations + user question
    messages = []
    messages.append({
        "role": "system",
        "content": "You are a helpful assistant that provides personalized responses based on the user's conversation history."
    })
    
    # Add conversation history
    for conv in conversations:
        messages.append(conv)
    
    # Add thinking instruction to question
    question_with_instruction = question + " Always perform your reasoning inside <think> and </think> tags before your final answer."
    
    # Add user question
    messages.append({
        "role": "user",
        "content": question_with_instruction
    })
    
    # Build VERL record
    verl_data = {
        "data_source": "longmemeval",
        "prompt": messages,
        "ability": "personalization",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "groundtruth_preference": str(answer),  # Use answer as groundtruth_preference
                "correct_answer": str(answer),
                "all_answers": [str(answer)],  # Only one answer available
                "pref_type": item.get('question_type', ''),
                "is_mcq": False,
            },
        },
        "extra_info": {
            "index": idx,
            "persona_id": question_id,
            "question": question,
            "question_date": item.get('question_date', ''),
            "question_type": item.get('question_type', ''),
        },
    }
    
    return verl_data


def main():
    parser = argparse.ArgumentParser(description="Convert LongMemEval data to VERL parquet format")
    parser.add_argument(
        "--input_json",
        default="data/LongMemEval/longmemeval_s_cleaned.json",
        help="Path to LongMemEval JSON file"
    )
    parser.add_argument(
        "--output_dir",
        default="data/longmemeval",
        help="Directory to save output parquet file"
    )
    parser.add_argument(
        "--output_name",
        default="longmemeval_s.parquet",
        help="Output parquet filename"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_json}...")
    raw_data = load_longmemeval_data(args.input_json)
    print(f"Loaded {len(raw_data)} instances")
    
    # Convert to VERL format
    print("Converting to VERL format...")
    verl_data = []
    for idx, item in enumerate(raw_data):
        try:
            verl_item = convert_to_verl_format(item, idx)
            verl_data.append(verl_item)
        except Exception as e:
            print(f"Warning: Error converting item {idx}: {e}")
    
    print(f"Successfully converted {len(verl_data)} instances")
    
    # Create DataFrame
    df = pd.DataFrame(verl_data)
    
    # Convert nested structures to JSON strings for parquet compatibility
    df['prompt'] = df['prompt'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df['reward_model'] = df['reward_model'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df['extra_info'] = df['extra_info'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    
    # Save to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, engine='pyarrow')
    
    print(f"Done! Saved {len(df)} samples to {output_path}")
    
    # Show sample
    if len(df) > 0:
        print("\n=== Sample Record ===")
        print(f"Data source: {df.iloc[0]['data_source']}")
        print(f"Ability: {df.iloc[0]['ability']}")
        prompt = json.loads(df.iloc[0]['prompt'])
        print(f"Number of messages in prompt: {len(prompt)}")
        print(f"First message role: {prompt[0]['role']}")
        print(f"Last message role: {prompt[-1]['role']}")
        extra_info = json.loads(df.iloc[0]['extra_info'])
        print(f"Question: {extra_info['question'][:100]}...")
        reward_model = json.loads(df.iloc[0]['reward_model'])
        print(f"Correct answer: {reward_model['ground_truth']['correct_answer'][:100]}...")


if __name__ == "__main__":
    main()
