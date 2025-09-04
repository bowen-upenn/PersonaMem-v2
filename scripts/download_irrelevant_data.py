#!/usr/bin/env python3
"""
Script to download irrelevant datasets for context padding in 128k token version.
Downloads HotpotQA, MMLU, GSM8K, and BigCodeBench datasets.
"""

import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm
import tiktoken

# Initialize tokenizer
ENCODER = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    """Count tokens in text."""
    if isinstance(text, str):
        return len(ENCODER.encode(text))
    else:
        return len(ENCODER.encode(str(text)))

def download_hotpotqa(output_dir):
    """Download HotpotQA dataset (train split only)."""
    output_file = os.path.join(output_dir, "hotpotqa_train.json")
    
    # Check if already downloaded
    if os.path.exists(output_file):
        print(f"HotpotQA train data already exists at {output_file}, skipping download")
        return
    
    print("Downloading HotpotQA dataset...")
    
    try:
        # Load the dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        
        # Process training split only
        train_data = []
        for item in tqdm(ds['train'], desc="Processing HotpotQA train"):
            # Create chat format
            question = item['question']
            answer = item['answer']
            
            chat_entry = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "tokens": count_tokens(question) + count_tokens(answer)
            }
            train_data.append(chat_entry)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(train_data)} HotpotQA training examples to {output_file}")
    
    except Exception as e:
        print(f"Error downloading HotpotQA: {e}")

def download_mmlu(output_dir):
    """Download MMLU dataset (dev/train split only)."""
    output_file = os.path.join(output_dir, "mmlu_train.json")
    
    # Check if already downloaded
    if os.path.exists(output_file):
        print(f"MMLU train data already exists at {output_file}, skipping download")
        return
    
    print("Downloading MMLU dataset...")
    
    try:
        # Load the dataset
        ds = load_dataset("cais/mmlu", "all")
        
        # Process dev split (closest to train data)
        train_data = []
        split_to_use = 'dev' if 'dev' in ds else 'train' if 'train' in ds else 'test'
        
        for item in tqdm(ds[split_to_use], desc=f"Processing MMLU {split_to_use}"):
            # Create multiple choice question
            question = item['question']
            choices = item['choices']
            answer_idx = item['answer']
            
            # Format as multiple choice
            formatted_question = f"{question}\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{chr(65+i)}. {choice}\n"
            
            answer = f"The answer is {chr(65+answer_idx)}."
            
            chat_entry = {
                "messages": [
                    {"role": "user", "content": formatted_question.strip()},
                    {"role": "assistant", "content": answer}
                ],
                "tokens": count_tokens(formatted_question) + count_tokens(answer),
                "subject": item['subject']
            }
            train_data.append(chat_entry)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(train_data)} MMLU {split_to_use} examples to {output_file}")
    
    except Exception as e:
        print(f"Error downloading MMLU: {e}")

def download_gsm8k(output_dir):
    """Download GSM8K dataset (train split only)."""
    output_file = os.path.join(output_dir, "gsm8k_train.json")
    
    # Check if already downloaded
    if os.path.exists(output_file):
        print(f"GSM8K train data already exists at {output_file}, skipping download")
        return
    
    print("Downloading GSM8K dataset...")
    
    try:
        # Load the dataset
        ds = load_dataset("openai/gsm8k", "main")
        
        # Process training split only
        train_data = []
        for item in tqdm(ds['train'], desc="Processing GSM8K train"):
            question = item['question']
            answer = item['answer']
            
            chat_entry = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "tokens": count_tokens(question) + count_tokens(answer)
            }
            train_data.append(chat_entry)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(train_data)} GSM8K training examples to {output_file}")
    
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")

def download_bigcodebench(output_dir):
    """Download BigCodeBench dataset."""
    output_file = os.path.join(output_dir, "bigcodebench_train.json")
    
    # Check if already downloaded
    if os.path.exists(output_file):
        print(f"BigCodeBench data already exists at {output_file}, skipping download")
        return
    
    print("Downloading BigCodeBench dataset...")
    
    try:
        # Load the dataset
        ds = load_dataset("bigcode/bigcodebench")
        
        # Process the dataset (usually has 'test' split, but we'll use it as training data)
        split_name = list(ds.keys())[0]  # Get the first available split
        data = []
        
        for item in tqdm(ds[split_name], desc="Processing BigCodeBench"):
            # Extract relevant fields
            task_id = item.get('task_id', '')
            prompt = item.get('prompt', '')
            canonical_solution = item.get('canonical_solution', '')
            test = item.get('test', '')
            
            # Create a programming question format
            question = f"Task ID: {task_id}\n\nProblem:\n{prompt}"
            if test:
                question += f"\n\nTest cases:\n{test}"
            
            answer = f"Solution:\n{canonical_solution}"
            
            chat_entry = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "tokens": count_tokens(question) + count_tokens(answer),
                "task_id": task_id
            }
            data.append(chat_entry)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} BigCodeBench examples to {output_file}")
    
    except Exception as e:
        print(f"Error downloading BigCodeBench: {e}")

def create_combined_irrelevant_data(output_dir):
    """Combine all downloaded datasets into a single shuffled file for easy sampling."""
    output_file = os.path.join(output_dir, "combined_irrelevant_data.json")
    
    # Check if combined file already exists
    if os.path.exists(output_file):
        print(f"Combined irrelevant data already exists at {output_file}, skipping creation")
        return
    
    print("Creating combined irrelevant data file...")
    
    combined_data = []
    
    # Load all individual dataset files (train data only)
    dataset_files = [
        "hotpotqa_train.json",
        "mmlu_train.json", 
        "gsm8k_train.json",
        "bigcodebench_train.json"
    ]
    
    for filename in dataset_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_data.extend(data)
                    print(f"Loaded {len(data)} examples from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Shuffle the combined data
    random.shuffle(combined_data)
    
    # Save combined file
    output_file = os.path.join(output_dir, "combined_irrelevant_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    # Calculate total tokens
    total_tokens = sum(item['tokens'] for item in combined_data)
    
    print(f"Created combined dataset with {len(combined_data)} examples and {total_tokens:,} tokens")
    print(f"Saved to {output_file}")
    
    # Create summary statistics
    stats = {
        "total_examples": len(combined_data),
        "total_tokens": total_tokens,
        "average_tokens_per_example": total_tokens / len(combined_data) if combined_data else 0,
        "datasets_included": [f for f in dataset_files if os.path.exists(os.path.join(output_dir, f))]
    }
    
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset statistics saved to {stats_file}")

def main():
    """Main function to download all irrelevant datasets."""
    # Create output directory
    output_dir = "data/irrelevant"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading datasets to {output_dir}")
    print("Note: You may need to login with 'huggingface-cli login' for some datasets")
    
    # Download each dataset
    download_hotpotqa(output_dir)
    download_mmlu(output_dir)
    download_gsm8k(output_dir)
    download_bigcodebench(output_dir)
    
    # Create combined file
    create_combined_irrelevant_data(output_dir)
    
    print("All datasets downloaded successfully!")

if __name__ == "__main__":
    main()
