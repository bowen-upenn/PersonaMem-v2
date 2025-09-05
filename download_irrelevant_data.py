#!/usr/bin/env python3
"""
Script to download irrelevant datasets for context padding in 128k token version.
Downloads HotpotQA, MMLU, GSM8K, and BigCodeBench datasets.
"""

import os
import json
import random
import sys
from datasets import load_dataset
from tqdm import tqdm
import tiktoken
import yaml
import argparse
import concurrent.futures
import threading
import math
import time


# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from query_llm import QueryLLM

# Initialize tokenizer
ENCODER = tiktoken.encoding_for_model("gpt-4o")

# Thread-safe lock for collecting results
RESULTS_LOCK = threading.Lock()

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

def process_single_query_thread(args):
    """
    Thread-safe function to process a single user query.
    
    Args:
        args: Tuple containing (llm, user_query, user_message, filename, verbose)
    
    Returns:
        chat_entry dict or None if processing failed
    """
    llm, user_query, user_message, filename, verbose = args
    
    # Set a unique random seed for this thread
    random.seed(int(time.time() * 1000000) + threading.get_ident())
    
    try:
        if verbose:
            print(f"Processing query: {user_query[:100]}...")
        
        # Query the LLM with just the user query (no conversation history for irrelevant data)
        llm.reset_history()
        model_response = llm.query_llm(user_query, use_history=False, verbose=verbose)
        
        if model_response:
            # Create simple OpenAI format with user and assistant messages only
            chat_entry = {
                "messages": [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": model_response}
                ],
                "tokens": count_tokens(user_query) + count_tokens(model_response)
            }
            
            if verbose:
                query_tokens = count_tokens(user_query)
                response_tokens = count_tokens(model_response)
                print(f"Added query-response pair ({query_tokens + response_tokens} tokens)")
            
            return chat_entry
        else:
            if verbose:
                print("Warning: No response received from LLM")
            return None
    
    except Exception as e:
        if verbose:
            print(f"Error querying LLM for user query: {e}")
        return None


def process_user_queries_and_responses(llm, output_dir, parallel=False, sample_size=1000, verbose=False):
    """
    Process user queries from downloaded datasets, generate LLM responses, and return as irrelevant data.
    
    Args:
        llm: QueryLLM instance for generating responses
        output_dir: Directory containing downloaded datasets
        parallel: Whether to process queries in parallel batches
        sample_size: Number of queries to randomly sample for processing
        verbose: Whether to print detailed information
    
    Returns:
        List of processed query-response pairs in OpenAI format
    """
    print("Processing user queries from downloaded datasets and generating LLM responses...")
    
    # Dataset files to process for user queries
    dataset_files = [
        "hotpotqa_train.json",
        "mmlu_train.json", 
        "gsm8k_train.json",
        "bigcodebench_train.json"
    ]
    
    # Collect all queries from all datasets first
    all_queries_pool = []
    
    for filename in tqdm(dataset_files, desc="Loading datasets"):
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            if verbose:
                print(f"Dataset file {filename} not found, skipping...")
            continue
        
        if verbose:
            print(f"Loading queries from {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            # Collect all queries from this dataset
            for item in dataset:
                # Extract the user query from the first message
                messages = item.get('messages', [])
                if not messages or len(messages) < 1:
                    continue
                
                user_message = messages[0]
                if user_message.get('role') != 'user':
                    continue
                
                user_query = user_message.get('content', '')
                if not user_query:
                    continue
                
                # Add to the pool for sampling
                all_queries_pool.append((llm, user_query, user_message, filename, verbose))
        
        except Exception as e:
            print(f"Error loading dataset file {filepath}: {e}")
            continue
    
    if not all_queries_pool:
        print("No queries found to process")
        return
    
    print(f"Found {len(all_queries_pool)} total queries from all datasets")
    
    # Randomly sample the specified number of queries (or all if less than sample_size)
    actual_sample_size = min(sample_size, len(all_queries_pool))
    random.shuffle(all_queries_pool)
    all_query_args = all_queries_pool[:actual_sample_size]
    
    print(f"Randomly sampled {len(all_query_args)} queries for processing")
    query_response_data = []
    
    if parallel:
        # Parallel processing in batches
        max_workers = min(llm.rate_limit_per_min, len(all_query_args))
        batch_size = max_workers
        num_batches = math.ceil(len(all_query_args) / batch_size)
        
        print(f"Processing in {num_batches} batches with up to {batch_size} parallel workers each")
        
        for batch_idx in tqdm(range(num_batches)):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, len(all_query_args))
            batch_args = all_query_args[batch_start_idx:batch_end_idx]
            
            # print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_args)} queries)")
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                # Submit all tasks for this batch
                future_to_args = {executor.submit(process_single_query_thread, args): args for args in batch_args}
                
                # Collect results with progress bar
                for future in concurrent.futures.as_completed(future_to_args):
                    try:
                        result = future.result()
                        if result is not None:
                            with RESULTS_LOCK:
                                query_response_data.append(result)
                    except Exception as e:
                        args = future_to_args[future]
                        user_query = args[1][:50] + "..." if len(args[1]) > 50 else args[1]
                        print(f"Error in future for query '{user_query}': {e}")
    else:
        # Sequential processing
        for args in tqdm(all_query_args, desc="Processing queries sequentially"):
            result = process_single_query_thread(args)
            if result is not None:
                query_response_data.append(result)
    
    if query_response_data:
        total_tokens = sum(item['tokens'] for item in query_response_data)
        print(f"Generated {len(query_response_data)} user query-response pairs")
        print(f"Total tokens from query-response pairs: {total_tokens:,}")
        return query_response_data
    else:
        print("No user query-response pairs were generated")
        return []


def create_combined_irrelevant_data(output_dir, processed_queries=None):
    """Create combined irrelevant data file with processed queries and original datasets."""
    output_file = os.path.join(output_dir, "combined_irrelevant_data.json")
    
    # Always regenerate the combined file and stats
    if os.path.exists(output_file):
        print("Regenerating combined irrelevant data file...")
    else:
        print("Creating combined irrelevant data file...")
    
    combined_data = []
    
    if processed_queries:
        # If we have processed queries, use them as the main content
        combined_data.extend(processed_queries)
        print(f"Added {len(processed_queries)} processed query-response pairs")
    else:
        # If no processed queries, load original datasets (fallback behavior)
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
    if processed_queries:
        # When using processed queries, stats reflect only the processed sample
        stats = {
            "total_examples": len(combined_data),
            "total_tokens": total_tokens,
            "average_tokens_per_example": total_tokens / len(combined_data) if combined_data else 0,
            "datasets_included": [f"processed_queries_sample ({len(processed_queries)} queries from all datasets)"]
        }
    else:
        # Fallback: original dataset stats
        dataset_files = [
            "hotpotqa_train.json",
            "mmlu_train.json", 
            "gsm8k_train.json",
            "bigcodebench_train.json"
        ]
        datasets_included = [f for f in dataset_files if os.path.exists(os.path.join(output_dir, f))]
        stats = {
            "total_examples": len(combined_data),
            "total_tokens": total_tokens,
            "average_tokens_per_example": total_tokens / len(combined_data) if combined_data else 0,
            "datasets_included": datasets_included
        }
    
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset statistics saved to {stats_file}")

def main():
    """Main function to download all irrelevant datasets and process user queries."""
    # Load config from YAML file
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')
        return
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download irrelevant datasets and optionally process user queries")
    parser.add_argument('--process-queries', action='store_true', 
                       help='Process user queries from downloaded datasets and generate LLM responses')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', 
                       help='Set LLM model.')
    parser.add_argument('--parallel', action='store_true',
                       help='Process queries in parallel batches (default: sequential)')
    parser.add_argument('--rate-limit-per-min', type=int, default=20,
                       help='Maximum number of parallel requests per minute (default: 20)')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of queries to randomly sample for processing (default: 1000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed processing information')
    
    cmd_args = parser.parse_args()
    
    # Override model in config if specified in command line
    if cmd_args.model is not None:
        config['models']['llm_model'] = cmd_args.model
    
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
    
    # Process user queries if requested
    processed_queries = []
    if cmd_args.process_queries:
        print("\nProcessing user queries and generating LLM responses...")
        # Create LLM instance with custom rate limit
        llm = QueryLLM(config, rate_limit_per_min=cmd_args.rate_limit_per_min)
        
        # Process user queries and get results
        processed_queries = process_user_queries_and_responses(llm, output_dir, parallel=cmd_args.parallel, sample_size=cmd_args.sample_size, verbose=cmd_args.verbose)
            
    # Create combined file with processed queries
    create_combined_irrelevant_data(output_dir, processed_queries)
    
    print("All datasets downloaded successfully!")
    if cmd_args.process_queries:
        print("User query processing completed!")

if __name__ == "__main__":
    main()
