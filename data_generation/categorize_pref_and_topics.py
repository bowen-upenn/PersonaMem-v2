#!/usr/bin/env python3
"""
Script to categorize missing topic_preference and topic_query fields in JSON files.
Uses existing topics from preference_topic_counts_topic_counts.json and qa_topic_counts.json.
"""

import json
import os
import glob
import yaml
import argparse
import concurrent.futures
import threading
import re
from typing import Dict, List, Optional
from tqdm import tqdm

# Import necessary modules from the project
import prompts
import utils
from query_llm import QueryLLM

# Global lock for thread-safe operations
FILE_SAVE_LOCK = threading.Lock()


def extract_persona_number(filename: str) -> Optional[int]:
    """Extract persona number from filename ending with 'personaX.json'."""
    match = re.search(r'persona(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return None


def filter_persona_files(json_files: List[str], start_idx: int = -1, end_idx: int = -1) -> List[str]:
    """Filter JSON files based on persona index range."""
    if start_idx == -1 and end_idx == -1:
        return json_files
    
    # Extract persona numbers and create tuples of (filename, persona_number)
    file_persona_pairs = []
    for json_file in json_files:
        persona_num = extract_persona_number(os.path.basename(json_file))
        if persona_num is not None:
            file_persona_pairs.append((json_file, persona_num))
    
    # Sort by persona number
    file_persona_pairs.sort(key=lambda x: x[1])
    
    # Filter based on range
    filtered_files = []
    for json_file, persona_num in file_persona_pairs:
        if start_idx != -1 and persona_num < start_idx:
            continue
        if end_idx != -1 and persona_num > end_idx:
            continue
        filtered_files.append(json_file)
    
    return filtered_files


def load_existing_topics(preference_counts_file: str, qa_counts_file: str) -> tuple:
    """Load existing topics from count files."""
    preference_topics = []
    qa_topics = []
    
    # Load preference topics
    try:
        with open(preference_counts_file, 'r') as f:
            pref_data = json.load(f)
            if 'topic_counts' in pref_data:
                preference_topics = list(pref_data['topic_counts'].keys())
                print(f"Loaded {len(preference_topics)} preference topics")
    except FileNotFoundError:
        print(f"Warning: {preference_counts_file} not found")
    
    # Load QA topics
    try:
        with open(qa_counts_file, 'r') as f:
            qa_data = json.load(f)
            if 'topic_counts' in qa_data:
                qa_topics = list(qa_data['topic_counts'].keys())
                print(f"Loaded {len(qa_topics)} QA topics")
    except FileNotFoundError:
        print(f"Warning: {qa_counts_file} not found")
    
    return preference_topics, qa_topics


def categorize_single_item(llm, text, global_topics, verbose=False):
    """
    Categorize a single text item (preference or query) into topics using the LLM.
    Based on the original function from conv_generator.py
    """
    # Generate categorization prompt
    prompt = prompts.categorize_preference_topic(text, global_topics)
    
    try:
        llm.reset_history()  # Reset history for categorization
        topic = llm.query_llm(prompt, use_history=False, verbose=verbose)
        topic_temp = utils.extract_after_token(topic, '###Output').strip()
        if not topic_temp:
            topic = utils.extract_after_token(topic, '### Output').strip()
        else:
            topic = topic_temp

        # Clean the topic
        if topic:
            if verbose:
                print(f"Categorized '{text}' as topic: '{topic}'")
            return topic
        else:
            if verbose:
                print(f"Failed to categorize '{text}', using 'Uncategorized'")
            return "Uncategorized"
            
    except Exception as e:
        if verbose:
            print(f"Error categorizing '{text}': {e}, using 'Uncategorized'")
        return "Uncategorized"


def process_conversation_items(items: List[Dict], llm, preference_topics: List[str], 
                             qa_topics: List[str], verbose=False) -> bool:
    """Process conversation items and fill missing topic fields."""
    updated = False
    
    for item in items:
        # Check and fill topic_preference
        if 'preference' in item and item['preference'] and not item.get('topic_preference'):
            topic = categorize_single_item(llm, item['preference'], preference_topics, verbose)
            item['topic_preference'] = topic
            updated = True
            if verbose:
                print(f"Added topic_preference: {topic} for preference: {item['preference'][:50]}...")
        
        # Check and fill topic_query
        if 'user_query' in item and item['user_query'] and not item.get('topic_query'):
            topic = categorize_single_item(llm, item['user_query'], qa_topics, verbose)
            item['topic_query'] = topic
            updated = True
            if verbose:
                print(f"Added topic_query: {topic} for query: {item['user_query'][:50]}...")
    
    return updated


def process_single_file(json_file_path: str, llm, preference_topics: List[str], 
                       qa_topics: List[str], verbose=False) -> bool:
    """Process a single JSON file and fill missing topic fields."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_updated = False
        
        # Process each persona in the file
        for persona_id, persona_data in data.items():
            if 'conversations' in persona_data:
                conversations_data = persona_data['conversations']
                
                # Process all conversation scenarios
                for scenario_name, scenario_data in conversations_data.items():
                    if isinstance(scenario_data, list):
                        if process_conversation_items(scenario_data, llm, preference_topics, qa_topics, verbose):
                            file_updated = True
        
        # Save the file if it was updated (thread-safe)
        if file_updated:
            with FILE_SAVE_LOCK:
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                if verbose:
                    print(f"Updated file: {json_file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing file {json_file_path}: {e}")
        return False


def process_file_thread(args):
    """Thread function to process a single file."""
    json_file_path, config_args, preference_topics, qa_topics, verbose = args
    
    # Each thread gets its own LLM instance
    llm = QueryLLM(config_args)
    
    return process_single_file(json_file_path, llm, preference_topics, qa_topics, verbose)


def main():
    """Main function to process all JSON files."""
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Categorize missing topic preferences and queries')
    parser.add_argument('--model', type=str, default="gpt-5-chat", help='Set LLM model')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--rate_limit_per_min', type=int, default=5, help='Rate limit for API calls per minute')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 5 files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--persona_start_idx', type=int, default=-1, help='Starting persona index (-1 for beginning)')
    parser.add_argument('--persona_end_idx', type=int, default=-1, help='Ending persona index (-1 for end)')

    cmd_args = parser.parse_args()
    
    # File paths
    preference_counts_file = "data/preference_topic_counts_topic_counts.json"
    qa_counts_file = "data/qa_topic_counts.json"
    raw_data_dir = "data/raw_data"
    
    print("Loading existing topics...")
    preference_topics, qa_topics = load_existing_topics(preference_counts_file, qa_counts_file)
    
    if not preference_topics and not qa_topics:
        print("No topics loaded. Exiting.")
        return
    
    # Load config for LLM initialization
    try:
        with open('config.yaml', 'r') as file:
            config_args = yaml.safe_load(file)
    except Exception as e:
        print(f'Error reading config file: {e}')
        return
    
    # Override config with command line arguments
    config_args['models']['llm_model'] = cmd_args.model
    config_args['inference']['rate_limit_per_min'] = cmd_args.rate_limit_per_min
    config_args['inference']['verbose'] = cmd_args.verbose
    
    # Find all JSON files
    json_pattern = os.path.join(raw_data_dir, "*.json")
    all_json_files = glob.glob(json_pattern)
    
    if not all_json_files:
        print(f"No JSON files found in {raw_data_dir}")
        return
    
    # Filter files based on persona range
    json_files = filter_persona_files(all_json_files, cmd_args.persona_start_idx, cmd_args.persona_end_idx)
    
    if not json_files:
        print(f"No persona files found in specified range (start: {cmd_args.persona_start_idx}, end: {cmd_args.persona_end_idx})")
        return
    
    # Show range information
    if cmd_args.persona_start_idx != -1 or cmd_args.persona_end_idx != -1:
        start_str = str(cmd_args.persona_start_idx) if cmd_args.persona_start_idx != -1 else "beginning"
        end_str = str(cmd_args.persona_end_idx) if cmd_args.persona_end_idx != -1 else "end"
        print(f"Processing persona range: {start_str} to {end_str}")
    
    # Limit files for testing
    if cmd_args.test:
        json_files = json_files[:5]  # Process only first 5 files in test mode
        print(f"TEST MODE: Processing only {len(json_files)} files")
    else:
        print(f"Found {len(json_files)} persona files to process")
    
    updated_files = 0
    total_files = len(json_files)
    
    if cmd_args.parallel and total_files > 1:
        print(f"Processing files in parallel with rate limit of {cmd_args.rate_limit_per_min} workers...")
        
        # Prepare arguments for parallel processing
        file_args = [(json_file, config_args, preference_topics, qa_topics, cmd_args.verbose) 
                     for json_file in json_files]
        
        # Process files in parallel batches using rate_limit_per_min as batch size
        batch_size = min(cmd_args.rate_limit_per_min, total_files)
        num_batches = (total_files + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_files)
            batch_args = file_args[batch_start:batch_end]
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_args)} files)")
            
            # Process batch in parallel with rate_limit_per_min as max_workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                future_to_file = {executor.submit(process_file_thread, args): args[0] for args in batch_args}
                
                # Collect results with progress bar
                for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                                 desc=f"Batch {batch_idx + 1}", 
                                 total=len(batch_args)):
                    try:
                        result = future.result()
                        if result:
                            updated_files += 1
                    except Exception as e:
                        file_path = future_to_file[future]
                        print(f"Error processing {file_path}: {e}")
    else:
        print("Processing files sequentially...")
        # Initialize single LLM for sequential processing
        llm = QueryLLM(config_args)
        
        # Process files sequentially with progress bar
        for json_file in tqdm(json_files, desc="Processing files"):
            if process_single_file(json_file, llm, preference_topics, qa_topics, verbose=cmd_args.verbose):
                updated_files += 1
    
    print(f"\nCompleted processing!")
    print(f"Updated {updated_files} out of {total_files} files")


if __name__ == "__main__":
    main()
