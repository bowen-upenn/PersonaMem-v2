#!/usr/bin/env python3
"""
Script to prepare benchmark data from raw persona files.
Creates a single CSV file with one row per user_query for evaluation.
Each row contains persona info, chat history links, and QA data.
"""

import json
import csv
import os
import glob
import re
import tiktoken
import pandas as pd
import numpy as np
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Initialize encoder for token counting (same as contexts_builder.py)
ENCODER = tiktoken.encoding_for_model("gpt-4o")


def extract_persona_number(filename: str) -> Optional[int]:
    """Extract persona number from filename pattern."""
    match = re.search(r'persona(\d+)\.json$', filename)
    return int(match.group(1)) if match else None


def extract_expanded_persona(persona_data: Dict[str, Any]) -> str:
    """Extract everything before 'stereotypical_preferences' as expanded persona."""
    expanded_persona = {}
    
    for key, value in persona_data.items():
        if key == "stereotypical_preferences":
            break
        expanded_persona[key] = value
    
    return json.dumps(expanded_persona, indent=2, ensure_ascii=False)


def get_chat_history_links(persona_number: int) -> Tuple[str, str, str, str]:
    """Generate chat history file links for a persona (32k and 128k versions)."""
    # Check if 128k directories exist
    chat_history_128k_dir_exists = os.path.exists("data/chat_history_128k")
    multimodal_128k_dir_exists = os.path.exists("data/chat_history_multimodal_128k")
    
    # Print warning once if 128k directories don't exist
    if not hasattr(get_chat_history_links, '_warned_128k') and (not chat_history_128k_dir_exists or not multimodal_128k_dir_exists):
        print("Warning: 128k directories not found. Using null values for 128k-related columns.")
        if not chat_history_128k_dir_exists:
            print("  Missing: data/chat_history_128k/")
        if not multimodal_128k_dir_exists:
            print("  Missing: data/chat_history_multimodal_128k/")
        get_chat_history_links._warned_128k = True
    
    # Look for chat history files with the pattern for both versions
    chat_history_32k_pattern = f"data/chat_history_32k/chat_history_*_persona{persona_number}.json"
    multimodal_32k_pattern = f"data/chat_history_multimodal_32k/chat_history_*_persona{persona_number}.json"
    
    chat_history_32k_files = glob.glob(chat_history_32k_pattern)
    multimodal_32k_files = glob.glob(multimodal_32k_pattern)
    
    chat_history_32k_link = chat_history_32k_files[0] if chat_history_32k_files else ""
    multimodal_32k_link = multimodal_32k_files[0] if multimodal_32k_files else ""
    
    # Only look for 128k files if directories exist
    if chat_history_128k_dir_exists:
        chat_history_128k_pattern = f"data/chat_history_128k/chat_history_*_persona{persona_number}.json"
        chat_history_128k_files = glob.glob(chat_history_128k_pattern)
        chat_history_128k_link = chat_history_128k_files[0] if chat_history_128k_files else ""
    else:
        chat_history_128k_link = ""
    
    if multimodal_128k_dir_exists:
        multimodal_128k_pattern = f"data/chat_history_multimodal_128k/chat_history_*_persona{persona_number}.json"
        multimodal_128k_files = glob.glob(multimodal_128k_pattern)
        multimodal_128k_link = multimodal_128k_files[0] if multimodal_128k_files else ""
    else:
        multimodal_128k_link = ""
    
    return chat_history_32k_link, chat_history_128k_link, multimodal_32k_link, multimodal_128k_link


def get_total_tokens_from_chat_history(chat_history_file: str) -> int:
    """Get total token count from chat history JSON file."""
    if not chat_history_file or not os.path.exists(chat_history_file):
        return 0
    
    try:
        with open(chat_history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get final_token_count from metadata
        return data.get("metadata", {}).get("final_token_count", 0)
    except Exception as e:
        print(f"Error reading chat history file {chat_history_file}: {str(e)}")
        return 0


def get_token_breakdown_from_128k_chat_history(chat_history_file: str) -> Tuple[int, int, int]:
    """
    Get token breakdown from 128k chat history JSON file.
    
    Returns:
        Tuple of (total_tokens, num_relevant_tokens, num_irrelevant_tokens)
    """
    if not chat_history_file or not os.path.exists(chat_history_file):
        return 0, 0, 0
    
    try:
        with open(chat_history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get("metadata", {})
        total_tokens = metadata.get("final_token_count", 0)
        num_relevant_tokens = metadata.get("num_relevant_tokens", 0)
        num_irrelevant_tokens = metadata.get("num_irrelevant_tokens", 0)
        
        return total_tokens, num_relevant_tokens, num_irrelevant_tokens
    except Exception as e:
        print(f"Error reading 128k chat history file {chat_history_file}: {str(e)}")
        return 0, 0, 0


def count_tokens_in_text(text: str) -> int:
    """Count tokens in text using the same encoder as contexts_builder.py."""
    if not text:
        return 0
    return len(ENCODER.encode(text))


def find_conversation_snippet_in_chat_history(chat_history_file: str, conversation_snippet: str, user_query: str) -> int:
    """
    Find the position of conversation snippet in chat history and calculate tokens from the end of chat history 
    back to where the snippet first appears.
    
    Args:
        chat_history_file: Path to the chat history JSON file
        conversation_snippet: The related conversation snippet to find
        user_query: The user query (as dict with role and content) - used for verification
    
    Returns:
        Number of tokens from the end of chat history back to the conversation snippet
    """
    if not chat_history_file or not os.path.exists(chat_history_file) or not conversation_snippet:
        return 0
    
    try:
        with open(chat_history_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        chat_history = chat_data.get("chat_history", [])
        if not chat_history:
            return 0
        
        # Parse the conversation snippet JSON
        try:
            snippet_messages = json.loads(conversation_snippet)
            if not snippet_messages:
                return 0
        except json.JSONDecodeError:
            return 0
        
        # Get the first message content from the snippet to search for
        first_snippet_msg = snippet_messages[0]
        first_snippet_content = first_snippet_msg.get("content", "") if isinstance(first_snippet_msg, dict) else str(first_snippet_msg)
        
        if not first_snippet_content:
            return 0
        
        # Find the snippet in chat history by matching the first message content
        snippet_start_idx = -1
        for i, msg in enumerate(chat_history):
            msg_content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            # Ensure both are strings before comparison
            if isinstance(msg_content, str) and isinstance(first_snippet_content, str):
                # Use substring matching to find the snippet
                if first_snippet_content.strip() in msg_content.strip() or msg_content.strip() in first_snippet_content.strip():
                    snippet_start_idx = i
                    break
        
        if snippet_start_idx == -1:
            # Snippet not found in chat history
            return 0
        
        # Calculate tokens from the snippet position to the end of chat history
        token_count = 0
        for i in range(snippet_start_idx, len(chat_history)):
            msg = chat_history[i]
            msg_content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(msg_content, str):
                token_count += count_tokens_in_text(msg_content)
            # Handle multimodal content
            elif isinstance(msg_content, list):
                for item in msg_content:
                    if isinstance(item, dict) and 'text' in item:
                        token_count += count_tokens_in_text(item['text'])
                    elif isinstance(item, str):
                        token_count += count_tokens_in_text(item)
        
        return token_count
        
    except Exception as e:
        print(f"Error processing chat history file {chat_history_file}: {str(e)}")
        return 0


def process_persona_file(file_path: str, persona_number: int, use_multimodal: bool = False) -> List[Dict[str, Any]]:
    """Process a single persona file and extract all user_query rows."""
    rows = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get chat history links for both versions
        chat_history_32k_link, chat_history_128k_link, multimodal_32k_link, multimodal_128k_link = get_chat_history_links(persona_number)
        
        # Select appropriate links based on version type
        if use_multimodal:
            active_32k_link = multimodal_32k_link
            active_128k_link = multimodal_128k_link
        else:
            active_32k_link = chat_history_32k_link
            active_128k_link = chat_history_128k_link
        
        # Process each persona in the file (usually just one)
        for persona_id_key, persona_data in data.items():
            short_persona = persona_data.get("short_persona", "")
            expanded_persona = extract_expanded_persona(persona_data)
            expanded_persona_file = file_path
            
            # Process conversations if they exist
            conversations_data = persona_data.get("conversations", {})
            
            for scenario_name, scenario_items in conversations_data.items():
                if not isinstance(scenario_items, list):
                    continue
                
                for item in scenario_items:
                    # Only include items that have user_query (skip if no user_query)
                    user_query = item.get("user_query", "")
                    if not user_query:
                        continue
                    # Format in OpenAI dict format
                    user_query = {
                        "role": "user",
                        "content": user_query
                    }
                    
                    # Extract all required fields
                    preference = item.get("preference", "sensitive_info")
                    pref_type = item.get("pref_type", "")
                    topic_preference = item.get("topic_preference", "")
                    topic_query = item.get("topic_query", "")
                    correct_answer = item.get("correct_answer", "")
                    incorrect_answers = item.get("incorrect_answers", [])
                    who = item.get("who", "")
                    updated = item.get("updated", False)
                    prev_pref = item.get("prev_pref", "")
                    sensitive_info = True if "sensitive_info" in item else False
                    
                    # Keep conversations as properly formatted JSON string
                    conversations = item.get("conversations", [])
                    conversations_json = json.dumps(conversations, ensure_ascii=False) if conversations else ""
                    
                    # Get total tokens from both chat history versions
                    total_tokens_32k = get_total_tokens_from_chat_history(active_32k_link)
                    
                    # Get detailed token breakdown for 128k version
                    total_tokens_128k = None
                    num_relevant_tokens_128k = None
                    num_irrelevant_tokens_128k = None
                    
                    if active_128k_link:
                        total_tokens_128k, num_relevant_tokens_128k, num_irrelevant_tokens_128k = get_token_breakdown_from_128k_chat_history(active_128k_link)
                    
                    # Calculate tokens from user_query to related_conversation_snippet for both versions
                    tokens_to_snippet_32k = find_conversation_snippet_in_chat_history(
                        active_32k_link, conversations_json, user_query
                    )
                    tokens_to_snippet_128k = find_conversation_snippet_in_chat_history(
                        active_128k_link, conversations_json, user_query
                    ) if active_128k_link else None
                    
                    # Skip this row if either distance_from_related_snippet_to_query is 0
                    if tokens_to_snippet_32k == 0 or tokens_to_snippet_128k == 0:
                        continue
                    
                    # Create row for this user_query
                    row = {
                        "persona_id": persona_number,
                        "chat_history_32k_link": active_32k_link,
                        "chat_history_128k_link": active_128k_link if active_128k_link else None,
                        "raw_persona_file": expanded_persona_file,
                        "short_persona": short_persona,
                        "expanded_persona": expanded_persona,
                        "user_query": user_query,
                        "correct_answer": correct_answer,
                        "incorrect_answers": json.dumps(incorrect_answers, ensure_ascii=False) if incorrect_answers else "",
                        "topic_query": topic_query,
                        "preference": preference,
                        "topic_preference": topic_preference,
                        "conversation_scenario": scenario_name,
                        "pref_type": pref_type,
                        "related_conversation_snippet": conversations_json,
                        "who": who,
                        "updated": str(updated),
                        "prev_pref": prev_pref,
                        "sensitive_info": sensitive_info,
                        "total_tokens_in_chat_history_32k": total_tokens_32k,
                        "total_tokens_in_chat_history_128k": total_tokens_128k,
                        "distance_from_related_snippet_to_query_32k": tokens_to_snippet_32k,
                        "distance_from_related_snippet_to_query_128k": tokens_to_snippet_128k,
                        # Token breakdown for 32k version (all tokens are relevant)
                        "num_persona_relevant_tokens_32k": total_tokens_32k,
                        "num_persona_irrelevant_tokens_32k": 0,
                        # Token breakdown for 128k version (from metadata)
                        "num_persona_relevant_tokens_128k": num_relevant_tokens_128k,
                        "num_persona_irrelevant_tokens_128k": num_irrelevant_tokens_128k
                    }
                    
                    rows.append(row)
        
        print(f"Processed {file_path}: found {len(rows)} user queries")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    
    return rows


def create_benchmark_csv(raw_data_dir: str, output_file: str, use_multimodal: bool = False) -> None:
    """Create comprehensive benchmark CSV from all persona files."""
    
    # Find all JSON files in raw_data directory
    json_pattern = os.path.join(raw_data_dir, "raw_data_*_persona*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {raw_data_dir}")
        return
    
    print(f"Found {len(json_files)} persona files to process")
    
    # Extract persona numbers and sort files by persona number (0 to 999)
    file_persona_pairs = []
    for json_file in json_files:
        persona_number = extract_persona_number(os.path.basename(json_file))
        if persona_number is not None:
            file_persona_pairs.append((json_file, persona_number))
        else:
            print(f"Warning: Could not extract persona number from {json_file}") 
    
    # Sort by persona number
    file_persona_pairs.sort(key=lambda x: x[1])
    
    if not file_persona_pairs:
        print("No valid persona files found")
        return
    
    print(f"Processing personas from {file_persona_pairs[0][1]} to {file_persona_pairs[-1][1]}")
    
    # Collect all rows from all personas
    all_rows = []
    
    # Process each file in order (persona 0 to 999)
    for file_path, persona_number in tqdm(file_persona_pairs, desc="Processing personas", total=len(file_persona_pairs)):
        persona_rows = process_persona_file(file_path, persona_number, use_multimodal)
        all_rows.extend(persona_rows)

    # Write to CSV file
    if all_rows:
        # Define column order as specified
        fieldnames = [
            "persona_id",
            "chat_history_32k_link", 
            "chat_history_128k_link",
            "raw_persona_file",
            "short_persona",
            "expanded_persona",
            "user_query",
            "correct_answer",
            "incorrect_answers",
            "topic_query",
            "preference",
            "topic_preference",
            "conversation_scenario",
            "pref_type",
            "related_conversation_snippet",
            "who",
            "updated",
            "prev_pref",
            "sensitive_info",
            "total_tokens_in_chat_history_32k",
            "total_tokens_in_chat_history_128k",
            "distance_from_related_snippet_to_query_32k",
            "distance_from_related_snippet_to_query_128k",
            "num_persona_relevant_tokens_32k",
            "num_persona_irrelevant_tokens_32k",
            "num_persona_relevant_tokens_128k",
            "num_persona_irrelevant_tokens_128k"
        ]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\nBenchmark CSV created successfully!")
        print(f"Output file: {output_file}")
        print(f"Total rows: {len(all_rows)}")
        print(f"Total personas processed: {len(file_persona_pairs)}")
        
        # Generate summary statistics
        generate_summary_stats(all_rows)
        
    else:
        print("No user queries found to write to CSV")


def generate_summary_stats(rows: List[Dict[str, Any]]) -> None:
    """Generate and print summary statistics."""
    if not rows:
        return
    
    print(f"\n=== Summary Statistics ===")
    print(f"Total user queries: {len(rows)}")
    
    # Count unique personas
    unique_personas = set(row['persona_id'] for row in rows)
    print(f"Unique personas: {len(unique_personas)}")
    
    # Count by conversation scenario
    scenario_counts = {}
    for row in rows:
        scenario = row['conversation_scenario']
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    
    print(f"\nBy conversation scenario:")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count}")
    
    # Count by preference type
    pref_type_counts = {}
    for row in rows:
        pref_type = row['pref_type']
        pref_type_counts[pref_type] = pref_type_counts.get(pref_type, 0) + 1
    
    print(f"\nBy preference type:")
    for pref_type, count in sorted(pref_type_counts.items()):
        print(f"  {pref_type}: {count}")
    
    # Count updated vs non-updated
    updated_counts = {}
    for row in rows:
        updated = row['updated']
        updated_counts[updated] = updated_counts.get(updated, 0) + 1
    
    print(f"\nBy updated status:")
    for updated, count in sorted(updated_counts.items()):
        print(f"  {updated}: {count}")


def ensure_coverage(df: pd.DataFrame, min_coverage: int = 100) -> pd.DataFrame:
    """
    Ensure that the benchmark set has at least min_coverage items for each unique value
    in the key columns: persona_id, conversation_scenario, pref_type, who, updated, sensitive_info.
    
    Args:
        df: The full dataset
        min_coverage: Minimum number of items per unique value (default 100)
    
    Returns:
        DataFrame subset that ensures coverage
    """
    key_columns = ['persona_id', 'conversation_scenario', 'pref_type', 'who', 'updated', 'sensitive_info']
    
    # Get all unique values for each key column
    coverage_requirements = {}
    for col in key_columns:
        if col in df.columns:
            unique_values = df[col].unique()
            coverage_requirements[col] = unique_values
            print(f"Column '{col}': {len(unique_values)} unique values")
    
    # Start with empty benchmark set
    benchmark_indices = set()
    
    # For each column and its unique values, ensure we have enough samples
    for col, unique_values in coverage_requirements.items():
        print(f"\nEnsuring coverage for column '{col}':")
        
        for value in unique_values:
            # Get all rows with this value
            matching_rows = df[df[col] == value]
            
            if len(matching_rows) < min_coverage:
                # If we don't have enough samples, take all available
                selected_indices = matching_rows.index.tolist()
                print(f"  {col}={value}: Only {len(matching_rows)} available (< {min_coverage}), taking all")
            else:
                # Randomly sample min_coverage items
                selected_indices = matching_rows.sample(n=min_coverage, random_state=42).index.tolist()
                print(f"  {col}={value}: Selected {min_coverage} out of {len(matching_rows)}")
            
            benchmark_indices.update(selected_indices)
    
    print(f"\nTotal unique indices selected for coverage: {len(benchmark_indices)}")
    return df.loc[list(benchmark_indices)]


def stratified_sample_remaining(df: pd.DataFrame, target_size: int, coverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample additional rows to reach target_size while maintaining distribution balance.
    
    Args:
        df: The full dataset
        target_size: Target size for the benchmark set
        coverage_df: DataFrame already selected for coverage
    
    Returns:
        Additional rows to add to the benchmark set
    """
    # Remove already selected rows
    remaining_df = df.drop(coverage_df.index)
    additional_needed = target_size - len(coverage_df)
    
    if additional_needed <= 0:
        print(f"Coverage selection already meets target size ({len(coverage_df)} >= {target_size})")
        return pd.DataFrame()
    
    if len(remaining_df) < additional_needed:
        print(f"Not enough remaining data. Taking all {len(remaining_df)} remaining rows.")
        return remaining_df
    
    # Stratified sampling based on key columns
    key_columns = ['conversation_scenario', 'pref_type', 'who', 'updated']
    
    # Create stratification key
    remaining_df = remaining_df.copy()
    remaining_df['strat_key'] = remaining_df[key_columns].astype(str).agg('_'.join, axis=1)
    
    # Calculate proportional sampling
    strat_counts = remaining_df['strat_key'].value_counts()
    strat_proportions = strat_counts / len(remaining_df)
    
    additional_samples = []
    remaining_to_sample = additional_needed
    
    # Sample proportionally from each stratum
    for strat_key, proportion in strat_proportions.items():
        if remaining_to_sample <= 0:
            break
            
        strat_data = remaining_df[remaining_df['strat_key'] == strat_key]
        n_to_sample = min(int(proportion * additional_needed), len(strat_data), remaining_to_sample)
        
        if n_to_sample > 0:
            sampled = strat_data.sample(n=n_to_sample, random_state=42)
            additional_samples.append(sampled)
            remaining_to_sample -= n_to_sample
    
    # If we still need more samples, randomly sample from remaining
    if remaining_to_sample > 0:
        already_sampled_indices = set()
        for sample_df in additional_samples:
            already_sampled_indices.update(sample_df.index)
        
        still_remaining = remaining_df.drop(list(already_sampled_indices))
        if len(still_remaining) > 0:
            final_sample = still_remaining.sample(n=min(remaining_to_sample, len(still_remaining)), random_state=42)
            additional_samples.append(final_sample)
    
    if additional_samples:
        additional_df = pd.concat(additional_samples, ignore_index=False)
        additional_df = additional_df.drop('strat_key', axis=1)
        return additional_df
    else:
        return pd.DataFrame()


def print_split_summary_statistics(train_df: pd.DataFrame, val_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> None:
    """Print summary statistics for the splits."""
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    datasets = {
        'Train': train_df,
        'Validation': val_df,
        'Benchmark': benchmark_df
    }
    
    # Overall sizes
    print(f"\nDataset Sizes:")
    total_size = len(train_df) + len(val_df) + len(benchmark_df)
    for name, df in datasets.items():
        percentage = len(df) / total_size * 100
        print(f"  {name}: {len(df):,} rows ({percentage:.1f}%)")
    print(f"  Total: {total_size:,} rows")
    
    # Key columns for analysis
    key_columns = ['persona_id', 'conversation_scenario', 'pref_type', 'who', 'updated', 'sensitive_info']
    
    for col in key_columns:
        if col in train_df.columns:
            print(f"\n{col.upper()} Distribution:")
            print(f"{'Value':<20} {'Train':<10} {'Val':<10} {'Benchmark':<12} {'Total':<10}")
            print("-" * 62)
            
            # Get all unique values across all datasets
            all_values = set()
            for df in datasets.values():
                all_values.update(df[col].unique())
            
            for value in sorted(all_values):
                counts = []
                for name, df in datasets.items():
                    count = len(df[df[col] == value])
                    counts.append(count)
                
                total_count = sum(counts)
                value_str = str(value)[:19]  # Truncate long values
                print(f"{value_str:<20} {counts[0]:<10} {counts[1]:<10} {counts[2]:<12} {total_count:<10}")
    
    # Check coverage in benchmark set
    print(f"\nBENCHMARK COVERAGE ANALYSIS:")
    print("-" * 40)
    
    for col in key_columns:
        if col in benchmark_df.columns:
            value_counts = benchmark_df[col].value_counts()
            min_count = value_counts.min()
            max_count = value_counts.max()
            mean_count = value_counts.mean()
            
            print(f"{col}:")
            print(f"  Unique values: {len(value_counts)}")
            print(f"  Min coverage: {min_count}")
            print(f"  Max coverage: {max_count}")
            print(f"  Mean coverage: {mean_count:.1f}")
            
            # Show values with low coverage
            low_coverage = value_counts[value_counts < 100]
            if len(low_coverage) > 0:
                print(f"  Values with <100 coverage: {len(low_coverage)}")
                for val, count in low_coverage.items():
                    print(f"    {val}: {count}")
            print()


def split_benchmark_data(input_file: str, benchmark_size: int = 5000, train_val_split: float = 0.8, 
                        min_coverage: int = 100, random_seed: int = 42) -> None:
    """
    Split benchmark data into train, validation, and benchmark sets.
    
    Args:
        input_file: Path to the input CSV file
        benchmark_size: Size of the benchmark set (default 5000)
        train_val_split: Ratio for train/val split of remaining data (default 0.8)
        min_coverage: Minimum coverage per variable value (default 100)
        random_seed: Random seed for reproducibility
    """
    print(f"Loading data from {input_file}...")
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle the entire dataset first
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"Data shuffled with random seed {random_seed}")
    
    # Step 1: Ensure coverage for benchmark set
    print(f"\nStep 1: Ensuring coverage of at least {min_coverage} items per variable...")
    coverage_df = ensure_coverage(df_shuffled, min_coverage)
    
    # Step 2: Sample additional rows to reach target benchmark size
    print(f"\nStep 2: Sampling additional rows to reach benchmark size of {benchmark_size}...")
    additional_df = stratified_sample_remaining(df_shuffled, benchmark_size, coverage_df)
    
    # Combine coverage and additional samples for benchmark set
    if len(additional_df) > 0:
        benchmark_df = pd.concat([coverage_df, additional_df]).drop_duplicates()
    else:
        benchmark_df = coverage_df
    
    # Trim to exact target size if needed
    if len(benchmark_df) > benchmark_size:
        benchmark_df = benchmark_df.sample(n=benchmark_size, random_state=random_seed)
    
    print(f"Final benchmark set size: {len(benchmark_df)}")
    
    # Step 3: Split remaining data into train and validation
    remaining_df = df_shuffled.drop(benchmark_df.index)
    
    # Calculate train/val sizes
    train_size = int(len(remaining_df) * train_val_split)
    val_size = len(remaining_df) - train_size
    
    print(f"\nStep 3: Splitting remaining {len(remaining_df)} rows into train/val...")
    print(f"Train size: {train_size} ({train_val_split:.1%})")
    print(f"Validation size: {val_size} ({1-train_val_split:.1%})")
    
    # Shuffle remaining data and split
    remaining_shuffled = remaining_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train_df = remaining_shuffled.iloc[:train_size]
    val_df = remaining_shuffled.iloc[train_size:]
    
    # Generate output filenames
    input_path = Path(input_file)
    output_dir = input_path.parent
    
    train_file = output_dir / "train.csv"
    val_file = output_dir / "val.csv"
    benchmark_file = output_dir / "benchmark.csv"
    
    # Save the splits
    print(f"\nSaving splits...")
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    benchmark_df.to_csv(benchmark_file, index=False)
    
    print(f"Train set saved to: {train_file}")
    print(f"Validation set saved to: {val_file}")
    print(f"Benchmark set saved to: {benchmark_file}")
    
    # Generate summary statistics
    print_split_summary_statistics(train_df, val_df, benchmark_df)


def main():
    """Main function to create benchmark CSV files and optionally split them."""
    parser = argparse.ArgumentParser(description="Prepare benchmark data from raw persona files")
    parser.add_argument("--raw-data-dir", type=str, default="data/raw_data",
                       help="Directory containing raw persona JSON files (default: data/raw_data)")
    parser.add_argument("--output-text", type=str, default="benchmark/text/benchmark.csv",
                       help="Output file for text benchmark (default: benchmark/text/benchmark.csv)")
    parser.add_argument("--output-multimodal", type=str, default="benchmark/multimodal/benchmark.csv",
                       help="Output file for multimodal benchmark (default: benchmark/multimodal/benchmark.csv)")
    parser.add_argument("--split", action="store_true",
                       help="Split the generated benchmark files into train/val/benchmark sets")
    parser.add_argument("--benchmark-size", type=int, default=5000,
                       help="Size of benchmark set when splitting (default: 5000)")
    parser.add_argument("--train-val-split", type=float, default=0.8,
                       help="Train/validation split ratio when splitting (default: 0.8)")
    parser.add_argument("--min-coverage", type=int, default=100,
                       help="Minimum coverage per variable value when splitting (default: 100)")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility when splitting (default: 42)")
    
    args = parser.parse_args()
    
    print("Creating comprehensive benchmark CSV files...")
    print(f"Input directory: {args.raw_data_dir}")
    
    # Create text-only benchmark CSV
    print(f"\nCreating text-only benchmark: {args.output_text}")
    create_benchmark_csv(args.raw_data_dir, args.output_text, use_multimodal=False)
    
    # Create multimodal benchmark CSV
    print(f"\nCreating multimodal benchmark: {args.output_multimodal}")
    create_benchmark_csv(args.raw_data_dir, args.output_multimodal, use_multimodal=True)
    
    print("\nBenchmark preparation complete!")
    print(f"Generated files:")
    print(f"  - {args.output_text}")
    print(f"  - {args.output_multimodal}")
    
    # Optionally split the benchmark files
    if args.split:
        print(f"\n{'='*80}")
        print("SPLITTING BENCHMARK FILES")
        print(f"{'='*80}")
        print(f"Split parameters:")
        print(f"  Benchmark size: {args.benchmark_size}")
        print(f"  Train/val split: {args.train_val_split:.1%}/{1-args.train_val_split:.1%}")
        print(f"  Minimum coverage: {args.min_coverage}")
        print(f"  Random seed: {args.random_seed}")
        
        # Split text benchmark
        if os.path.exists(args.output_text):
            print(f"\n{'-'*60}")
            print(f"Splitting text benchmark: {args.output_text}")
            print(f"{'-'*60}")
            split_benchmark_data(
                args.output_text,
                args.benchmark_size,
                args.train_val_split,
                args.min_coverage,
                args.random_seed
            )
        
        # Split multimodal benchmark
        if os.path.exists(args.output_multimodal):
            print(f"\n{'-'*60}")
            print(f"Splitting multimodal benchmark: {args.output_multimodal}")
            print(f"{'-'*60}")
            split_benchmark_data(
                args.output_multimodal,
                args.benchmark_size,
                args.train_val_split,
                args.min_coverage,
                args.random_seed
            )
        
        print(f"\n{'='*80}")
        print("SPLITTING COMPLETE")
        print(f"{'='*80}")


if __name__ == "__main__":
    main() 