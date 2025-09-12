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


def main():
    """Main function to create benchmark CSV files."""
    raw_data_dir = "data/raw_data"
    output_file_text = "data/benchmark.csv"
    output_file_multimodal = "data/benchmark_multimodal.csv"
    
    print("Creating comprehensive benchmark CSV files...")
    print(f"Input directory: {raw_data_dir}")
    
    # Create text-only benchmark CSV
    print(f"\nCreating text-only benchmark: {output_file_text}")
    create_benchmark_csv(raw_data_dir, output_file_text, use_multimodal=False)
    
    # Create multimodal benchmark CSV
    print(f"\nCreating multimodal benchmark: {output_file_multimodal}")
    create_benchmark_csv(raw_data_dir, output_file_multimodal, use_multimodal=True)
    
    print("\nBenchmark preparation complete!")
    print(f"Generated files:")
    print(f"  - {output_file_text}")
    print(f"  - {output_file_multimodal}")


if __name__ == "__main__":
    main() 