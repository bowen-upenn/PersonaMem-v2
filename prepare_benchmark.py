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
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


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


def get_chat_history_links(persona_number: int) -> Tuple[str, str]:
    """Generate chat history file links for a persona."""
    # Look for chat history files with the pattern
    chat_history_pattern = f"data/chat_history/chat_history_*_persona{persona_number}.json"
    multimodal_pattern = f"data/chat_history_multimodal/chat_history_*_persona{persona_number}.json"
    
    chat_history_files = glob.glob(chat_history_pattern)
    multimodal_files = glob.glob(multimodal_pattern)
    
    chat_history_link = chat_history_files[0] if chat_history_files else ""
    multimodal_link = multimodal_files[0] if multimodal_files else ""
    
    return chat_history_link, multimodal_link


def process_persona_file(file_path: str, persona_number: int) -> List[Dict[str, Any]]:
    """Process a single persona file and extract all user_query rows."""
    rows = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get chat history links
        chat_history_link, multimodal_link = get_chat_history_links(persona_number)
        
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
                    preference = item.get("preference", "")
                    pref_type = item.get("pref_type", "")
                    topic_preference = item.get("topic_preference", "")
                    topic_query = item.get("topic_query", "")
                    correct_answer = item.get("correct_answer", "")
                    incorrect_answers = item.get("incorrect_answers", [])
                    who = item.get("who", "")
                    updated = item.get("updated", False)
                    prev_pref = item.get("prev_pref", "")
                    
                    # Keep conversations as properly formatted JSON string
                    conversations = item.get("conversations", [])
                    conversations_json = json.dumps(conversations, ensure_ascii=False) if conversations else ""
                    
                    # Create row for this user_query
                    row = {
                        "persona_id": persona_number,
                        "chat_history_link": chat_history_link,
                        "chat_history_multimodal_link": multimodal_link,
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
                        "prev_pref": prev_pref
                    }
                    
                    rows.append(row)
        
        print(f"Processed {file_path}: found {len(rows)} user queries")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    
    return rows


def create_benchmark_csv(raw_data_dir: str, output_file: str) -> None:
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
    for file_path, persona_number in file_persona_pairs:
        persona_rows = process_persona_file(file_path, persona_number)
        all_rows.extend(persona_rows)
    
    # Write to CSV file
    if all_rows:
        # Define column order as specified
        fieldnames = [
            "persona_id",
            "chat_history_link", 
            "chat_history_multimodal_link",
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
            "prev_pref"
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
    """Main function to create benchmark CSV."""
    raw_data_dir = "data/raw_data"
    output_file = "data/benchmark.csv"
    
    print("Creating comprehensive benchmark CSV...")
    print(f"Input directory: {raw_data_dir}")
    print(f"Output file: {output_file}")
    
    create_benchmark_csv(raw_data_dir, output_file)
    print("\nBenchmark preparation complete!")


if __name__ == "__main__":
    main() 