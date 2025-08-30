#!/usr/bin/env python3
"""
Script to reformat JSON persona files into CSV format for better visualization on HuggingFace.
Each row represents one conversation snippet with specified columns.
"""

import json
import csv
import os
import glob
from typing import Dict, List, Any, Optional


def extract_expanded_persona(persona_data: Dict[str, Any]) -> str:
    """Extract everything before 'stereotypical_preferences' as expanded persona."""
    expanded_persona = {}
    
    for key, value in persona_data.items():
        if key == "stereotypical_preferences":
            break
        expanded_persona[key] = value
    
    return json.dumps(expanded_persona, indent=2)


def process_conversation_section(section_name: str, section_data: List[Dict], 
                               short_persona: str, expanded_persona: str) -> List[Dict]:
    """Process a conversation section (e.g., persona_email, persona_text) and return CSV rows."""
    rows = []
    
    for item in section_data:
        preference = item.get("preference", "")
        pref_type = item.get("pref_type", "")
        who = item.get("who", "")
        updated = item.get("updated", "")
        prev_pref = item.get("prev_pref", "") if updated else ""
        topic_preference = item.get("topic_preference", "")
        topic_query = item.get("topic_query", "")
        user_query = item.get("user_query", "")
        correct_answer = item.get("correct_answer", "")
        incorrect_answers = json.dumps(item.get("incorrect_answers", [])) if item.get("incorrect_answers") else ""
        
        # Keep conversations as properly formatted JSON string
        conversations = item.get("conversations", [])
        conversations_json = json.dumps(conversations) if conversations else ""
        
        row = {
            "short_persona": short_persona,
            "preference": preference,
            "conversation_scenario": section_name,
            "pref_type": pref_type,
            "topic_preference": topic_preference,
            "conversations": conversations_json,
            "topic_query": topic_query,
            "user_query": user_query,
            "correct_answer": correct_answer,
            "incorrect_answers": incorrect_answers,
            "who": who,
            "updated": str(updated),
            "prev_pref": prev_pref,
            "expanded_persona": expanded_persona  # Moved to rightmost position
        }
        
        rows.append(row)
    
    return rows


def json_to_csv(json_file_path: str, output_csv_path: str) -> None:
    """Convert a single JSON file to CSV format."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    csv_rows = []
    
    # Process each persona in the JSON file
    for persona_id, persona_data in data.items():
        short_persona = persona_data.get("short_persona", "")
        expanded_persona = extract_expanded_persona(persona_data)
        
        # Process conversations if they exist
        if "conversations" in persona_data:
            conversations_data = persona_data["conversations"]
            
            # Process all conversation scenarios that exist in the data
            for scenario_name, scenario_data in conversations_data.items():
                if isinstance(scenario_data, list):  # Ensure it's a list of conversation items
                    scenario_rows = process_conversation_section(
                        scenario_name, scenario_data, short_persona, expanded_persona
                    )
                    csv_rows.extend(scenario_rows)
    
    # Write to CSV
    if csv_rows:
        fieldnames = [
            "short_persona", "preference", "conversation_scenario", "pref_type", 
            "topic_preference", "conversations", "topic_query", "user_query", "correct_answer", 
            "incorrect_answers", "who", "updated", "prev_pref", "expanded_persona"
        ]
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"Successfully converted {json_file_path} to {output_csv_path}")
        print(f"Created {len(csv_rows)} rows")
    else:
        print(f"No conversation data found in {json_file_path}")


def process_all_json_files(input_dir: str, output_dir: str) -> None:
    """Process all JSON files in the input directory."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_pattern = os.path.join(input_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        # Create output CSV filename
        base_name = os.path.basename(json_file)
        csv_name = base_name.replace('.json', '.csv')
        output_csv = os.path.join(output_dir, csv_name)
        
        try:
            json_to_csv(json_file, output_csv)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")


def main():
    """Main function to process all JSON files."""
    input_directory = "data/raw_data"
    output_directory = "data/raw_data_csv"
    
    print("Starting JSON to CSV conversion...")
    process_all_json_files(input_directory, output_directory)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
