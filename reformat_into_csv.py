#!/usr/bin/env python3
"""
Script to reformat JSON persona files into CSV format for better visualization on HuggingFace.
Each row represents one conversation snippet with specified columns.
"""

import json
import csv
import os
import glob
import re
from typing import Dict, List, Any, Optional


def extract_persona_number(filename: str) -> Optional[int]:
    """Extract persona number from filename ending with 'personaX.json'."""
    match = re.search(r'persona(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return None


def extract_expanded_persona(persona_data: Dict[str, Any]) -> str:
    """Extract everything before 'stereotypical_preferences' as expanded persona."""
    expanded_persona = {}
    
    for key, value in persona_data.items():
        if key == "stereotypical_preferences":
            break
        expanded_persona[key] = value
    
    return json.dumps(expanded_persona, indent=2)


def process_conversation_section(section_name: str, section_data: List[Dict], 
                               short_persona: str, expanded_persona: str, persona_id: int) -> List[Dict]:
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
            "persona_id": persona_id,
            "short_persona": short_persona,
            "preference": preference,
            "conversation_scenario": section_name,
            "pref_type": pref_type,
            "topic_preference": topic_preference,
            "conversations": conversations_json,
            "user_query": user_query,
            "topic_query": topic_query,
            "correct_answer": correct_answer,
            "incorrect_answers": incorrect_answers,
            "who": who,
            "updated": str(updated),
            "prev_pref": prev_pref
        }
        
        rows.append(row)
    
    return rows


def process_all_json_files(input_dir: str, output_dir: str) -> None:
    """Process all JSON files and create one giant CSV file and metadata file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_pattern = os.path.join(input_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Sort files by persona number (0 to 999)
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
    
    # Initialize collections for all data
    all_csv_rows = []
    
    # Process each JSON file in sorted order
    for json_file, persona_number in file_persona_pairs:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each persona in the JSON file
            for persona_id, persona_data in data.items():
                short_persona = persona_data.get("short_persona", "")
                expanded_persona = extract_expanded_persona(persona_data)
                
                # Add special row with expanded persona at the beginning of each persona
                metadata_row = {
                    "persona_id": f"{persona_number} (full persona)",
                    "short_persona": expanded_persona,  # Put expanded persona here
                    "preference": "",
                    "conversation_scenario": "",
                    "pref_type": "",
                    "topic_preference": "",
                    "conversations": "",
                    "user_query": "",
                    "topic_query": "",
                    "correct_answer": "",
                    "incorrect_answers": "",
                    "who": "",
                    "updated": "",
                    "prev_pref": ""
                }
                all_csv_rows.append(metadata_row)
                
                # Process conversations if they exist
                if "conversations" in persona_data:
                    conversations_data = persona_data["conversations"]
                    
                    # Process all conversation scenarios
                    for scenario_name, scenario_data in conversations_data.items():
                        if isinstance(scenario_data, list):  # Ensure it's a list of conversation items
                            scenario_rows = process_conversation_section(
                                scenario_name, scenario_data, short_persona, expanded_persona, persona_number
                            )
                            # Remove expanded_persona from each row since we're not including it in CSV
                            for row in scenario_rows:
                                if 'expanded_persona' in row:
                                    del row['expanded_persona']
                            all_csv_rows.extend(scenario_rows)
            
            print(f"Processed {json_file}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Write the giant CSV file
    if all_csv_rows:
        output_csv = os.path.join(output_dir, "all_personas_preview.csv")
        fieldnames = [
            "persona_id", "short_persona", "preference", "conversation_scenario", "pref_type", 
            "topic_preference", "conversations", "user_query", "topic_query", "correct_answer", 
            "incorrect_answers", "who", "updated", "prev_pref"
        ]
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        
        print(f"Created giant CSV file: {output_csv}")
        print(f"Total rows: {len(all_csv_rows)}")
        print(f"Expanded persona data included as special rows at the beginning of each persona")


def main():
    """Main function to process all JSON files."""
    input_directory = "data/raw_data"
    output_directory = "data/raw_data_csv"
    
    print("Starting JSON to CSV conversion...")
    process_all_json_files(input_directory, output_directory)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
