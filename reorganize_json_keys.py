#!/usr/bin/env python3
"""
Script to reorganize JSON persona files by moving all keys that appear before 
"short_persona" to positions right before "stereotypical_preferences".
"""

import json
import os
from pathlib import Path
from collections import OrderedDict


def reorganize_persona_keys(data):
    """
    Reorganize the keys in a persona dictionary by moving keys that appear 
    before "short_persona" to positions right before "stereotypical_preferences".
    
    Args:
        data (dict): The persona data dictionary
        
    Returns:
        dict: The reorganized dictionary
    """
    if not isinstance(data, dict):
        return data
    
    # Process each persona in the data
    reorganized_data = {}
    
    for persona_id, persona_info in data.items():
        if not isinstance(persona_info, dict):
            reorganized_data[persona_id] = persona_info
            continue
            
        # Find the position of "short_persona" and "stereotypical_preferences"
        keys_list = list(persona_info.keys())
        
        try:
            short_persona_idx = keys_list.index("short_persona")
        except ValueError:
            # If "short_persona" doesn't exist, keep the original order
            reorganized_data[persona_id] = persona_info
            continue
            
        try:
            stereotypical_prefs_idx = keys_list.index("stereotypical_preferences")
        except ValueError:
            # If "stereotypical_preferences" doesn't exist, keep the original order
            reorganized_data[persona_id] = persona_info
            continue
        
        # Identify keys that come before "short_persona"
        keys_before_short_persona = keys_list[:short_persona_idx]
        
        # Build the new order
        new_ordered_dict = OrderedDict()
        
        # First, add "short_persona" and all keys after it until "stereotypical_preferences"
        for i in range(short_persona_idx, stereotypical_prefs_idx):
            key = keys_list[i]
            new_ordered_dict[key] = persona_info[key]
            
        # Then, add the keys that were originally before "short_persona"
        for key in keys_before_short_persona:
            new_ordered_dict[key] = persona_info[key]
            
        # Finally, add "stereotypical_preferences" and all remaining keys
        for i in range(stereotypical_prefs_idx, len(keys_list)):
            key = keys_list[i]
            new_ordered_dict[key] = persona_info[key]
            
        reorganized_data[persona_id] = new_ordered_dict
    
    return reorganized_data


def process_json_files(raw_data_dir):
    """
    Process all JSON files in the raw_data directory and reorganize their keys.
    
    Args:
        raw_data_dir (str): Path to the directory containing JSON files
    """
    raw_data_path = Path(raw_data_dir)
    
    if not raw_data_path.exists():
        print(f"Directory {raw_data_dir} does not exist!")
        return
    
    # Find all JSON files
    json_files = list(raw_data_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {raw_data_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        
        try:
            # Read the original file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reorganize the keys
            reorganized_data = reorganize_persona_keys(data)
            
            # Write back to the same file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(reorganized_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Successfully processed {json_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")


def main():
    """Main function to run the reorganization process."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    raw_data_dir = script_dir / "data" / "raw_data"
    
    print("Starting JSON key reorganization process...")
    print(f"Processing files in: {raw_data_dir}")
    
    process_json_files(raw_data_dir)
    
    print("Process completed!")


if __name__ == "__main__":
    main()
