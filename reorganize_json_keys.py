#!/usr/bin/env python3
"""
Script to reorganize JSON persona files with proper key ordering:
1. "short_persona" comes first
2. All other unfixed persona attributes come next  
3. Fixed final keys in specific order: stereotypical_preferences, anti_stereotypical_preferences,
   neutral_preferences, therapy_background, health_and_medical_conditions, sensitive_information,
   matched_images, preference_updates, conversations
Also removes any "alternate_personas" or "alternate_persona" keys from the files.
"""

import json
import os
from pathlib import Path
from collections import OrderedDict


def move_multimodal_to_bottom(conversations_dict):
    """
    Move the "multimodal" key and its values to the bottom of the conversations dictionary.
    
    Args:
        conversations_dict (dict): The conversations dictionary
        
    Returns:
        dict: The reorganized conversations dictionary with multimodal at the bottom
    """
    if not isinstance(conversations_dict, dict) or "multimodal" not in conversations_dict:
        return conversations_dict
    
    # Create a new ordered dictionary
    reorganized_conversations = OrderedDict()
    
    # Add all keys except "multimodal" first
    for key, value in conversations_dict.items():
        if key != "multimodal":
            reorganized_conversations[key] = value
    
    # Add "multimodal" at the end
    reorganized_conversations["multimodal"] = conversations_dict["multimodal"]
    
    return reorganized_conversations


def reorganize_persona_keys(data):
    """
    Reorganize the keys in a persona dictionary to ensure proper ordering:
    1. "short_persona" comes first
    2. All other unfixed persona attributes come next
    3. Fixed final keys come in this order: stereotypical_preferences, 
       anti_stereotypical_preferences, neutral_preferences, therapy_background, 
       health_and_medical_conditions, sensitive_information, matched_images, 
       preference_updates, conversations
    Also removes any "alternate_personas" or "alternate_persona" keys.
    Additionally, moves the "multimodal" key to the bottom of the "conversations" dictionary.
    
    Args:
        data (dict): The persona data dictionary
        
    Returns:
        dict: The reorganized dictionary
    """
    if not isinstance(data, dict):
        return data
    
    # Define the fixed order of final keys
    final_keys_order = [
        "stereotypical_preferences",
        "anti_stereotypical_preferences", 
        "neutral_preferences",
        "therapy_background",
        "health_and_medical_conditions",
        "sensitive_information",
        "matched_images",
        "preference_updates",
        "conversations"
    ]
    
    # Process each persona in the data
    reorganized_data = {}
    
    for persona_id, persona_info in data.items():
        if not isinstance(persona_info, dict):
            reorganized_data[persona_id] = persona_info
            continue
        
        # Remove alternate_personas or alternate_persona keys if they exist
        keys_to_remove = ['alternate_personas', 'alternate_persona', 'persona_variations', 'persona_variation', 'alternative_personas', 'alternative_persona']
        persona_info_copy = persona_info.copy()
        removed_keys = []
        
        for key_to_remove in keys_to_remove:
            if key_to_remove in persona_info_copy:
                del persona_info_copy[key_to_remove]
                removed_keys.append(key_to_remove)
        
        if removed_keys:
            print(f"  Removed keys {removed_keys} from persona {persona_id}")
            
        keys_list = list(persona_info_copy.keys())
        
        # Check if "short_persona" exists
        if "short_persona" not in keys_list:
            # If "short_persona" doesn't exist, keep the original order but still reorganize conversations
            if "conversations" in persona_info_copy:
                persona_info_copy["conversations"] = move_multimodal_to_bottom(persona_info_copy["conversations"])
            reorganized_data[persona_id] = persona_info_copy
            continue
        
        # Build the new order
        new_ordered_dict = OrderedDict()
        
        # 1. First, add "short_persona"
        new_ordered_dict["short_persona"] = persona_info_copy["short_persona"]
        
        # 2. Then, add all other keys that are NOT in the final_keys_order and NOT "short_persona"
        for key in keys_list:
            if key != "short_persona" and key not in final_keys_order:
                new_ordered_dict[key] = persona_info_copy[key]
        
        # 3. Finally, add the final keys in their specified order (only if they exist)
        for final_key in final_keys_order:
            if final_key in persona_info_copy:
                if final_key == "conversations":
                    # Reorganize conversations to move multimodal to bottom
                    new_ordered_dict[final_key] = move_multimodal_to_bottom(persona_info_copy[final_key])
                else:
                    new_ordered_dict[final_key] = persona_info_copy[final_key]
            
        reorganized_data[persona_id] = new_ordered_dict
    
    return reorganized_data


def process_json_files(raw_data_dir):
    """
    Process all JSON files in the raw_data directory and reorganize their keys.
    Also moves the "multimodal" key to the bottom of the "conversations" dictionary.
    
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
    print("- Ensuring 'short_persona' is the first key")
    print("- Moving unfixed persona attributes after 'short_persona'")
    print("- Organizing final keys in fixed order: stereotypical_preferences, anti_stereotypical_preferences, neutral_preferences, therapy_background, health_and_medical_conditions, sensitive_information, matched_images, preference_updates, conversations")
    print("- Removing alternate persona keys")
    print("- Moving 'multimodal' key to bottom of 'conversations' dictionary")
    print(f"Processing files in: {raw_data_dir}")
    
    process_json_files(raw_data_dir)
    
    print("Process completed!")


if __name__ == "__main__":
    main()
