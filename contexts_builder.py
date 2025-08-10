import random
import tiktoken
import json
from uuid import uuid4
import re

import utils

ENCODER = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text) -> int:
    if isinstance(text, str):
        return len(ENCODER.encode(text))
    elif isinstance(text, list):
        total_tokens = 0
        for msg in text:
            if isinstance(msg, dict) and 'content' in msg:
                content = msg['content']
                if isinstance(content, str):
                    total_tokens += len(ENCODER.encode(content))
                elif isinstance(content, list):
                    # Handle multimodal content (e.g., text + image)
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            total_tokens += len(ENCODER.encode(item['text']))
                        elif isinstance(item, str):
                            total_tokens += len(ENCODER.encode(item))
                else:
                    # Fallback: convert to string
                    total_tokens += len(ENCODER.encode(str(content)))
            else:
                # Fallback: convert to string
                total_tokens += len(ENCODER.encode(str(msg)))
        return total_tokens
    elif isinstance(text, dict):
        if 'content' in text:
            content = text['content']
            if isinstance(content, str):
                return len(ENCODER.encode(content))
            elif isinstance(content, list):
                # Handle multimodal content
                total_tokens = 0
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        total_tokens += len(ENCODER.encode(item['text']))
                    elif isinstance(item, str):
                        total_tokens += len(ENCODER.encode(item))
                return total_tokens
            else:
                return len(ENCODER.encode(str(content)))
        else:
            return len(ENCODER.encode(str(text)))
    else:
        raise ValueError("Input must be a string, dict, or list of dicts.")


def extract_conversation_blocks(interactions):
    """
    Extract all conversation blocks from interactions and return them as a list.
    Each block contains the conversation messages and metadata.
    """
    all_blocks = []
    
    for uuid, persona in interactions.items():
        conversations = persona.get("conversations", {})
        
        for conv_type, blocks in conversations.items():
            for block in blocks:
                msgs = block.get("conversations", [])
                preference = block.get('preference', '')
                
                # Skip blocks with null, empty, or missing preferences
                if not preference:
                    continue
                    
                if msgs:  # Only add non-empty conversation blocks
                    # Create a block with metadata for ordering
                    block_info = {
                        'messages': msgs,
                        'preference': preference,
                        'prev_pref': block.get('prev_pref', None),
                        'updated': block.get('updated', False),
                        'pref_type': block.get('pref_type', ''),
                        'who': block.get('who', ''),
                        'conv_type': conv_type
                    }
                    all_blocks.append(block_info)
    
    return all_blocks


def order_conversations_by_dependencies(blocks):
    """
    Order conversation blocks based on dependencies:
    1. Blocks with prev_pref must come after blocks with that preference
    2. "Do not remember" blocks must come at the end of their sequence
    3. Other blocks are shuffled randomly
    """
    # Create mappings for tracking dependencies
    preference_to_blocks = {}
    prev_pref_to_blocks = {}
    do_not_remember_blocks = []
    independent_blocks = []
    
    # Categorize blocks
    for block in blocks:
        preference = block['preference']  # We know this is not None due to filtering
        prev_pref = block['prev_pref']
        
        # Check if this is a "Do not remember" block
        if preference.startswith("Do not remember"):
            do_not_remember_blocks.append(block)
        else:
            # Regular block
            if preference not in preference_to_blocks:
                preference_to_blocks[preference] = []
            preference_to_blocks[preference].append(block)
            
            # Track blocks that have prev_pref
            if prev_pref:
                if prev_pref not in prev_pref_to_blocks:
                    prev_pref_to_blocks[prev_pref] = []
                prev_pref_to_blocks[prev_pref].append(block)
            else:
                independent_blocks.append(block)
    
    # Build ordered sequence
    ordered_blocks = []
    
    # Step 1: Add all independent blocks first (shuffled)
    random.shuffle(independent_blocks)
    ordered_blocks.extend(independent_blocks)
    
    # Step 2: Add updated blocks (those with prev_pref) after their original preference
    for original_pref, updated_blocks in prev_pref_to_blocks.items():
        # Find where the original preference blocks are in the ordered list
        original_positions = []
        for i, block in enumerate(ordered_blocks):
            if block['preference'] == original_pref:
                original_positions.append(i)
        
        if original_positions:
            # Insert updated blocks after the last occurrence of the original preference
            insert_position = max(original_positions) + 1
            for block in updated_blocks:
                ordered_blocks.insert(insert_position, block)
                insert_position += 1
        else:
            # If original preference not found, add at the end
            ordered_blocks.extend(updated_blocks)
    
    # Step 3: Add "Do not remember" blocks at the end of their sequences
    for block in do_not_remember_blocks:
        # Extract the original preference from "Do not remember 'X' in memory"
        match = re.search(r"Do not remember '([^']+)' in memory", block['preference'])
        if match:
            original_preference = match.group(1)
            # Find the position after all blocks with this preference or prev_pref
            insert_position = len(ordered_blocks)
            for i, existing_block in enumerate(ordered_blocks):
                if (existing_block['preference'] == original_preference or 
                    existing_block['prev_pref'] == original_preference):
                    insert_position = i + 1
            
            # Insert at the calculated position
            ordered_blocks.insert(insert_position, block)
        else:
            # If we can't parse the preference, add at the end
            ordered_blocks.append(block)
    
    return ordered_blocks


def build_context(interactions, context_len=None, input_filename=None):
    """
    Constructs a multi-turn conversation list for each persona by extracting all conversations,
    ordering them based on dependencies, and optionally trimming by token budget.
    Saves flattened messages to JSON and returns a list of dictionaries:
        [ {role, content}, ... ]
    """
    for uuid, persona in interactions.items():
        # Extract all conversation blocks
        all_blocks = extract_conversation_blocks({uuid: persona})
        
        if not all_blocks:
            print(f"No conversation blocks found for {uuid}")
            continue
        
        print(f"Processing {len(all_blocks)} conversation blocks for {uuid}")
        
        # Debug: Show some block information
        independent_count = sum(1 for b in all_blocks if not b['prev_pref'] and not b['preference'].startswith("Do not remember"))
        updated_count = sum(1 for b in all_blocks if b['prev_pref'])
        do_not_remember_count = sum(1 for b in all_blocks if b['preference'].startswith("Do not remember"))
        
        print(f"  Independent blocks: {independent_count}")
        print(f"  Updated blocks: {updated_count}")
        print(f"  Do not remember blocks: {do_not_remember_count}")
        
        # Order conversations based on dependencies
        ordered_blocks = order_conversations_by_dependencies(all_blocks)
        
        print(f"  Ordered {len(ordered_blocks)} blocks")
        
        # Debug: Show ordering verification
        for i, block in enumerate(ordered_blocks[:5]):  # Show first 5 blocks
            pref_type = block['pref_type']
            preference = block['preference'][:50] + "..." if len(block['preference']) > 50 else block['preference']
            prev_pref = block['prev_pref'][:30] + "..." if block['prev_pref'] and len(block['prev_pref']) > 30 else block['prev_pref']
            print(f"    {i+1}. {pref_type}: {preference}")
            if prev_pref:
                print(f"       prev_pref: {prev_pref}")
        
        # Flatten to single list of messages
        all_messages = []
        for block in ordered_blocks:
            all_messages.extend(block['messages'])
        
        # Compute total tokens
        total_tokens = count_tokens(all_messages)
        print(f"Total tokens: {total_tokens}.")

        print("context length: ", context_len)

        # Optional trimming by token budget
        if context_len is not None and total_tokens > context_len:
            trimmed = []
            tokens = 0
            for msg in all_messages:
                tcount = count_tokens(msg)
                if tokens + tcount > context_len:
                    break
                trimmed.append(msg)
                tokens += tcount
            all_messages = trimmed
            print(f"Trimmed to {len(trimmed)} messages, {tokens} tokens.")

        # Count final tokens using only message content
        final_content_tokens = 0
        for msg in all_messages:
            if isinstance(msg, dict) and 'content' in msg:
                content = msg['content']
                if isinstance(content, str):
                    final_content_tokens += len(ENCODER.encode(content))
                elif isinstance(content, list):
                    # Handle multimodal content - only count text parts
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            final_content_tokens += len(ENCODER.encode(item['text']))
                        elif isinstance(item, str):
                            final_content_tokens += len(ENCODER.encode(item))
        
        print(f"Final token count (content only): {final_content_tokens} tokens from {len(all_messages)} messages.")

        # Extract timestamp and persona info from input filename if provided
        timestamp = None
        persona_number = None
        if input_filename:
            # Extract from pattern: interactions_YYMMDD_HHMMSS_personaXXX.json
            import os
            basename = os.path.basename(input_filename)
            match = re.search(r'interactions_(\d{6})_(\d{6})_persona(\d+)\.json', basename)
            if match:
                date_part = match.group(1)  # YYMMDD
                time_part = match.group(2)  # HHMMSS
                persona_number = int(match.group(3))
                timestamp = f"{date_part}_{time_part}"

        # Create output object with metadata and messages
        output_data = {
            "metadata": {
                "total_messages": len(all_messages),
                "final_token_count": final_content_tokens,
                "context_length_limit": context_len,
                "was_trimmed": context_len is not None and total_tokens > context_len,
                "persona_uuid": uuid,
                "timestamp": timestamp,
                "persona_number": persona_number,
                "input_filename": input_filename
            },
            "messages": all_messages
        }

        # Generate output filename using same naming convention
        if timestamp and persona_number is not None:
            filename = f"data/contexts/context_{timestamp}_persona{persona_number}.json"
        else:
            # Fallback to old naming if we can't extract info
            filename = f"data/contexts/context_{uuid}.json"
        
        # Save raw messages with metadata
        utils.save_json(output_data, filename, clean=True)
        print(f"Saved contexts to {filename}.")