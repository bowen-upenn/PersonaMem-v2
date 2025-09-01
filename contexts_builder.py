import random
import tiktoken
import json
from uuid import uuid4
import re
from tqdm import tqdm
import base64
import os

import utils


ENCODER = tiktoken.encoding_for_model("gpt-4o")


def load_image_as_base64(image_path):
    """Load an image file and convert it to base64 encoding."""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            return base64_encoded
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None


def process_multimodal_content(content, base_dir="data"):
    """Process content that may contain image URLs and replace them with base64 encoded images."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        processed_content = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'image_url':
                image_url = item.get('image_url', {}).get('url', '')
                if image_url.startswith('data:image/jpeg;base64,'):
                    # Extract the path after the base64 prefix
                    path_part = image_url.replace('data:image/jpeg;base64,', '')
                    if not path_part.startswith('data:') and not path_part.startswith('/'):
                        # Load and encode the image
                        base64_image = load_image_as_base64(path_part)
                        if base64_image:
                            # Create new item with actual base64 data
                            processed_item = item.copy()
                            processed_item['image_url']['url'] = f'data:image/jpeg;base64,{base64_image}'
                            processed_content.append(processed_item)
                        else:
                            # Keep original if loading fails
                            processed_content.append(item)
                    else:
                        # Already has actual base64 data or absolute path
                        processed_content.append(item)
                else:
                    # Not our expected format
                    processed_content.append(item)
            else:
                processed_content.append(item)
        return processed_content
    else:
        return content


def create_text_only_content(content):
    """Extract only text content from multimodal content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
            elif isinstance(item, str):
                text_parts.append(item)
        return ' '.join(text_parts) if text_parts else ''
    else:
        return str(content)


def save_chat_history_versions(output_data, filename, persona_number, shared_timestamp):
    """Save both text-only and multimodal versions of chat history."""
    
    # Create text-only version
    text_only_data = output_data.copy()
    text_only_messages = []
    
    for message in output_data['chat_history']:
        text_only_message = message.copy()
        if 'content' in text_only_message:
            text_only_message['content'] = create_text_only_content(message['content'])
        text_only_messages.append(text_only_message)
    
    text_only_data['chat_history'] = text_only_messages
    
    # Update token count for text-only version
    text_only_token_count = count_tokens(text_only_messages, include_images=False)
    text_only_data['metadata']['final_token_count'] = text_only_token_count
    
    # Save text-only version (original location)
    utils.save_json(text_only_data, filename, clean=True)
    
    # Create multimodal version with base64 encoded images
    multimodal_data = output_data.copy()
    multimodal_messages = []
    
    for message in output_data['chat_history']:
        multimodal_message = message.copy()
        if 'content' in multimodal_message:
            multimodal_message['content'] = process_multimodal_content(message['content'])
        multimodal_messages.append(multimodal_message)
    
    multimodal_data['chat_history'] = multimodal_messages
    
    # Update token count for multimodal version (including image tokens)
    multimodal_token_count = count_tokens(multimodal_messages, include_images=True)
    multimodal_data['metadata']['final_token_count'] = multimodal_token_count
    
    # Save multimodal version
    if persona_number is not None and shared_timestamp:
        multimodal_filename = f"data/chat_history_multimodal/chat_history_{shared_timestamp}_persona{persona_number}.json"
    else:
        # Fallback naming
        base_name = os.path.splitext(os.path.basename(filename))[0]
        multimodal_filename = f"data/chat_history_multimodal/{base_name}.json"
    
    # Ensure the multimodal directory exists
    os.makedirs("data/chat_history_multimodal", exist_ok=True)
    
    utils.save_json(multimodal_data, multimodal_filename, clean=True)


def count_tokens(text, include_images=False) -> int:
    """Count tokens in text, optionally including image tokens.
    
    Args:
        text: The text content to count tokens for
        include_images: If True, includes estimated tokens for images (85 tokens per image for gpt-4o)
    """
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
                        elif isinstance(item, dict) and 'type' in item and item['type'] == 'image_url' and include_images:
                            # Add estimated tokens for images (85 tokens for gpt-4o)
                            total_tokens += 85
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
                    elif isinstance(item, dict) and 'type' in item and item['type'] == 'image_url' and include_images:
                        # Add estimated tokens for images (85 tokens for gpt-4o)
                        total_tokens += 85
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
                        'conv_type': conv_type,
                        'topic_preference': block.get('topic_preference', ''),
                    }
                    if 'topic_query' in block:
                        block_info['topic_query'] = block.get('topic_query', '')
                        block_info['user_query'] = block.get('user_query', '')
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
    
    # Step 2: Collect all updated blocks to add them much later in the sequence
    all_updated_blocks = []
    for original_pref, updated_blocks in prev_pref_to_blocks.items():
        # Verify that the original preference exists in the ordered list
        original_exists = any(block['preference'] == original_pref for block in ordered_blocks)
        
        if original_exists:
            # Add updated blocks to be inserted later
            all_updated_blocks.extend(updated_blocks)
        else:
            # If original preference not found, add at the end as fallback
            ordered_blocks.extend(updated_blocks)
    
    # Step 3: Add updated blocks much later in the sequence (before "Do not remember" blocks)
    # Insert updated blocks towards the end, but leave some space from the very end
    if all_updated_blocks:
        # Shuffle updated blocks to avoid any ordering bias
        random.shuffle(all_updated_blocks)
        # Add them to the end of the current sequence
        ordered_blocks.extend(all_updated_blocks)
    
    # Step 4: Add "Do not remember" blocks at the end of their sequences
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


def _build_context_for_single_file(interactions, context_len=None, input_filename=None, verbose=False, shared_timestamp=None):
    """
    Internal function to build context for a single file's interactions.
    """
    for uuid, persona in interactions.items():
        # Extract all conversation blocks
        all_blocks = extract_conversation_blocks({uuid: persona})
        
        if not all_blocks:
            if verbose:
                print(f"No conversation blocks found for {uuid}")
            continue
        
        if verbose:
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
        
        if verbose:
            print(f"  Ordered {len(ordered_blocks)} blocks")
        
        # Extract persona information (everything before "stereotypical_preferences") to create system prompt
        persona_info = {}
        for key, value in persona.items():
            if key == "stereotypical_preferences":
                break
            persona_info[key] = value
        
        # Create system prompt from persona information
        import json
        persona_json_str = json.dumps(persona_info, indent=2, ensure_ascii=False)
        system_prompt = f"You are an AI assistant helping a user with the following persona:\n\n{persona_json_str}\n\nPlease respond in a way that's appropriate for this user's background, interests, and communication style."
        
        # Create system message
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        
        # Flatten to single list of messages
        all_messages = [system_message]  # Start with system message
        for block in ordered_blocks:
            all_messages.extend(block['messages'])
        
        # Compute total tokens
        total_tokens = count_tokens(all_messages, include_images=False)
        if verbose:
            print(f"Total tokens: {total_tokens}.")
            print("context length: ", context_len)

        # Intelligent trimming by token budget
        if context_len is not None and total_tokens > context_len:
            # Step 1: Identify protected conversation blocks (those with user_query or linked to updated preferences with user_query)
            protected_blocks = set()
            
            # Find blocks with user_query (evaluation targets)
            for i, block in enumerate(ordered_blocks):
                if 'user_query' in block:
                    protected_blocks.add(i)
            
            # Find blocks whose preference appears as prev_pref in updated blocks with user_query
            for i, block in enumerate(ordered_blocks):
                if 'user_query' in block and block.get('prev_pref'):
                    # Find the original block with this preference
                    for j, orig_block in enumerate(ordered_blocks):
                        if orig_block['preference'] == block['prev_pref']:
                            protected_blocks.add(j)
                            protected_blocks.add(i)  # Also protect the updated block
            
            # Step 2: Create mapping of message indices to blocks
            message_to_block = {}
            protected_message_indices = set([0])  # Always protect system message
            message_idx = 1  # Start after system message
            
            for block_idx, block in enumerate(ordered_blocks):
                block_start = message_idx
                block_end = message_idx + len(block['messages'])
                
                for msg_idx in range(block_start, block_end):
                    message_to_block[msg_idx] = block_idx
                    if block_idx in protected_blocks:
                        protected_message_indices.add(msg_idx)
                
                message_idx = block_end
            
            if verbose:
                print(f"Protected {len(protected_blocks)} blocks with {len(protected_message_indices)} messages")
            
            # Step 3: Build final message list based on token budget
            final_messages = []
            current_tokens = 0
            
            # First pass: add all protected messages in order
            for i in range(len(all_messages)):
                if i in protected_message_indices:
                    msg_tokens = count_tokens(all_messages[i], include_images=False)
                    final_messages.append(all_messages[i])
                    current_tokens += msg_tokens
            
            if verbose:
                print(f"Protected messages: {len(final_messages)} messages, {current_tokens} tokens")
            
            # Second pass: add non-protected messages if budget allows
            if current_tokens <= context_len:
                remaining_budget = context_len - current_tokens
                additional_messages = []
                
                for i in range(len(all_messages)):
                    if i not in protected_message_indices:
                        msg_tokens = count_tokens(all_messages[i], include_images=False)
                        if msg_tokens <= remaining_budget:
                            additional_messages.append((i, all_messages[i]))
                            remaining_budget -= msg_tokens
                        else:
                            break
                
                # Insert additional messages in their original order
                if additional_messages:
                    # Merge protected and additional messages while preserving order
                    all_indexed_messages = [(i, msg) for i, msg in enumerate(all_messages) if i in protected_message_indices]
                    all_indexed_messages.extend(additional_messages)
                    all_indexed_messages.sort(key=lambda x: x[0])  # Sort by original index
                    
                    final_messages = [msg for _, msg in all_indexed_messages]
                    current_tokens = sum(count_tokens(msg, include_images=False) for msg in final_messages)
                
                if verbose:
                    print(f"Intelligent trimming: kept {len(final_messages)} messages, {current_tokens} tokens")
            else:
                if verbose:
                    print(f"Warning: Protected content ({current_tokens} tokens) exceeds budget ({context_len} tokens)")
                    print(f"Using only protected content: {len(final_messages)} messages")
            
            all_messages = final_messages

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
        
        if verbose:
            print(f"Final token count (content only): {final_content_tokens} tokens from {len(all_messages)} messages.")

        # Extract timestamp and persona info from input filename if provided
        timestamp = None
        persona_number = None
        if input_filename:
            # Extract from pattern: raw_data_YYMMDD_HHMMSS_personaXXX.json (new format)
            # or interactions_YYMMDD_HHMMSS_personaXXX.json (legacy format)
            import os
            basename = os.path.basename(input_filename)
            
            # Try new format first: raw_data_YYMMDD_HHMMSS_personaXXX.json
            match = re.search(r'raw_data_(\d{6})_(\d{6})_persona(\d+)\.json', basename)
            if not match:
                # Try legacy format: interactions_YYMMDD_HHMMSS_personaXXX.json
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
                "persona_id": persona_number,
                "input_filename": input_filename
            },
            "chat_history": all_messages
        }

        # Generate output filename using shared timestamp and persona format like conv_generator
        if persona_number is not None and shared_timestamp:
            filename = f"data/chat_history/chat_history_{shared_timestamp}_persona{persona_number}.json"
        elif persona_number is not None:
            # Fallback: generate timestamp if shared_timestamp not provided
            import pytz
            from datetime import datetime
            
            pacific_tz = pytz.timezone('US/Pacific')
            now = datetime.now(pacific_tz)
            pacific_timestamp = now.strftime('%y%m%d_%H%M%S')
            
            filename = f"data/chat_history/chat_history_{pacific_timestamp}_persona{persona_number}.json"
        else:
            # Fallback to old naming if we can't extract persona info
            filename = f"data/chat_history/chat_history_{uuid}.json"
        
        # Ensure the chat_history directory exists
        import os
        os.makedirs("data/chat_history", exist_ok=True)
        
        # Save both text-only and multimodal versions
        save_chat_history_versions(output_data, filename, persona_number, shared_timestamp)
        print(f"Saved chat history to {filename} and multimodal version.")


def build_context(conv_output_dir=None, interactions=None, context_len=None, input_filename=None, persona_start_idx=-1, persona_end_idx=-1, verbose=False):
    """
    Constructs a multi-turn conversation list for each persona by extracting all conversations,
    ordering them based on dependencies, and optionally trimming by token budget.
    Saves flattened messages to JSON and returns a list of dictionaries:
        [ {role, content}, ... ]
    
    Args:
        conv_output_dir: Directory containing persona files (if processing multiple files)
        interactions: Dictionary of persona data (if processing single file data)
        context_len: Optional token limit for trimming conversations
        input_filename: Optional filename for metadata extraction (used when interactions is provided)
        persona_start_idx: Starting persona index (-1 for beginning)
        persona_end_idx: Ending persona index (-1 for end)
        verbose: Whether to print detailed processing information
    """
    import json
    import os
    import pytz
    from datetime import datetime
    
    # Generate shared timestamp for all files processed in this batch
    pacific_tz = pytz.timezone('US/Pacific')
    now = datetime.now(pacific_tz)
    shared_timestamp = now.strftime('%y%m%d_%H%M%S')
    
    # If conv_output_dir is provided, process multiple files
    if conv_output_dir:
        # Get persona files within the specified range
        persona_files = utils.get_persona_files_in_range(
            conv_output_dir,
            'raw_data',
            persona_start_idx,
            persona_end_idx
        )
        
        if not persona_files:
            print("No persona files found in the specified range.")
            return
        
        if verbose:
            print(f"Found {len(persona_files)} persona files to build context from")

            
        
        # Process each persona file individually to preserve filename information
        for file_path in tqdm(persona_files):
            if verbose:
                print(f"\nProcessing {file_path}")
            try:
                with open(file_path, 'r') as file:
                    file_data = json.load(file)
                
                # Process this individual persona file
                _build_context_for_single_file(file_data, context_len, file_path, verbose, shared_timestamp)
                
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON in {file_path}")
                if verbose:
                    print(f"JSON Error: {e}")
                print(f"Skipping this file and continuing with next...")
                continue
            except Exception as e:
                print(f"ERROR: Failed to process {file_path}")
                if verbose:
                    print(f"Error: {e}")
                print(f"Skipping this file and continuing with next...")
                continue
        return
    
    # If interactions is provided, process single file data
    elif interactions:
        _build_context_for_single_file(interactions, context_len, input_filename, verbose, shared_timestamp)
    else:
        raise ValueError("Either conv_output_dir or interactions must be provided")
