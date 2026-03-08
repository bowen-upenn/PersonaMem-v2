import json
import random
import os
import concurrent.futures
import threading
from tqdm import tqdm

import utils
from qa_generator import generate_qa_for_each_element, generate_qa_for_sensitive_info
from conv_generator import generate_cross_domain_conversations, FILE_SAVE_LOCK
from conv_generator import get_random_sensitive_info
import prompts


def process_existing_files_for_others_preferences(llm, persona_files, parallel=False, verbose=False):
    """
    Process existing JSON files to regenerate conversations for "others" preferences.
    
    For each file:
    - Go to the "conversations" key (which is a dict)
    - Under each key of this dict, it is a list of dicts
    - For each dict: if "who" is "self" and "updated" is False, check if "preference" 
      appears in any "prev_pref" of any other dicts in the whole JSON file
    - If not, with probability 0.3, regenerate conversations and QAs using "who" == "others"
    - Remove existing "who", "conversations", "user_query", "topic_query", "correct_answer", "incorrect_answers"
    
    Args:
        llm: QueryLLM instance
        persona_files: List of JSON file paths to process
        parallel: Whether to use parallel processing
        verbose: Whether to print debug information
    """
    
    # Global statistics tracking
    total_candidate_samples = 0
    total_selected_samples = 0 
    total_successful_replacements = 0
    total_files_modified = 0
    files_processed = 0
    
    def process_single_file(file_path):
        """Process a single JSON file."""
        if verbose:
            print(f"Processing file: {file_path}")
        
        # File-level statistics
        file_candidate_samples = 0
        file_selected_samples = 0
        file_successful_replacements = 0
        
        # try:
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get the first (and typically only) persona in the file
        persona_key = list(data.keys())[0]
        persona_data = data[persona_key]
        conversations = persona_data['conversations']
        
        # Initialize global topics at the file level for topic categorization consistency
        global_topics = []
    
        # Collect all prev_pref values in the entire file
        all_prev_prefs = set()
        for conv_type, conv_list in conversations.items():
            for item in conv_list:
                if 'prev_pref' in item and item['prev_pref']:
                    all_prev_prefs.add(item['prev_pref'])
        
        # Track modifications
        modified = False
        
        # Process each conversation type
        for conv_type, conv_list in conversations.items():
            if conv_type == "knowledge_query":
                continue
            # Process items in reverse order to avoid index issues when modifying
            for i in range(len(conv_list) - 1, -1, -1):
                item = conv_list[i]
                
                # Check if this item meets our criteria
                if (item.get('who') == 'self' or (item.get('who') == 'others' and 'user_query' not in item)) \
                    and 'preference' in item and item["pref_type"] != "ask_to_forget":

                    preference = item['preference']
                    
                    # Check if this preference appears in any prev_pref
                    if preference not in all_prev_prefs:
                        # This is a candidate sample
                        file_candidate_samples += 1
                    
                        threshold = 0.1
                        if random.random() < threshold:
                            # This sample was selected
                            file_selected_samples += 1
                            if verbose:
                                print(f"Regenerating item {i} in {conv_type} for 'others': {preference[:50]}...")
                            
                            # Extract persona info for conversation generation
                            persona_str = json.dumps(persona_data, ensure_ascii=False)
                            pref_key = item.get('pref_type')
                            topic_preference = item.get('topic_preference')
                            
                            # Generate new conversations with is_others_pref=True
                            llm.reset_history()
                            new_element = generate_cross_domain_conversations(
                                llm, persona_str, preference, pref_key, conv_type, 
                                conversations, is_others_pref=True, topic_preference=topic_preference, verbose=verbose
                            )
                            
                            if new_element:
                                # The new element was appended to conversations[conv_type] by generate_cross_domain_conversations
                                # Get the newly appended item (last item in the list)
                                newly_appended_item = conversations[conv_type][-1]
                                
                                # Generate QA pairs for the newly appended element with retry logic
                                qa_success = False
                                qa_attempts = 0
                                max_qa_attempts = 3
                                
                                while qa_attempts < max_qa_attempts and not qa_success:
                                    qa_attempts += 1
                                    try:
                                        llm.reset_history()
                                        qa_element = generate_qa_for_each_element(
                                            llm, newly_appended_item, persona_str, global_topics=global_topics, verbose=verbose
                                        )
                                        if qa_element:
                                            qa_success = True
                                            if verbose:
                                                print(f"Successfully generated QA for 'others' preference on attempt {qa_attempts}")
                                            break  # Exit the while loop immediately on success
                                        else:
                                            if verbose:
                                                print(f"Failed to generate QA for 'others' preference on attempt {qa_attempts}")
                                    except Exception as e:
                                        if verbose:
                                            print(f"Error generating QA on attempt {qa_attempts}: {e}")
                                
                                if not qa_success and verbose:
                                    print(f"Failed to generate QA after {max_qa_attempts} attempts")
                                
                                # Only remove the old item if both conversation and QA generation succeeded
                                if qa_success:
                                    # Remove the old item from conversations (the new one is already in the list)
                                    conv_list.pop(i)
                                    modified = True
                                    file_successful_replacements += 1
                                    if verbose:
                                        print(f"Successfully replaced item for 'others' with both conversations and QA")
                                else:
                                    # Remove the newly appended item since QA generation failed
                                    conversations[conv_type].pop()  # Remove the last appended item
                                    if verbose:
                                        print(f"Skipping item replacement due to QA generation failure")
                            else:
                                if verbose:
                                    print(f"Failed to generate new conversations for item")
                            
                            # except Exception as e:
                            #     if verbose:
                            #         print(f"Error regenerating conversations: {e}")
                            #     continue
        
            # Save the file if it was modified
            if modified:
                with FILE_SAVE_LOCK:
                    utils.save_json(data, file_path, clean=False)
                    if verbose:
                        print(f"Saved modifications to {file_path}")
            
        print(f"File {os.path.basename(file_path)}: {file_candidate_samples} candidates, {file_selected_samples} selected, {file_successful_replacements} successfully replaced")
        
        return file_candidate_samples, file_selected_samples, file_successful_replacements, 1 if modified else 0
    
        # except Exception as e:
        #     print(f"Error processing file {file_path}: {e}")
        #     return 0, 0, 0, 0
    
    # Process files
    if parallel:
        # Parallel processing
        max_workers = min(llm.rate_limit_per_min, len(persona_files))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_file, file_path): file_path for file_path in persona_files}
            
            # Collect results with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             desc="Processing files for others preferences", 
                             total=len(persona_files)):
                try:
                    file_candidates, file_selected, file_successful, file_modified = future.result()
                    total_candidate_samples += file_candidates
                    total_selected_samples += file_selected
                    total_successful_replacements += file_successful
                    total_files_modified += file_modified
                    files_processed += 1
                except Exception as e:
                    file_path = futures[future]
                    print(f"Error in future for file {file_path}: {e}")
                    files_processed += 1
    else:
        # Sequential processing
        for file_path in tqdm(persona_files, desc="Processing files for others preferences"):
            file_candidates, file_selected, file_successful, file_modified = process_single_file(file_path)
            total_candidate_samples += file_candidates
            total_selected_samples += file_selected
            total_successful_replacements += file_successful
            total_files_modified += file_modified
            files_processed += 1
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Processing files for 'others' preferences")
    print(f"{'='*60}")
    print(f"Files processed: {files_processed}/{len(persona_files)}")
    print(f"Files modified: {total_files_modified}")
    print(f"")
    print(f"Sample Statistics:")
    print(f"  Candidate samples found: {total_candidate_samples}")
    print(f"  Samples selected (30% probability): {total_selected_samples}")
    print(f"  Samples successfully replaced: {total_successful_replacements}")
    print(f"")
    if total_selected_samples > 0:
        success_rate = (total_successful_replacements / total_selected_samples * 100)
        print(f"Selection rate: {(total_selected_samples/total_candidate_samples*100):.1f}% of candidates were selected" if total_candidate_samples > 0 else "Selection rate: 0%")
        print(f"Success rate: {success_rate:.1f}% of selected samples were successfully replaced")
    else:
        print(f"Selection rate: 0% (no samples selected)")
        print(f"Success rate: N/A (no samples selected)")
    print(f"{'='*60}")
    
    if total_successful_replacements == 0:
        print("No items were converted to 'others' preferences. This could be due to:")
        if total_candidate_samples == 0:
            print("- All eligible preferences already appear in 'prev_pref' fields")
        elif total_selected_samples == 0:
            print("- Random 30% selection didn't pick any items")
        else:
            print("- Generation failures for all selected items")
    else:
        print(f"Successfully converted {total_successful_replacements} preferences from 'self' to 'others' context")


def process_existing_files_for_sensitive_info(llm, persona_files, parallel=False, verbose=False):
    """
    Process existing JSON files to add sensitive information conversations.
    
    For each file:
    - Go to the "conversations" key (which is a dict)
    - Under each key of this dict, it is a list of dicts
    - For each dict: if it doesn't have "sensitive_info" key, with probability 0.1, 
      randomly pick the item and copy its preference, but add sensitive_information
    - Generate new conversations and QAs for the sensitive information
    
    Args:
        llm: QueryLLM instance
        persona_files: List of JSON file paths to process
        parallel: Whether to use parallel processing
        verbose: Whether to print debug information
    """
    
    # Global statistics tracking
    total_candidate_samples = 0
    total_selected_samples = 0 
    total_successful_replacements = 0
    total_files_modified = 0
    files_processed = 0
    
    def process_single_file(file_path):
        """Process a single JSON file."""
        if verbose:
            print(f"Processing file: {file_path}")
        
        # File-level statistics
        file_candidate_samples = 0
        file_selected_samples = 0
        file_successful_replacements = 0
        
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get the first (and typically only) persona in the file
        persona_key = list(data.keys())[0]
        persona_data = data[persona_key]
        conversations = persona_data['conversations']
        
        # Initialize global topics at the file level for topic categorization consistency
        global_topics = []
        
        # Get sensitive information from persona data
        sensitive_info = persona_data.get('sensitive_information', {})
        if not sensitive_info:
            if verbose:
                print(f"No sensitive information found in {file_path}, skipping")
            return 0, 0, 0, 0
        
        # Track modifications
        modified = False
        
        # Process each conversation type
        for conv_type, conv_list in conversations.items():
            if conv_type == "knowledge_query":
                continue
            
            # Process items in reverse order to avoid index issues when modifying
            for i in range(len(conv_list) - 1, -1, -1):
                item = conv_list[i]
                
                # Check if this item meets our criteria (doesn't have sensitive_info key)
                if 'sensitive_info' not in item and 'preference' in item and item['updated'] == False and item['who'] == 'self':
                    # This is a candidate sample
                    file_candidate_samples += 1
                    
                    # With 10% probability, add sensitive information
                    if random.random() < 0.1:
                        # This sample was selected
                        file_selected_samples += 1
                        if verbose:
                            print(f"Adding sensitive info to item {i} in {conv_type}: {item.get('preference', '')[:50]}...")
                        
                        # Extract persona info for conversation generation
                        persona_str = json.dumps(persona_data, ensure_ascii=False)
                        
                        # Get random sensitive information
                        random_sensitive_info = get_random_sensitive_info(sensitive_info, verbose)
                        
                        # Generate new conversations with sensitive information
                        llm.reset_history()
                        prompt = prompts.generate_conversations_sensitive_info(persona_str, random_sensitive_info, conv_type)
                        conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)
                        
                        conv_turns = utils.extract_json_from_response(conv_turns)
                        conv_turns = utils.merge_consecutive_roles(conv_turns)
                        
                        if conv_turns:
                            # Create new element with sensitive information
                            new_element = {
                                'sensitive_info': random_sensitive_info, 
                                'who': 'self', 
                                'conversations': conv_turns
                            }
                            
                            # Generate QA pairs for the newly created element with retry logic
                            qa_success = False
                            qa_attempts = 0
                            max_qa_attempts = 3
                            
                            while qa_attempts < max_qa_attempts and not qa_success:
                                qa_attempts += 1
                                try:
                                    llm.reset_history()
                                    qa_element = generate_qa_for_sensitive_info(
                                        llm, new_element.copy(), persona_str, global_topics=global_topics, verbose=verbose
                                    )
                                    if qa_element:
                                        # Update the new element with QA fields
                                        new_element.update({
                                            'user_query': qa_element.get('user_query'),
                                            'topic_query': qa_element.get('topic_query'),
                                            'correct_answer': qa_element.get('correct_answer'),
                                            'incorrect_answers': qa_element.get('incorrect_answers')
                                        })
                                        qa_success = True
                                        if verbose:
                                            print(f"Successfully generated QA for sensitive info on attempt {qa_attempts}")
                                        break  # Exit the while loop immediately on success
                                    else:
                                        if verbose:
                                            print(f"Failed to generate QA for sensitive info on attempt {qa_attempts}")
                                except Exception as e:
                                    if verbose:
                                        print(f"Error generating QA on attempt {qa_attempts}: {e}")
                            
                            if qa_success:
                                # Add the new element to conversations
                                conversations[conv_type].append(new_element)
                                modified = True
                                file_successful_replacements += 1
                                if verbose:
                                    print(f"Successfully added sensitive info conversation with QA")
                            else:
                                if verbose:
                                    print(f"Failed to generate QA after {max_qa_attempts} attempts, skipping")
                        else:
                            if verbose:
                                print(f"Failed to generate conversations for sensitive info")

        # Save the file if it was modified
        if modified:
            with FILE_SAVE_LOCK:
                utils.save_json(data, file_path, clean=False)
                if verbose:
                    print(f"Saved modifications to {file_path}")
        
        print(f"File {os.path.basename(file_path)}: {file_candidate_samples} candidates, {file_selected_samples} selected, {file_successful_replacements} successfully added")
        
        return file_candidate_samples, file_selected_samples, file_successful_replacements, 1 if modified else 0
    
    # Process files
    if parallel:
        # Parallel processing
        max_workers = min(llm.rate_limit_per_min, len(persona_files))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_file, file_path): file_path for file_path in persona_files}
            
            # Collect results with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             desc="Processing files for sensitive info", 
                             total=len(persona_files)):
                try:
                    file_candidates, file_selected, file_successful, file_modified = future.result()
                    total_candidate_samples += file_candidates
                    total_selected_samples += file_selected
                    total_successful_replacements += file_successful
                    total_files_modified += file_modified
                    files_processed += 1
                except Exception as e:
                    file_path = futures[future]
                    print(f"Error in future for file {file_path}: {e}")
                    files_processed += 1
    else:
        # Sequential processing
        for file_path in tqdm(persona_files, desc="Processing files for sensitive info"):
            file_candidates, file_selected, file_successful, file_modified = process_single_file(file_path)
            total_candidate_samples += file_candidates
            total_selected_samples += file_selected
            total_successful_replacements += file_successful
            total_files_modified += file_modified
            files_processed += 1
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Processing files for sensitive information")
    print(f"{'='*60}")
    print(f"Files processed: {files_processed}/{len(persona_files)}")
    print(f"Files modified: {total_files_modified}")
    print(f"")
    print(f"Sample Statistics:")
    print(f"  Candidate samples found: {total_candidate_samples}")
    print(f"  Samples selected (10% probability): {total_selected_samples}")
    print(f"  Samples successfully added: {total_successful_replacements}")
    print(f"")
    if total_selected_samples > 0:
        success_rate = (total_successful_replacements / total_selected_samples * 100)
        print(f"Selection rate: {(total_selected_samples/total_candidate_samples*100):.1f}% of candidates were selected" if total_candidate_samples > 0 else "Selection rate: 0%")
        print(f"Success rate: {success_rate:.1f}% of selected samples were successfully added")
    else:
        print(f"Selection rate: 0% (no samples selected)")
        print(f"Success rate: N/A (no samples selected)")
    print(f"{'='*60}")
    
    if total_successful_replacements == 0:
        print("No sensitive information conversations were added. This could be due to:")
        if total_candidate_samples == 0:
            print("- No eligible preferences found (all already have sensitive_info)")
        elif total_selected_samples == 0:
            print("- Random 10% selection didn't pick any items")
        else:
            print("- Generation failures for all selected items")
    else:
        print(f"Successfully added {total_successful_replacements} sensitive information conversations")
