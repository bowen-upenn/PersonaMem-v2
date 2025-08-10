import json
from pathlib import Path
import argparse
from tqdm import tqdm
import concurrent.futures
import math
import os
import random
import re

import prompts
import utils
from query_llm import QueryLLM


def save_intermediate_topics(global_topics, batch_number, output_dir="data/intermediate_topics"):
    """
    Save intermediate global topics to a timestamped file.
    
    Args:
        global_topics: Set of global topics
        batch_number: Current batch number
        output_dir: Directory to save intermediate topics
    """
    os.makedirs(output_dir, exist_ok=True)
    topics_file = os.path.join(output_dir, f"global_topics_batch_{batch_number:04d}.json")
    
    topics_data = {
        "batch_number": batch_number,
        "total_topics": len(global_topics),
        "topics": sorted(list(global_topics)),
        "timestamp": utils.get_timestamp() if hasattr(utils, 'get_timestamp') else str(batch_number)
    }
    
    utils.save_json(topics_data, topics_file)
    return topics_file


def load_intermediate_topics(topics_file):
    """
    Load intermediate global topics from a file.
    
    Args:
        topics_file: Path to the topics file
        
    Returns:
        set: Set of topics
    """
    try:
        with open(topics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return set(data.get("topics", []))
    except Exception as e:
        print(f"Warning: Could not load topics from {topics_file}: {e}")
        return set()


def merge_all_intermediate_topics(output_dir="data/intermediate_topics", verbose=False):
    """
    Merge all intermediate topic files into a single global topics set.
    
    Args:
        output_dir: Directory containing intermediate topics files
        verbose: Whether to print verbose output
        
    Returns:
        set: Merged set of all topics
    """
    if not os.path.exists(output_dir):
        return set()
    
    all_topics = set()
    topic_files = sorted([f for f in os.listdir(output_dir) if f.startswith("global_topics_batch_") and f.endswith(".json")])
    
    if verbose:
        print(f"Merging {len(topic_files)} intermediate topic files...")
    
    for topic_file in topic_files:
        file_path = os.path.join(output_dir, topic_file)
        topics = load_intermediate_topics(file_path)
        all_topics.update(topics)
        if verbose:
            print(f"  Loaded {len(topics)} topics from {topic_file}")
    
    print(f"Merged global topics: {len(all_topics)} unique topics from {len(topic_files)} batches")
    return all_topics


def clean_topic_value(value):
    """Clean and normalize a topic string.
    - If '###Output' exists, keep only the part after it (using utils.extract_after_token if possible)
    - Remove newlines and trim whitespace
    """
    if value is None:
        return value
    if not isinstance(value, str):
        value = str(value)
    try:
        # Prefer using shared util when available
        cleaned = utils.extract_after_token(value, '###Output').strip()
        # If extract produced empty because token not present, fall back to original
        if not cleaned:
            cleaned = value
    except Exception:
        cleaned = value
    # Remove newlines and excessive spaces
    cleaned = cleaned.replace('\n', ' ').strip()
    return cleaned


def categorize_single_preference(llm, preference, global_topics, verbose=False):
    """
    Categorize a single preference into topics using the LLM.
    
    Args:
        llm: QueryLLM instance
        preference: The preference string to categorize
        global_topics: List of existing global topics
        verbose: Whether to print debug information
        
    Returns:
        str: The topic name
    """
    # Generate categorization prompt
    prompt = prompts.categorize_preference_topic(preference, global_topics)
    
    try:
        response = llm.query_llm(prompt, use_history=False, verbose=verbose)
        topic = utils.extract_after_token(response, '###Output').strip()
        # Ensure topic is cleaned and normalized
        topic = clean_topic_value(topic)
        
        if not topic:
            # Fallback to default category
            topic = "Uncategorized"
        
        return topic
        
    except Exception as e:
        if verbose:
            print(f"Error categorizing preference '{preference}': {e}")
        # Return a default category on error
        return "General"


def categorize_topics_for_single_file(args):
    """Categorize topics for a single file in parallel processing."""
    file_path, llm, global_topics, verbose = args
    
    # try:
    return file_path, categorize_topics_sequential(file_path, llm, global_topics, verbose)
    # except Exception as e:
    #     print(f"Error processing file {file_path}: {e}")
    #     return file_path, {}


def categorize_topics_sequential(file_path, llm, global_topics, verbose=False):
    """Categorize topics for a single file sequentially."""
    # Load the persona file
    print(f"Loading persona file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Keep track of categorization statistics
    total_elements = 0
    new_topics_added = 0
    file_topics = set()
    
    # Process each persona in the file
    for uuid, persona in data.items():
        conversations_by_type = persona.get("conversations", {})
        
        for conv_type, conv_list in conversations_by_type.items():
            if verbose:
                print(f'Processing conv_type: {conv_type} in {os.path.basename(file_path)}')
            
            for conv_elem in tqdm(conv_list, desc=f"Categorizing {conv_type} for {os.path.basename(file_path)}", leave=False):
                try:
                    # Extract preference and user_query from the element
                    preference = conv_elem.get('preference')
                    user_query = None #conv_elem.get('user_query')
                    
                    # Remove legacy "topic" field if it exists
                    if 'topic' in conv_elem:
                        del conv_elem['topic']
                    
                    # Skip elements that have neither preference nor user_query
                    if not preference and not user_query:
                        continue
                    
                    total_elements += 1
                    
                    # Categorize the preference if it exists
                    if preference:
                        topic_preference = categorize_single_preference(llm, preference, list(global_topics), verbose=verbose)
                        # Clean again defensively before saving
                        topic_preference = clean_topic_value(topic_preference)
                        conv_elem["topic_preference"] = topic_preference
                        
                        # Update global topics if this is a new topic
                        if topic_preference not in global_topics:
                            global_topics.add(topic_preference)
                            new_topics_added += 1
                            if verbose:
                                print(f"✓ New preference topic added: '{topic_preference}'")
                        else:
                            if verbose:
                                print(f"✓ Preference assigned to existing topic: '{topic_preference}'")
                        
                        # Track topics in this file
                        file_topics.add(topic_preference)
                    
                    # Categorize the user_query if it exists
                    if user_query:
                        topic_query = categorize_single_preference(llm, user_query, list(global_topics), verbose=verbose)
                        topic_query = clean_topic_value(topic_query)
                        conv_elem["topic_query"] = topic_query
                        
                        # Update global topics if this is a new topic
                        if topic_query not in global_topics:
                            global_topics.add(topic_query)
                            new_topics_added += 1
                            if verbose:
                                print(f"✓ New query topic added: '{topic_query}'")
                        else:
                            if verbose:
                                print(f"✓ Query assigned to existing topic: '{topic_query}'")
                        
                        # Track topics in this file
                        file_topics.add(topic_query)
                
                except Exception as e:
                    pref_info = conv_elem.get('preference', 'Unknown preference')
                    query_info = conv_elem.get('user_query', 'Unknown query')
                    print(f"Error categorizing preference '{pref_info}' or query '{query_info}': {e}")
                    continue
    
    # Print categorization statistics
    print(f"Topic Categorization Summary for {os.path.basename(file_path)}:")
    print(f"  Total elements categorized: {total_elements}")
    print(f"  New topics created: {new_topics_added}")
    print(f"  Unique topics in this file: {len(file_topics)}")
    if verbose and file_topics:
        print(f"  File topics: {', '.join(sorted(file_topics))}")

    # Clean topics before saving to ensure consistency
    for uuid, persona in data.items():
        conversations_by_type = persona.get("conversations", {})
        for conv_type, conv_list in conversations_by_type.items():
            for conv_elem in conv_list:
                if 'topic_preference' in conv_elem:
                    conv_elem['topic_preference'] = clean_topic_value(conv_elem['topic_preference'])
                if 'topic_query' in conv_elem:
                    conv_elem['topic_query'] = clean_topic_value(conv_elem['topic_query'])

    # Save the updated data back to the same file
    utils.save_json(data, file_path, clean=True)  # Use clean=True to overwrite
    
    return global_topics


def count_topics_across_files(input_path):
    """
    Count the occurrences of each topic across all files.
    
    Args:
        input_path: List of persona files to analyze
        
    Returns:
        tuple: (topic_counts dict, topic_examples dict)
    """
    topic_counts = {}
    topic_examples = {}  # Store example preferences for each topic
    
    for file_path in input_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for uuid, persona in data.items():
                conversations_by_type = persona.get("conversations", {})
                
                for conv_type, conv_list in conversations_by_type.items():
                    for conv_elem in conv_list:
                        # Count topic_preference if it exists
                        topic_pref = conv_elem.get('topic_preference')
                        if topic_pref:
                            topic_counts[topic_pref] = topic_counts.get(topic_pref, 0) + 1
                            if topic_pref not in topic_examples:
                                topic_examples[topic_pref] = conv_elem.get('preference', 'No preference found')
                        
                        # Count topic_query if it exists
                        topic_query = conv_elem.get('topic_query')
                        if topic_query:
                            topic_counts[topic_query] = topic_counts.get(topic_query, 0) + 1
                            if topic_query not in topic_examples:
                                topic_examples[topic_query] = conv_elem.get('user_query', 'No query found')
        
        except Exception as e:
            print(f"Error reading file {file_path} for topic counting: {e}")
            continue
    
    return topic_counts, topic_examples


def recategorize_least_frequent_topics(llm, topic_counts, topic_examples, verbose=False):
    """
    Re-categorize the 10% least frequent topics by asking LLM if they should be merged
    with more frequent categories.
    
    Args:
        llm: QueryLLM instance
        topic_counts: Dictionary of {topic: count}
        topic_examples: Dictionary of {topic: example_preference}
        verbose: Whether to print verbose output
        
    Returns:
        dict: Mapping of {old_topic: new_topic} for topics that should be changed
    """
    if not topic_counts:
        print("No topics found for re-categorization")
        return {}
    
    # Find the 10% least frequent topics, and ensure they have no more than 100 occurrences
    total_topics = len(topic_counts)
    least_frequent_count = max(1, int(total_topics * 0.1))
    
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1])
    least_frequent_topics = sorted_topics[:least_frequent_count]

    # Filter out topics with more than 100 occurrences
    least_frequent_topics = [(topic, count) for topic, count in least_frequent_topics if count <= 100]
    
    print(f"\nRe-categorizing {len(least_frequent_topics)} least frequent topics (10% of {total_topics} total topics)")
    
    # Prepare top topics string for the prompt
    sorted_topics_desc = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    top_topics = sorted_topics_desc
    top_topics_str = "\n".join([f"  - {topic}: {count} occurrences" for topic, count in top_topics])
    
    topic_changes = {}
    
    for topic, count in tqdm(least_frequent_topics, desc="Re-categorizing least frequent topics"):
        try:
            example_preference = topic_examples.get(topic, "No example available")
            
            if verbose:
                print(f"Re-evaluating topic: '{topic}' ({count} occurrences)")
                print(f"Example preference: '{example_preference}'")
            
            # Generate prompt for re-categorization
            prompt = prompts.recategorize_least_frequent_topic(
                topic, count, top_topics_str, example_preference
            )
            
            # Query LLM for re-categorization decision
            response = llm.query_llm(prompt, use_history=False, verbose=verbose)
            decision = utils.extract_after_token(response, '###Output').strip()
            
            if decision.startswith("MERGE:"):
                # Extract the target topic name
                new_topic = decision[6:].strip()
                topic_changes[topic] = new_topic
                if verbose:
                    print(f"✓ Decision: Merge '{topic}' -> '{new_topic}'")
            elif decision.startswith("KEEP:"):
                if verbose:
                    print(f"✓ Decision: Keep '{topic}' as separate category")
            else:
                print(f"⚠ Unclear decision for '{topic}': {decision}")
        
        except Exception as e:
            print(f"Error re-categorizing topic '{topic}': {e}")
            continue
    
    print(f"\nRe-categorization Summary:")
    print(f"  Topics reviewed: {len(least_frequent_topics)}")
    print(f"  Topics to be merged: {len(topic_changes)}")
    
    if topic_changes and verbose:
        print("  Merging decisions:")
        for old_topic, new_topic in topic_changes.items():
            print(f"    '{old_topic}' -> '{new_topic}'")
    
    return topic_changes


def apply_topic_changes_single_file(args):
    """Apply topic changes for a single file in parallel processing."""
    file_path, topic_changes, verbose = args
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_changes = 0
        
        for uuid, persona in data.items():
            conversations_by_type = persona.get("conversations", {})
            
            for conv_type, conv_list in conversations_by_type.items():
                for conv_elem in conv_list:
                    # Update topic_preference if it needs changing
                    topic_pref = conv_elem.get('topic_preference')
                    if topic_pref in topic_changes:
                        conv_elem['topic_preference'] = topic_changes[topic_pref]
                        file_changes += 1
                    
                    # Update topic_query if it needs changing
                    topic_query = conv_elem.get('topic_query')
                    if topic_query in topic_changes:
                        conv_elem['topic_query'] = topic_changes[topic_query]
                        file_changes += 1
        
        # Clean topics before saving
        for uuid, persona in data.items():
            conversations_by_type = persona.get("conversations", {})
            for conv_type, conv_list in conversations_by_type.items():
                for conv_elem in conv_list:
                    if 'topic_preference' in conv_elem:
                        conv_elem['topic_preference'] = clean_topic_value(conv_elem['topic_preference'])
                    if 'topic_query' in conv_elem:
                        conv_elem['topic_query'] = clean_topic_value(conv_elem['topic_query'])
        
        # Save the updated data if changes were made
        if file_changes > 0:
            utils.save_json(data, file_path, clean=True)
            if verbose:
                print(f"✓ Updated {file_changes} topic assignments in {os.path.basename(file_path)}")
        
        return file_path, file_changes
        
    except Exception as e:
        print(f"Error applying topic changes to {file_path}: {e}")
        return file_path, 0


def apply_topic_changes(input_path, topic_changes, verbose=False, parallel=False):
    """
    Apply the topic changes to all files by updating the topic fields.
    
    Args:
        input_path: List of persona files to update
        topic_changes: Dictionary of {old_topic: new_topic}
        verbose: Whether to print verbose output
        parallel: Whether to process files in parallel
    """
    if not topic_changes:
        print("No topic changes to apply")
        return
    
    print(f"Applying topic changes to {len(input_path)} files...")
    total_changes = 0
    
    if parallel:
        # Parallel processing - process multiple files at once
        file_args = [(file_path, topic_changes, verbose) for file_path in input_path]
        
        # Use a reasonable number of workers (max 10 or number of files, whichever is smaller)
        max_workers = min(10, len(file_args))
        batch_size = max_workers
        num_batches = math.ceil(len(file_args) / batch_size)
        
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, len(file_args))
            batch_args = file_args[batch_start_idx:batch_end_idx]
            
            batch_file_paths = [args[0] for args in batch_args]
            print(f"Processing topic changes batch {batch_idx + 1}/{num_batches} ({len(batch_args)} files)")
            if verbose:
                print(f"  Batch files: {[os.path.basename(fp) for fp in batch_file_paths[:3]]}{'...' if len(batch_file_paths) > 3 else ''}")
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                # Submit all tasks for this batch
                future_to_args = {executor.submit(apply_topic_changes_single_file, args): args for args in batch_args}
                
                # Collect results with progress bar
                batch_results = []
                for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                                 desc=f"Applying Changes Batch {batch_idx + 1}", 
                                 total=len(batch_args),
                                 disable=not verbose):
                    try:
                        file_path, file_changes = future.result()
                        batch_results.append((file_path, file_changes))
                        total_changes += file_changes
                        
                    except Exception as e:
                        args = future_to_args[future]
                        file_path = args[0]
                        batch_results.append((file_path, 0))
                        print(f"Error in topic changes future for {os.path.basename(file_path)}: {e}")
                
                # Summary for this batch
                successful_files = [os.path.basename(fp) for fp, changes in batch_results if changes > 0]
                batch_total_changes = sum(changes for _, changes in batch_results)
                print(f"Batch {batch_idx + 1} completed: {len(successful_files)}/{len(batch_results)} files updated, {batch_total_changes} total changes")
    
    else:
        # Sequential processing - process one file at a time
        for file_path in tqdm(input_path, desc="Applying topic changes"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_changes = 0
                
                for uuid, persona in data.items():
                    conversations_by_type = persona.get("conversations", {})
                    
                    for conv_type, conv_list in conversations_by_type.items():
                        for conv_elem in conv_list:
                            # Update topic_preference if it needs changing
                            topic_pref = conv_elem.get('topic_preference')
                            if topic_pref in topic_changes:
                                conv_elem['topic_preference'] = topic_changes[topic_pref]
                                file_changes += 1
                            
                            # Update topic_query if it needs changing
                            topic_query = conv_elem.get('topic_query')
                            if topic_query in topic_changes:
                                conv_elem['topic_query'] = topic_changes[topic_query]
                                file_changes += 1
                
                # Clean topics before saving
                for uuid, persona in data.items():
                    conversations_by_type = persona.get("conversations", {})
                    for conv_type, conv_list in conversations_by_type.items():
                        for conv_elem in conv_list:
                            if 'topic_preference' in conv_elem:
                                conv_elem['topic_preference'] = clean_topic_value(conv_elem['topic_preference'])
                            if 'topic_query' in conv_elem:
                                conv_elem['topic_query'] = clean_topic_value(conv_elem['topic_query'])
                
                # Save the updated data if changes were made
                if file_changes > 0:
                    utils.save_json(data, file_path, clean=True)
                    total_changes += file_changes
                    if verbose:
                        print(f"✓ Updated {file_changes} topic assignments in {os.path.basename(file_path)}")
            
            except Exception as e:
                print(f"Error applying topic changes to {file_path}: {e}")
                continue
    
    print(f"Topic changes applied: {total_changes} total assignments updated")


def categorize_topics(llm, input_path, output_dir=None, parallel=False, verbose=False, refresh_mem=None):
    """
    Main function to categorize topics from preferences in persona files.
    
    Args:
        llm: QueryLLM instance
        input_path: List of persona files to process
        output_dir: Output directory (not used, maintaining compatibility with qa_generator)
        parallel: Whether to process files in parallel
        verbose: Whether to print verbose output
        refresh_mem: Number of files to process before refreshing memory (saving and reloading topics)
                    If None, no memory refresh is performed
    """
    # Load data - input_path should be a list of files
    if not isinstance(input_path, list):
        raise ValueError("input_path should be a list of persona files for topic categorization")
    
    # Sort files numerically by persona number to ensure proper consecutive batching
    def extract_persona_number(file_path):
        """Extract numeric persona ID from file path for proper sorting."""
        match = re.search(r'persona(\d+)', os.path.basename(file_path))
        return int(match.group(1)) if match else 0
    
    # Sort input files by persona number to ensure consecutive ordering
    input_path_sorted = sorted(input_path, key=extract_persona_number)
    
    # Initialize global topics memory
    global_topics = set()
    
    # Memory refresh setup
    if refresh_mem is not None and refresh_mem > 0:
        print(f"Memory refresh enabled: saving and reloading global topics every {refresh_mem} files")
        print(f"  - Working memory will be completely reset after each refresh")
        print(f"  - All topics will be saved to intermediate files before reset")
        print(f"  - Final merge will combine all intermediate topics")
        intermediate_dir = "data/intermediate_topics"
        os.makedirs(intermediate_dir, exist_ok=True)
        # Clean up any existing intermediate files
        for f in os.listdir(intermediate_dir):
            if f.startswith("global_topics_batch_") and f.endswith(".json"):
                os.remove(os.path.join(intermediate_dir, f))
    
    # Process each file individually to maintain file structure
    processed_files = 0
    current_batch = 0
    
    if parallel:
        # Parallel processing - process multiple files at once
        # Note: In parallel mode, global_topics sharing is more complex
        # We'll process in batches and merge topics after each batch
        
        # Prepare arguments for each file using sorted file list
        file_args = []
        for file_path in input_path_sorted:
            # Each file gets a copy of current global topics
            file_args.append((file_path, llm, global_topics.copy(), verbose))
        
        # Process files in parallel batches
        max_workers = min(llm.rate_limit_per_min, len(file_args))
        batch_size = max_workers
        num_batches = math.ceil(len(file_args) / batch_size)
        
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, len(file_args))
            batch_args = file_args[batch_start_idx:batch_end_idx]
            
            # Update each file's global_topics to current state
            for i, (file_path, llm_instance, _, verbose_flag) in enumerate(batch_args):
                batch_args[i] = (file_path, llm_instance, global_topics.copy(), verbose_flag)
            
            # Extract file paths for logging consecutive order
            batch_file_paths = [args[0] for args in batch_args]
            print(f"Processing topic categorization batch {batch_idx + 1}/{num_batches} ({len(batch_args)} files)")
            print(f"  Batch files (indices {batch_start_idx}-{batch_end_idx-1}): {[os.path.basename(fp) for fp in batch_file_paths[:3]]}{'...' if len(batch_file_paths) > 3 else ''}")
            print(f"  Current global topics count: {len(global_topics)}")
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                # Submit all tasks for this batch - order preserved by batch_args
                future_to_args = {executor.submit(categorize_topics_for_single_file, args): args for args in batch_args}

                # Collect results with progress bar
                batch_results = []
                for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                                 desc=f"Topic Categorization Batch {batch_idx + 1}", 
                                 total=len(batch_args)):
                    try:
                        file_path, file_global_topics = future.result()
                        batch_results.append((file_path, file_global_topics))
                        
                        # Merge topics from this file into global topics
                        if isinstance(file_global_topics, set):
                            global_topics.update(file_global_topics)
                        
                        processed_files += 1
                        if verbose:
                            print(f"✓ Processed topics for {os.path.basename(file_path)}")
                        
                    except Exception as e:
                        args = future_to_args[future]
                        file_path = args[0]
                        batch_results.append((file_path, set()))
                        print(f"Error in topic categorization future for {os.path.basename(file_path)}: {e}")
                
                # Summary for this batch
                successful_files = [os.path.basename(fp) for fp, topics in batch_results if isinstance(topics, set)]
                print(f"Batch {batch_idx + 1} completed: {len(successful_files)}/{len(batch_results)} files processed successfully")
                print(f"Global topics after batch {batch_idx + 1}: {len(global_topics)} total")
                
                # Memory refresh check (for parallel mode)
                if refresh_mem is not None and refresh_mem > 0:
                    files_in_current_batch = len(batch_results)
                    if processed_files % refresh_mem <= files_in_current_batch:
                        current_batch += 1
                        topics_file = save_intermediate_topics(global_topics, current_batch)
                        print(f"🔄 Memory refresh after {processed_files} files: saved {len(global_topics)} topics to {topics_file}")
                        # Reset global_topics completely to start from scratch
                        global_topics = set()
                        if verbose:
                            print(f"  Reset working memory completely - starting fresh with 0 topics")
    
    else:
        # Sequential processing - process one file at a time
        for file_path in tqdm(input_path_sorted, desc="Processing persona files sequentially"):
            try:
                updated_global_topics = categorize_topics_sequential(file_path, llm, global_topics, verbose)
                global_topics = updated_global_topics
                processed_files += 1
                if verbose:
                    print(f"✓ Processed topics for {os.path.basename(file_path)}")
                
                # Memory refresh check (for sequential mode)
                if refresh_mem is not None and refresh_mem > 0 and processed_files % refresh_mem == 0:
                    current_batch += 1
                    topics_file = save_intermediate_topics(global_topics, current_batch)
                    print(f"🔄 Memory refresh after {processed_files} files: saved {len(global_topics)} topics to {topics_file}")
                    # Reset global_topics completely to start from scratch
                    global_topics = set()
                    if verbose:
                        print(f"  Reset working memory completely - starting fresh with 0 topics")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # If memory refresh was used, merge all intermediate topics at the end
    if refresh_mem is not None and refresh_mem > 0:
        print(f"\n🔄 Final memory refresh: saving final batch and merging all topics...")
        # Save the final batch
        if global_topics:
            current_batch += 1
            save_intermediate_topics(global_topics, current_batch)
        
        # Merge all intermediate topics
        merged_global_topics = merge_all_intermediate_topics(verbose=verbose)
        if merged_global_topics:
            global_topics = merged_global_topics
            print(f"  Final merged topics: {len(global_topics)}")
        else:
            print(f"  Warning: No intermediate topics found, using current topics: {len(global_topics)}")

    # Final summary
    print(f"\nTopic Categorization Complete:")
    print(f"  Processed files: {processed_files}/{len(input_path_sorted)}")
    print(f"  Total unique topics discovered: {len(global_topics)}")
    
    if verbose and global_topics:
        print(f"  All topics: {', '.join(sorted(global_topics))}")
    
    # Count topics across all files
    print(f"\nCounting topic occurrences across all files...")
    topic_counts, topic_examples = count_topics_across_files(input_path_sorted)
    
    # Display topic statistics
    print(f"\nTopic Statistics:")
    print(f"  Total topic assignments: {sum(topic_counts.values())}")
    print(f"  Unique topics found: {len(topic_counts)}")
    
    if topic_counts:
        # Sort topics by count (most frequent first)
        sorted_topic_counts = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 10 most frequent topics:")
        for i, (topic, count) in enumerate(sorted_topic_counts[:10]):
            print(f"    {i+1}. {topic}: {count} occurrences")
        
        if verbose:
            print(f"  All topics with counts:")
            for topic, count in sorted_topic_counts:
                print(f"    - {topic}: {count} occurrences")
    
    # Re-categorize least frequent topics
    # If the least frequent topic count is more than 100, we will ignore the whole process
    least_topic_count = sorted_topic_counts[-1][1] if sorted_topic_counts else 0
    if least_topic_count < 100:
        print(f"\nLeast frequent topic count is {least_topic_count}, proceeding with re-categorization...")
        if topic_counts and len(topic_counts) > 1:
            print(f"\nRe-categorizing least frequent topics...")
            topic_changes = recategorize_least_frequent_topics(llm, topic_counts, topic_examples, verbose)
            
            # Apply the topic changes to all files
            if topic_changes:
                apply_topic_changes(input_path_sorted, topic_changes, verbose, parallel)
                
                # Recount topics after changes
                print(f"\nRecounting topics after re-categorization...")
                updated_topic_counts, _ = count_topics_across_files(input_path_sorted)
                
                print(f"Updated Topic Statistics:")
                print(f"  Total topic assignments: {sum(updated_topic_counts.values())}")
                print(f"  Unique topics after re-categorization: {len(updated_topic_counts)}")
                
                # Update global_topics with the final set
                global_topics = set(updated_topic_counts.keys())
    
    # Save global topics to a separate file for reference
    if global_topics:
        topics_file = "data/global_topics.json"
        topics_data = {
            "total_topics": len(global_topics),
            "topics": sorted(list(global_topics)),
            "created_from_files": len(input_path_sorted)
        }
        if topic_counts:
            topics_data["topic_counts"] = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
        utils.save_json(topics_data, topics_file)
        print(f"  Global topics and statistics saved to: {topics_file}")
    
    return global_topics


def main():
    parser = argparse.ArgumentParser(description="Aggregate and optionally clean topic_preference across persona files")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing persona JSON files (defaults to data/raw_data relative to this script)")
    parser.add_argument("--check", action="store_true", help="Clean topic_preference fields in-place: extract after ###Output and remove newlines")
    args = parser.parse_args()

    # Resolve input directory
    if args.input_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(current_dir, "data", "raw_data")
    else:
        input_dir = args.input_dir

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    # Find all JSON files
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    # Sort by persona number for determinism
    def extract_persona_number(file_path):
        match = re.search(r'persona(\d+)', os.path.basename(file_path))
        return int(match.group(1)) if match else 0
    files = sorted(files, key=extract_persona_number)

    topic_counts = {}
    total_files = len(files)
    print(f"Scanning {total_files} files in {input_dir}...")

    for file_path in tqdm(files, desc="Scanning files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read {os.path.basename(file_path)}: {e}")
            continue

        dirty = False
        for _, persona in data.items():
            conversations_by_type = persona.get("conversations", {})
            for conv_list in conversations_by_type.values():
                for conv_elem in conv_list:
                    if 'topic_preference' in conv_elem:
                        raw = conv_elem['topic_preference']
                        cleaned = clean_topic_value(raw)
                        # Count using cleaned value
                        if cleaned:
                            topic_counts[cleaned] = topic_counts.get(cleaned, 0) + 1
                        if args.check and cleaned != raw:
                            conv_elem['topic_preference'] = cleaned
                            dirty = True
        
        # Save cleaned file if needed
        if args.check and dirty:
            utils.save_json(data, file_path, clean=True)

    # Print counts sorted by frequency
    print("\nTopic Preference Counts (cleaned view):")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
