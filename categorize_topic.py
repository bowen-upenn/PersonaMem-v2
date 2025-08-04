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
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Keep track of categorization statistics
    total_preferences = 0
    new_topics_added = 0
    file_topics = set()
    
    # Process each persona in the file
    for uuid, persona in data.items():
        conversations_by_type = persona.get("conversations", {})
        
        for conv_type, conv_list in conversations_by_type.items():
            if verbose:
                print(f'Processing conv_type: {conv_type} in {os.path.basename(file_path)}')
            
            for conv_elem in tqdm(conv_list, desc=f"Categorizing {conv_type}", disable=not verbose, leave=False):
                try:
                    # Extract preference from the element
                    preference = conv_elem.get('preference')
                    if not preference:
                        continue  # Skip elements without preferences
                    
                    total_preferences += 1
                    
                    # Categorize the preference
                    topic = categorize_single_preference(llm, preference, list(global_topics), verbose=verbose)
                    
                    # Add topic to the element
                    conv_elem["topic"] = topic
                    
                    # Update global topics if this is a new topic
                    if topic not in global_topics:
                        global_topics.add(topic)
                        new_topics_added += 1
                        if verbose:
                            print(f"✓ New topic added: '{topic}'")
                    else:
                        if verbose:
                            print(f"✓ Assigned to existing topic: '{topic}'")
                    
                    # Track topics in this file
                    file_topics.add(topic)
                
                except Exception as e:
                    pref_info = conv_elem.get('preference', 'Unknown preference')
                    print(f"Error categorizing preference '{pref_info}': {e}")
                    continue
    
    # Print categorization statistics
    print(f"Topic Categorization Summary for {os.path.basename(file_path)}:")
    print(f"  Total preferences categorized: {total_preferences}")
    print(f"  New topics created: {new_topics_added}")
    print(f"  Unique topics in this file: {len(file_topics)}")
    if verbose and file_topics:
        print(f"  File topics: {', '.join(sorted(file_topics))}")

    # Save the updated data back to the same file
    utils.save_json(data, file_path, clean=True)  # Use clean=True to overwrite
    
    return global_topics


def categorize_topics(llm, input_path, output_dir=None, parallel=False, verbose=False):
    """
    Main function to categorize topics from preferences in persona files.
    
    Args:
        llm: QueryLLM instance
        input_path: List of persona files to process
        output_dir: Output directory (not used, maintaining compatibility with qa_generator)
        parallel: Whether to process files in parallel
        verbose: Whether to print verbose output
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
    
    # Process each file individually to maintain file structure
    processed_files = 0
    
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
    
    else:
        # Sequential processing - process one file at a time
        for file_path in tqdm(input_path_sorted, desc="Processing persona files sequentially"):
            try:
                updated_global_topics = categorize_topics_sequential(file_path, llm, global_topics, verbose)
                global_topics = updated_global_topics
                processed_files += 1
                if verbose:
                    print(f"✓ Processed topics for {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Final summary
    print(f"\nTopic Categorization Complete:")
    print(f"  Processed files: {processed_files}/{len(input_path_sorted)}")
    print(f"  Total unique topics discovered: {len(global_topics)}")
    
    if verbose and global_topics:
        print(f"  All topics: {', '.join(sorted(global_topics))}")
    
    # Save global topics to a separate file for reference
    if global_topics:
        topics_file = "global_topics.json"
        topics_data = {
            "total_topics": len(global_topics),
            "topics": sorted(list(global_topics)),
            "created_from_files": len(input_path_sorted)
        }
        utils.save_json(topics_data, topics_file)
        print(f"  Global topics saved to: {topics_file}")
    
    return global_topics
