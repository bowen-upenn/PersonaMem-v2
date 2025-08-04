#!/usr/bin/env python3
"""
Script to process only the multimodal conversations for existing persona JSON files.
This script will:
1. Read existing JSON files
2. Process matched_images to generate multimodal conversations
3. Generate QA pairs for multimodal conversations
4. Update only the conversations.multimodal field without affecting other content
"""

import json
import os
import glob
import argparse
from tqdm import tqdm
import random
import concurrent.futures
import math
import re

import utils
import prompts
from query_llm import QueryLLM
from qa_generator import generate_qa_for_each_element
from conv_generator import find_preference_from_image_and_generate_conversations


def natural_sort_key(text):
    """
    Generate a key for natural sorting that treats numbers numerically.
    This ensures proper ordering like: 1, 2, 9, 10, 11, 100 instead of 1, 10, 100, 11, 2, 9
    """
    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()
    
    return [convert(c) for c in re.split('([0-9]+)', text)]


def process_multimodal_for_single_file_thread(args):
    """
    Thread-safe function to process multimodal conversations for a single file.
    
    Args:
        args: tuple containing (file_path, llm_config, verbose, generate_qa)
    
    Returns:
        tuple: (file_path, success, multimodal_count, error_msg)
    """
    file_path, llm_config, verbose, generate_qa = args
    
    try:
        # Initialize LLM for this thread
        llm = QueryLLM(llm_config, use_o4=True)
        
        # Process the file
        success = process_multimodal_for_single_file(
            file_path, llm, verbose=verbose, generate_qa=generate_qa
        )
        
        if success:
            # Count multimodal conversations
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            persona_id = list(data.keys())[0]
            multimodal_count = len(data[persona_id]["conversations"]["multimodal"])
            return file_path, True, multimodal_count, None
        else:
            return file_path, False, 0, "Processing failed"
            
    except Exception as e:
        return file_path, False, 0, str(e)


def process_multimodal_for_single_file(file_path, llm, verbose=False, generate_qa=True):
    """
    Process multimodal conversations for a single JSON file.
    
    Args:
        file_path: Path to the JSON file
        llm: QueryLLM instance
        verbose: Whether to print verbose output
        generate_qa: Whether to generate QA pairs for conversations
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Load existing JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get the first (and likely only) persona from the file
        persona_id = list(data.keys())[0]
        persona_data = data[persona_id]
        
        # Check if matched_images exist
        matched_images = persona_data.get("matched_images", [])
        if not matched_images:
            if verbose:
                print(f"No matched_images found in {file_path}")
            return False
        
        # Initialize conversations structure if it doesn't exist
        if "conversations" not in persona_data:
            persona_data["conversations"] = {}
        
        # Initialize multimodal list (this will overwrite existing multimodal conversations)
        persona_data["conversations"]["multimodal"] = []
        
        # Get persona string for prompts
        persona_str = json.dumps(persona_data, ensure_ascii=False)
        
        print(f"Processing {len(matched_images)} images for {os.path.basename(file_path)}")
        
        # Process each image
        for image_idx, image_path in enumerate(tqdm(matched_images, desc="Processing images")):
            llm.reset_history()
            
            # Determine if this should be "others" preference (later images are less relevant)
            is_others_pref = image_idx > 0.67 * len(matched_images)
            
            # Check if image file exists
            if not os.path.exists(image_path):
                if verbose:
                    print(f"Image not found: {image_path}")
                continue
            
            try:
                # Generate preference and conversations for this image
                element = find_preference_from_image_and_generate_conversations(
                    llm, persona_str, image_path, persona_data["conversations"], 
                    is_others_pref, verbose=verbose
                )
                
                if element and generate_qa:
                    # Generate QA for this element
                    try:
                        qa_element = generate_qa_for_each_element(
                            llm, element, verbose=verbose
                        )
                        if qa_element:
                            # Update the element in conversations with QA data
                            for conv_element in persona_data["conversations"]["multimodal"]:
                                if (conv_element.get('preference') == element['preference'] and 
                                    conv_element.get('image_path') == element['image_path']):
                                    conv_element.update({
                                        'user_query': qa_element.get('user_query'),
                                        'correct_answer': qa_element.get('correct_answer'),
                                        'incorrect_answers': qa_element.get('incorrect_answers')
                                    })
                                    break
                    except Exception as e:
                        if verbose:
                            print(f"Failed to generate QA for image {image_idx}: {e}")
                        
            except Exception as e:
                if verbose:
                    print(f"Failed to process image {image_idx} ({image_path}): {e}")
                continue
        
        # Save the updated file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        multimodal_count = len(persona_data["conversations"]["multimodal"])
        print(f"✅ Updated {file_path} with {multimodal_count} multimodal conversations")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def process_multimodal_batch(input_pattern, llm_config, verbose=False, generate_qa=True, max_files=None, parallel=False):
    """
    Process multimodal conversations for multiple JSON files.
    
    Args:
        input_pattern: Glob pattern to match JSON files (e.g., "data/raw_data/*_persona*.json")
        llm_config: Dictionary with LLM configuration
        verbose: Whether to print verbose output
        generate_qa: Whether to generate QA pairs for conversations
        max_files: Maximum number of files to process (None for all)
        parallel: Whether to use parallel processing
        
    Returns:
        tuple: (successful_count, total_count)
    """
    # Find all matching files
    files = glob.glob(input_pattern)
    files = sorted(files, key=natural_sort_key)
    
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return 0, 0
    
    if max_files:
        files = files[:max_files]
    
    print(f"Found {len(files)} files to process")
    
    successful_count = 0
    
    if parallel:
        # Parallel processing
        rate_limit = llm_config.get("rate_limit_per_min", 10)
        max_workers = min(rate_limit, len(files))
        batch_size = max_workers
        num_batches = math.ceil(len(files) / batch_size)
        
        print(f"Using parallel processing with {max_workers} workers in {num_batches} batches")
        
        # Prepare arguments for each file
        file_args = [(file_path, llm_config, verbose, generate_qa) for file_path in files]
        
        # Process files in parallel batches
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, len(files))
            batch_args = file_args[batch_start_idx:batch_end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_args)} files)")
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                # Submit all tasks for this batch
                future_to_args = {executor.submit(process_multimodal_for_single_file_thread, args): args for args in batch_args}
                
                # Collect results with progress bar
                for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                                 desc=f"Batch {batch_idx + 1} files", 
                                 total=len(batch_args)):
                    try:
                        file_path, success, multimodal_count, error_msg = future.result()
                        
                        if success:
                            successful_count += 1
                            if verbose:
                                print(f"✅ {os.path.basename(file_path)}: {multimodal_count} multimodal conversations")
                        else:
                            if verbose:
                                print(f"❌ {os.path.basename(file_path)}: {error_msg}")
                                
                    except Exception as e:
                        args = future_to_args[future]
                        file_path = args[0]
                        if verbose:
                            print(f"❌ Error in future for {os.path.basename(file_path)}: {e}")
    else:
        # Sequential processing
        llm = QueryLLM(**llm_config)
        
        # Process each file
        for file_path in tqdm(files, desc="Processing files"):
            if verbose:
                print(f"\nProcessing: {file_path}")
            
            success = process_multimodal_for_single_file(
                file_path, llm, verbose=verbose, generate_qa=generate_qa
            )
            
            if success:
                successful_count += 1
    
    return successful_count, len(files)


def main():
    parser = argparse.ArgumentParser(description="Process multimodal conversations for existing persona files")
    parser.add_argument("--input_pattern", type=str, default="data/raw_data/*_persona*.json",
                        help="Glob pattern to match JSON files (e.g., 'data/raw_data/*_persona*.json')")
    parser.add_argument("--rate_limit", type=int, default=10,
                        help="Rate limit per minute")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process")
    parser.add_argument("--no_qa", action="store_true",
                        help="Skip QA generation, only generate conversations")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel processing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # Prepare LLM configuration
    llm_config = {
        "rate_limit_per_min": args.rate_limit
    }
    
    # Process files
    successful, total = process_multimodal_batch(
        input_pattern=args.input_pattern,
        llm_config=llm_config,
        verbose=args.verbose,
        generate_qa=not args.no_qa,
        max_files=args.max_files,
        parallel=args.parallel
    )
    
    print(f"\n📊 Results: {successful}/{total} files processed successfully")
    
    if successful < total:
        print(f"⚠️  {total - successful} files failed to process")


if __name__ == "__main__":
    main()
