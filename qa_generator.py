import json
from pathlib import Path
import argparse
from json_repair import repair_json
from tqdm import tqdm
import concurrent.futures
import math
import os
import random

import prompts
import utils
from query_llm import QueryLLM


def generate_qa_for_each_element(llm, element, conv_list=None, ask_to_forget=False, verbose=False):
    """
    Generate a QA set for a single scenario element by adding:
    - user_query: a question that elicits personalization based on the element's preference/background.
    - correct_answer: the model's personalized response.
    - incorrect_answers: a list of three incorrect responses (opposite, random, generic).
    Mutates the element dict and returns it.
    """
    # Validate input element structure
    if not isinstance(element, dict):
        print(f"Error: element is not a dictionary: {type(element)}")
        return None
    
    if 'preference' not in element:
        print(f"Error: element missing 'preference' key. Available keys: {list(element.keys())}")
        return None
    
    # Generate user query prompt and get the question
    prompt = prompts.generate_user_question(element)
    try:
        user_query = llm.query_llm(prompt, use_history=False, verbose=verbose)
        user_query = utils.extract_after_token(user_query, '###Output')

        # Generate answer options prompt and get JSON with labeled answers
        who = element['who']
        prompt = prompts.generate_answer_options(element, user_query, who)

        answers = llm.query_llm(prompt, use_history=True, verbose=verbose)
        answers = utils.extract_json_from_response(answers)
    except Exception as e:
        print(f"Error generating qa for element {element}: {e}")
        return None 

    # Set up correct and incorrect answers based on whether the preference belongs to the user themselves or others
    who = element.get("who", [])

    # Attach new keys to element
    element['user_query'] = user_query

    # For updated preference, we find its previous one and assign previous correct option as an incorrect one
    prev_correct = None
    if element.get('updated') and conv_list:
        prev_pref = element.get('prev_pref')
        if prev_pref:
            for curr_element in conv_list:
                # Add safety check for preference key and ensure it's a dict with the expected structure
                if isinstance(curr_element, dict) and curr_element.get('preference') == prev_pref:
                    # Previous preference must already have correct answers generated
                    if 'correct_answer' in curr_element:
                        prev_correct = curr_element['correct_answer']
                    break

    if who == 'self':
        incorrect = []
        if ask_to_forget:
            element['correct_answer'] = answers.get('generic')
            for key in ('random1', 'random2', 'correct'):
                if key in answers:
                    incorrect.append(answers[key])
            incorrect.append(prev_correct)
        elif prev_correct:    # only happen with who == self
            element['correct_answer'] = answers.get('correct')
            for key in ('random1', 'generic'):
                if key in answers:
                    incorrect.append(answers[key])
            incorrect.append(prev_correct)
        else:
            element['correct_answer'] = answers.get('correct')
            for key in ('random1', 'random2', 'generic'):
                if key in answers:
                    incorrect.append(answers[key])
        element['incorrect_answers'] = incorrect
    else:
        element['correct_answer'] = answers.get('incorrect')
        incorrect = []
        for key in ('correct1', 'correct2', 'correct3'):
            if key in answers:
                incorrect.append(answers[key])

        element['incorrect_answers'] = incorrect

    return element


def generate_qa_for_sensitive_info(llm, element, persona, verbose=False):
    # Generate user query prompt and get the question
    prompt = prompts.generate_user_question_sensitive_info(element, persona)
    try:
        user_query = llm.query_llm(prompt, use_history=False, verbose=verbose)
        user_query = utils.extract_after_token(user_query, '###Output')

        # Generate answer options prompt and get JSON with labeled answers
        who = element['who']
        prompt = prompts.generate_answer_options_sensitive_info(element, user_query)
        # try:
        answers = llm.query_llm(prompt, use_history=True, verbose=verbose)
        answers = utils.extract_json_from_response(answers)
    except Exception as e:
        print(f"Error generating qa for element {element}: {e} with and sensitive info {element.get('sensitive_info')}")
        return None 

    # Attach new keys to element
    element['user_query'] = user_query
    element['correct_answer'] = answers.get('correct')
    incorrect = []
    for key in ('leaked', 'incorrect', 'random'):
        if key in answers:
            incorrect.append(answers[key])
    element['incorrect_answers'] = incorrect

    return element


def validate_qa_pair(llm, user_query, correct_answer, incorrect_answers, verbose=False):
    """
    Validate a QA pair by testing if the model can answer correctly without context.
    
    Args:
        llm: QueryLLM instance
        user_query: The question to ask
        correct_answer: The expected correct answer
        incorrect_answers: List of incorrect answer options
        verbose: Whether to print debug information
        
    Returns:
        bool: True if the QA pair is valid (model answers incorrectly without context), 
              False if invalid (model answers correctly without context)
    """
    if not user_query or not correct_answer or not incorrect_answers:
        return False
    
    # Create a list of all answer options and shuffle them
    all_options = [correct_answer] + list(incorrect_answers)
    random.shuffle(all_options)
    
    # Find the position of the correct answer after shuffling
    correct_position = all_options.index(correct_answer)
    
    # Create a multiple choice prompt
    options_text = []
    for i, option in enumerate(all_options):
        options_text.append(f"{chr(65+i)}. {option}")  # A, B, C, D, etc.
    
    validation_prompt = f"""Question: {user_query}

Please select the most appropriate answer for the current user from the following options:
{chr(10).join(options_text)}

Think step by step and give your final answer in \\boxed{{}} with only the letter (A, B, C, or D) of your choice."""
    
    try:
        # Query the model without any conversation history
        llm.reset_history()  # Make sure no history is used
        response = llm.query_llm(validation_prompt, use_history=False, verbose=verbose)
        
        if not response:
            return True  # If no response, assume the pair is valid
        
        # Extract the letter from the response, looking for \boxed{} format first
        response = response.strip()
        selected_letter = None
        
        # First try to find answer in \boxed{} format
        import re
        boxed_match = re.search(r'\\boxed\{([A-D])\}', response, re.IGNORECASE)
        if boxed_match:
            selected_letter = boxed_match.group(1).upper()
        else:
            # Fall back to looking for the first letter A-D in the response
            for char in response.upper():
                if char in ['A', 'B', 'C', 'D']:
                    selected_letter = char
                    break
        
        if selected_letter is None:
            return True  # If we can't parse the answer, assume the pair is valid
        
        # Convert letter to index
        selected_index = ord(selected_letter) - ord('A')
        
        # Check if the model selected the correct answer
        model_answered_correctly = (selected_index == correct_position)
        
        if verbose:
            print(f"Question: {user_query}")
            print(f"Model selected: {selected_letter} (index {selected_index})")
            print(f"Correct answer was at index: {correct_position}")
            print(f"Model answered correctly: {model_answered_correctly}")
            print(f"QA pair is {'INVALID' if model_answered_correctly else 'VALID'}")
        
        # Return False if model answered correctly (meaning the QA pair is problematic)
        # Return True if model answered incorrectly (meaning the QA pair is good)
        return not model_answered_correctly
        
    except Exception as e:
        if verbose:
            print(f"Error during QA validation: {e}")
        return True  # If validation fails, assume the pair is valid to be safe


def generate_qa_for_single_persona(args):
    """Helper function to process QA for a single persona in parallel."""
    uuid, persona, llm, verbose = args
    
    # try:
    processed_persona = persona.copy()
    conversations_by_type = persona.get("conversations", {})
    
    for conv_type, conv_list in conversations_by_type.items():
        if verbose:
            print(f'Processing conv_type: {conv_type} for {uuid}')
        
        for conv_elem in tqdm(conv_list, desc=f"Processing {conv_type} for {uuid}", disable=not verbose):
            llm.reset_history()
            curr_persona = persona.get("persona", "")

            if conv_type == 'knowledge_query':
                if 'idx_repeat' not in conv_elem or conv_elem['idx_repeat'] < 2:
                    continue  # Skip if user didn't ask topic >= 3 times

            if "sensitive_info" in conv_elem:
                qa_fields = generate_qa_for_sensitive_info(llm, conv_elem, curr_persona, verbose=verbose)
                conv_elem.update({
                    "user_query": qa_fields.get("user_query"),
                    "correct_answer": qa_fields.get("correct_answer"),
                    "incorrect_answers": qa_fields.get("incorrect_answers"),
                })
            else:
                ask_to_forget = conv_elem['pref_type'] == "ask_to_forget"
                qa_fields = generate_qa_for_each_element(llm, conv_elem, conv_list, ask_to_forget=ask_to_forget, verbose=verbose)
                conv_elem.update({
                    "user_query": qa_fields.get("user_query"),
                    "correct_answer": qa_fields.get("correct_answer"),
                    "incorrect_answers": qa_fields.get("incorrect_answers"),
                })
    
    return uuid, processed_persona
        
    # except Exception as e:
    #     print(f"Error processing persona {uuid}: {e}")
    #     return uuid, None


def generate_qa(llm, input_path, output_dir, parallel=False, verbose=False, validate_qa=False):
    # Load data - input_path should be a list of files for this use case
    if not isinstance(input_path, list):
        raise ValueError("input_path should be a list of persona files for QA generation")
    
    # Sort files numerically by persona number to ensure proper consecutive batching
    def extract_persona_number(file_path):
        """Extract numeric persona ID from file path for proper sorting."""
        import re
        match = re.search(r'persona(\d+)', os.path.basename(file_path))
        return int(match.group(1)) if match else 0
    
    # Sort input files by persona number to ensure consecutive ordering
    input_path_sorted = sorted(input_path, key=extract_persona_number)
    
    # Process each file individually to maintain file structure
    processed_files = 0
    
    if parallel:
        # Parallel processing - process multiple files at once
        # Prepare arguments for each file using sorted file list
        file_args = []
        for file_path in input_path_sorted:
            file_args.append((file_path, llm, verbose, validate_qa))
        
        # Process files in parallel batches
        max_workers = min(llm.rate_limit_per_min, len(file_args))
        batch_size = max_workers
        num_batches = math.ceil(len(file_args) / batch_size)
        
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, len(file_args))
            batch_args = file_args[batch_start_idx:batch_end_idx]
            
            # Extract file paths for logging consecutive order
            batch_file_paths = [args[0] for args in batch_args]
            print(f"Processing QA file batch {batch_idx + 1}/{num_batches} ({len(batch_args)} files)")
            print(f"  Batch files (indices {batch_start_idx}-{batch_end_idx-1}): {[os.path.basename(fp) for fp in batch_file_paths[:3]]}{'...' if len(batch_file_paths) > 3 else ''}")
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                # Submit all tasks for this batch - order preserved by batch_args
                future_to_args = {executor.submit(process_single_file_qa, args): args for args in batch_args}

                # Collect results with progress bar
                for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                                 desc=f"QA File Batch {batch_idx + 1}", 
                                 total=len(batch_args)):
                    # try:
                    file_path, success = future.result()
                    
                    if success:
                        processed_files += 1
                        if verbose:
                            print(f"Processed QA for {file_path}")
                    else:
                        print(f"Failed to process QA for {file_path}")
                        
                    # except Exception as e:
                    #     args = future_to_args[future]
                    #     file_path = args[0]
                    #     print(f"Error in QA future for {file_path}: {e}")
    else:
        # Sequential processing - process one file at a time
        for file_path in tqdm(input_path_sorted, desc="Processing persona files sequentially"):
            try:
                success = process_single_file_qa_sequential(file_path, llm, verbose, validate_qa)
                if success:
                    processed_files += 1
                    if verbose:
                        print(f"Processed QA for {file_path}")
                else:
                    print(f"Failed to process QA for {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Processed QA for {processed_files}/{len(input_path_sorted)} persona files")


def process_single_file_qa(args):
    """Process QA for a single persona file in parallel."""
    file_path, llm, verbose, validate_qa = args
    
    # try:
    return file_path, process_single_file_qa_sequential(file_path, llm, verbose, validate_qa)
    # except Exception as e:
    #     print(f"Error processing file {file_path}: {e}")
    #     return file_path, False


def process_single_file_qa_sequential(file_path, llm, verbose, validate_qa=False):
    """Process QA for a single persona file sequentially."""
    # Load the persona file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if QA has already been generated for this file
    # Since there should be only 1 persona per file, check if any conversation has correct_answer
    # Check the same conversation types that will be processed later
    for uuid, persona in data.items():
        conversations_by_type = persona.get("conversations", {})
        
        for conv_type, conv_list in conversations_by_type.items():
            if conv_type != "personal_email":
                continue  # skip conversation types that won't be processed
            for conv_elem in conv_list:
                if "correct_answer" in conv_elem:
                    print(f"QA already exists for {file_path}, skipping...")
                    return True  # Return True to indicate successful processing (already done)
    
    # Keep track of validation statistics
    total_qa_pairs = 0
    valid_qa_pairs = 0
    invalid_qa_pairs = 0
    
    # Process each persona in the file
    for uuid, persona in data.items():
        conversations_by_type = persona.get("conversations", {})
        
        for i, (conv_type, conv_list) in enumerate(conversations_by_type.items()):
            # if conv_type not in ['personal_email', 'professional_email', 'social_media_post']:
            #     continue
            print(f'Processing conv_type: {conv_type} in {os.path.basename(file_path)}')
            
            # Create a new list to store only valid QA pairs
            valid_conv_list = []
            
            for conv_elem in tqdm(conv_list, desc=f"Processing {conv_type} in {os.path.basename(file_path)}", leave=False):
                try:
                    if conv_elem['preference'] not in persona["health_and_medical_conditions"]:
                        if conv_type not in ['personal_email', 'professional_email', 'social_media_post']:
                            continue

                    llm.reset_history()
                    curr_persona = persona.get("persona", "")

                    if conv_type == 'knowledge_query':
                        if 'idx_repeat' not in conv_elem or conv_elem['idx_repeat'] < 2:
                            continue  # Skip if user didn't ask topic >= 3 times

                    # Generate QA fields
                    qa_fields = None
                    if "sensitive_info" in conv_elem:
                        qa_fields = generate_qa_for_sensitive_info(llm, conv_elem, curr_persona, verbose=verbose)
                    else:
                        ask_to_forget = conv_elem.get('pref_type') == "ask_to_forget"
                        qa_fields = generate_qa_for_each_element(llm, conv_elem, conv_list, ask_to_forget=ask_to_forget, verbose=verbose)
                    
                    # Check if QA generation was successful
                    if qa_fields is None:
                        if verbose:
                            print(f"QA generation failed for element: {conv_elem.get('preference', 'Unknown preference')}")
                        continue  # Skip this element if QA generation failed
                    
                    user_query = qa_fields.get("user_query")
                    correct_answer = qa_fields.get("correct_answer")
                    incorrect_answers = qa_fields.get("incorrect_answers")
                    
                    # Validate the QA pair only if validate_qa is enabled
                    if validate_qa:
                        total_qa_pairs += 1
                        is_valid = validate_qa_pair(llm, user_query, correct_answer, incorrect_answers, verbose=verbose)
                        
                        if is_valid:
                            # Only add to the conversation list if validation passes
                            conv_elem.update({
                                "user_query": user_query,
                                "correct_answer": correct_answer,
                                "incorrect_answers": incorrect_answers,
                            })
                            valid_conv_list.append(conv_elem)
                            valid_qa_pairs += 1
                            if verbose:
                                print(f"✓ QA pair VALID - Added to dataset")
                        else:
                            invalid_qa_pairs += 1
                            if verbose:
                                print(f"✗ QA pair INVALID - Skipped (model can answer without context)")
                    else:
                        # Skip validation, add all QA pairs
                        conv_elem.update({
                            "user_query": user_query,
                            "correct_answer": correct_answer,
                            "incorrect_answers": incorrect_answers,
                        })
                        valid_conv_list.append(conv_elem)
                        total_qa_pairs += 1
                        valid_qa_pairs += 1
                
                except Exception as e:
                    if "sensitive_info" in conv_elem:
                        print(f"Error processing persona conv_type {conv_elem.get('preference', 'Unknown')}: {e} with sensitive info {conv_elem.get('sensitive_info', 'Unknown')}")
                    else:
                        pref_info = conv_elem.get('preference', 'Unknown preference')
                        pref_type = conv_elem.get('pref_type', 'Unknown type')
                        print(f"Error processing persona conv_type '{pref_info}' (type: {pref_type}): {e}")
                        if verbose:
                            print(f"  Full element keys: {list(conv_elem.keys())}")
                    continue
            
            # Replace the original conversation list with the filtered valid one
            conversations_by_type[conv_type] = valid_conv_list

    # Print validation statistics
    print(f"QA Generation Summary for {os.path.basename(file_path)}:")
    print(f"  Total QA pairs generated: {total_qa_pairs}")
    if validate_qa:
        print(f"  Valid QA pairs (saved): {valid_qa_pairs}")
        print(f"  Invalid QA pairs (discarded): {invalid_qa_pairs}")
        if total_qa_pairs > 0:
            print(f"  Validation pass rate: {valid_qa_pairs/total_qa_pairs*100:.1f}%")
    else:
        print(f"  QA pairs saved (validation disabled): {valid_qa_pairs}")
        print(f"  Note: QA validation was disabled - all generated pairs were saved")

    # Save the updated data back to the same file
    utils.save_json(data, file_path, clean=True)  # Use clean=True to overwrite
    return True
