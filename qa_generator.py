import json
from pathlib import Path
import argparse
from json_repair import repair_json
from tqdm import tqdm
import concurrent.futures
import math
import os

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
    # Generate user query prompt and get the question
    prompt = prompts.generate_user_question(element)
    user_query = llm.query_llm(prompt, use_history=False, verbose=verbose)
    user_query = utils.extract_after_token(user_query, '###Output')

    # Generate answer options prompt and get JSON with labeled answers
    who = element['who']
    prompt = prompts.generate_answer_options(element, user_query, who)
    answers = llm.query_llm(prompt, use_history=True, verbose=verbose)
    answers = utils.extract_json_from_response(answers)

    # Set up correct and incorrect answers based on whether the preference belongs to the user themselves or others
    who = element.get("who", [])

    # Attach new keys to element
    element['user_query'] = user_query

    # For updated preference, we find its previous one and assign previous correct option as an incorrect one
    prev_correct = None
    if element['updated'] and conv_list:
        prev_pref = element['prev_pref']
        for curr_element in conv_list:
            if curr_element['preference'] == prev_pref:
                # Previous preference must already have correct answers generated
                prev_correct = curr_element['correct_answer']
            else:
                continue

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
    user_query = llm.query_llm(prompt, use_history=False, verbose=verbose)
    user_query = utils.extract_after_token(user_query, '###Output')

    # Generate answer options prompt and get JSON with labeled answers
    who = element['who']
    prompt = prompts.generate_answer_options_sensitive_info(element, user_query)
    answers = llm.query_llm(prompt, use_history=True, verbose=verbose)
    answers = utils.extract_json_from_response(answers)

    # Attach new keys to element
    element['user_query'] = user_query
    element['correct_answer'] = answers.get('correct')
    incorrect = []
    for key in ('leaked', 'incorrect', 'random'):
        if key in answers:
            incorrect.append(answers[key])
    element['incorrect_answers'] = incorrect

    return element


def generate_qa_for_single_persona(args):
    """Helper function to process QA for a single persona in parallel."""
    uuid, persona, llm, verbose = args
    
    try:
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
        
    except Exception as e:
        print(f"Error processing persona {uuid}: {e}")
        return uuid, None


def generate_qa(llm, input_path, output_dir, parallel=False, verbose=False):
    # Load data - input_path should be a list of files for this use case
    if not isinstance(input_path, list):
        raise ValueError("input_path should be a list of persona files for QA generation")
    
    # Process each file individually to maintain file structure
    processed_files = 0
    
    if parallel:
        # Parallel processing - process multiple files at once
        # Prepare arguments for each file
        file_args = []
        for file_path in input_path:
            file_args.append((file_path, llm, verbose))
        
        # Process files in parallel batches
        max_workers = min(llm.rate_limit_per_min, len(file_args))
        batch_size = max_workers
        num_batches = math.ceil(len(file_args) / batch_size)
        
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, len(file_args))
            batch_args = file_args[batch_start_idx:batch_end_idx]
            
            print(f"Processing QA file batch {batch_idx + 1}/{num_batches} ({len(batch_args)} files)")
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                # Submit all tasks for this batch
                future_to_args = {executor.submit(process_single_file_qa, args): args for args in batch_args}

                # Collect results with progress bar
                for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                                 desc=f"QA File Batch {batch_idx + 1}", 
                                 total=len(batch_args)):
                    try:
                        file_path, success = future.result()
                        
                        if success:
                            processed_files += 1
                            if verbose:
                                print(f"Processed QA for {file_path}")
                        else:
                            print(f"Failed to process QA for {file_path}")
                        
                    except Exception as e:
                        args = future_to_args[future]
                        file_path = args[0]
                        print(f"Error in QA future for {file_path}: {e}")
    else:
        # Sequential processing - process one file at a time
        for file_path in tqdm(input_path, desc="Processing persona files sequentially"):
            try:
                success = process_single_file_qa_sequential(file_path, llm, verbose)
                if success:
                    processed_files += 1
                    if verbose:
                        print(f"Processed QA for {file_path}")
                else:
                    print(f"Failed to process QA for {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Processed QA for {processed_files}/{len(input_path)} persona files")


def process_single_file_qa(args):
    """Process QA for a single persona file in parallel."""
    file_path, llm, verbose = args
    
    try:
        return file_path, process_single_file_qa_sequential(file_path, llm, verbose)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return file_path, False


def process_single_file_qa_sequential(file_path, llm, verbose):
    """Process QA for a single persona file sequentially."""
    try:
        # Load the persona file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each persona in the file
        for uuid, persona in data.items():
            conversations_by_type = persona.get("conversations", {})
            
            for i, (conv_type, conv_list) in enumerate(conversations_by_type.items()):
                if i > 0:
                    continue
                print(f'Processing conv_type: {conv_type} in {file_path}')
                
                for conv_elem in tqdm(conv_list, desc=f"Processing {conv_type}", disable=not verbose, leave=False):
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
        
        # Save the updated data back to the same file
        utils.save_json(data, file_path, clean=True)  # Use clean=True to overwrite
        return True
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

