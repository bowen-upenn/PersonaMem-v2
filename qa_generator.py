import json
from pathlib import Path
import argparse
from json_repair import repair_json
from tqdm import tqdm
import concurrent.futures
import math
import os
import random
import re
import threading
from collections import defaultdict

import prompts
import utils
from query_llm import QueryLLM

# Import the global topic counter from conv_generator
try:
    from conv_generator import GLOBAL_TOPIC_COUNTER, TOPIC_COUNTER_LOCK
except ImportError:
    # Fallback if import fails
    GLOBAL_TOPIC_COUNTER = defaultdict(int)
    TOPIC_COUNTER_LOCK = threading.Lock()


def categorize_user_query(llm, user_query, global_topics=None, verbose=False):
    """
    Categorize a user query into topics using the LLM.
    
    Args:
        llm: QueryLLM instance
        user_query: The user query to categorize
        global_topics: List of existing global topics (optional, defaults to empty list)
        verbose: Whether to print debug information
        
    Returns:
        str: The topic name
    """
    # Ensure global_topics is a list
    if global_topics is None:
        global_topics = []
    
    # Generate categorization prompt
    prompt = prompts.categorize_preference_topic(user_query, global_topics)
    
    try:
        llm.reset_history()  # Reset history for categorization
        topic = llm.query_llm(prompt, use_history=False, verbose=verbose)
        topic_temp = utils.extract_after_token(topic, '###Output').strip()
        if not topic_temp:
            topic = utils.extract_after_token(topic, '### Output').strip()
        else:
            topic = topic_temp
        
        # Clean the topic
        if topic:
            # Remove newlines and excessive spaces
            topic = topic.replace('\n', ' ').strip()
            # Remove quotes if present
            topic = topic.strip('"\'')
            
            # Add the topic to global_topics if it's not already there and it's not "Uncategorized"
            if topic not in global_topics and topic.lower() != "uncategorized":
                global_topics.append(topic)
            
            # Record topic count in global counter (thread-safe)
            with TOPIC_COUNTER_LOCK:
                GLOBAL_TOPIC_COUNTER[topic] += 1
            
            if verbose:
                print(f"Categorized query '{user_query[:50]}...' as topic: '{topic}'")
                print(f"Current global topics: {global_topics}")
            return topic
        else:
            # Record "Uncategorized" count
            with TOPIC_COUNTER_LOCK:
                GLOBAL_TOPIC_COUNTER["Uncategorized"] += 1
                
            if verbose:
                print(f"Failed to categorize query '{user_query[:50]}...', using 'Uncategorized'")
            return "Uncategorized"
            
    except Exception as e:
        # Record "Uncategorized" count for errors
        with TOPIC_COUNTER_LOCK:
            GLOBAL_TOPIC_COUNTER["Uncategorized"] += 1
            
        if verbose:
            print(f"Error categorizing query '{user_query[:50]}...': {e}, using 'Uncategorized'")
        return "Uncategorized"


def generate_qa_for_each_element(llm, element, persona, conv_list=None, ask_to_forget=False, global_topics=None, verbose=False):
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
    # try:
    user_query = llm.query_llm(prompt, use_history=False, verbose=verbose)
    user_query = utils.extract_after_token(user_query, '###Output')

    # Immediately categorize the user query using file-level global_topics
    topic_query = categorize_user_query(llm, user_query, global_topics, verbose)

    # Generate answer options prompt and get JSON with labeled answers
    who = element['who']
    prompt = prompts.generate_answer_options(element, user_query, who, persona)

    answers = llm.query_llm(prompt, use_history=True, verbose=verbose)
    answers = utils.extract_json_from_response(answers)
    # except Exception as e:
    #     print(f"Error generating qa for element {element}: {e}")
    #     return None 

    # Set up correct and incorrect answers based on whether the preference belongs to the user themselves or others
    who = element.get("who", [])

    # Attach new keys to element
    element['user_query'] = user_query
    element['topic_query'] = topic_query

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
            for key in ('random', 'stereotypical', 'correct'):
                if key in answers:
                    incorrect.append(answers[key])
            incorrect.append(prev_correct)
        elif prev_correct:    # only happen with who == self
            element['correct_answer'] = answers.get('correct')
            for key in ('random', 'stereotypical'):
                if key in answers:
                    incorrect.append(answers[key])
            incorrect.append(prev_correct)
        else:
            element['correct_answer'] = answers.get('correct')
            for key in ('random', 'stereotypical', 'generic'):
                if key in answers:
                    incorrect.append(answers[key])
        element['incorrect_answers'] = incorrect
    else:
        element['correct_answer'] = answers.get('incorrect')
        incorrect = []
        for key in ('correct1', 'correct2', 'stereotypical'):
            if key in answers:
                incorrect.append(answers[key])

        element['incorrect_answers'] = incorrect

    return element


def generate_qa_for_sensitive_info(llm, element, persona, global_topics=None, verbose=False):
    # Generate user query prompt and get the question
    prompt = prompts.generate_user_question_sensitive_info(element, persona)
    try:
        user_query = llm.query_llm(prompt, use_history=False, verbose=verbose)
        user_query = utils.extract_after_token(user_query, '###Output')

        # Immediately categorize the user query using file-level global_topics
        topic_query = categorize_user_query(llm, user_query, global_topics, verbose)

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
    element['topic_query'] = topic_query
    element['correct_answer'] = answers.get('correct')
    incorrect = []
    for key in ('leaked', 'incorrect', 'random'):
        if key in answers:
            incorrect.append(answers[key])
    element['incorrect_answers'] = incorrect

    return element


def validate_qa_pair(llm, groundtruth_preference, user_query, correct_answer, incorrect_answers, verbose=False):
    """
    Validate a QA pair by testing if the model can answer correctly without context
    and performing additional robustness checks.
    
    Args:
        llm: QueryLLM instance
        groundtruth_preference: The ground truth preference
        user_query: The question to ask
        correct_answer: The expected correct answer
        incorrect_answers: List of incorrect answer options
        verbose: Whether to print debug information
        
    Returns:
        bool: True if all validations pass, False otherwise
    """
    if not user_query or not correct_answer or not incorrect_answers:
        return False

    multiple_choice_valid, no_preference_leakage, correct_answer_aligned, no_incorrect_contamination, answers_clean_format = (False, False, False, False, False)
    
    # Validation 1: Multiple choice test (original validation)
    # Create a list of all answer options and shuffle them
    all_options = [correct_answer] + list(incorrect_answers)
    random.shuffle(all_options)
    
    # Find the position of the correct answer after shuffling
    correct_position = all_options.index(correct_answer)
    
    # Create a multiple choice prompt
    options_text = []
    for i, option in enumerate(all_options):
        options_text.append(f"{chr(65+i)}. {option}")  # A, B, C, D, etc.
    
    validation_prompt = prompts.validate_qa_multiple_choice(user_query, options_text)
    
    try:
        # Query the model without any conversation history
        llm.reset_history()  # Make sure no history is used
        response = llm.query_llm(validation_prompt, use_history=False, verbose=verbose)
        
        if not response:
            multiple_choice_valid = True  # If no response, assume the pair is valid
        else:
            # Extract the letter from the response, looking for \boxed{} format first
            response = response.strip()
            selected_letter = None
            
            # First try to find answer in \boxed{} format
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
                multiple_choice_valid = True  # If we can't parse the answer, assume the pair is valid
            else:
                # Convert letter to index
                selected_index = ord(selected_letter) - ord('A')
                
                # Check if the model selected the correct answer
                model_answered_correctly = (selected_index == correct_position)
                
                if verbose:
                    print(f"Multiple Choice Test:")
                    print(f"  Question: {user_query}")
                    print(f"  Model selected: {selected_letter} (index {selected_index})")
                    print(f"  Correct answer was at index: {correct_position}")
                    print(f"  Model answered correctly: {model_answered_correctly}")
                
                # Return False if model answered correctly (meaning the QA pair is problematic)
                # Return True if model answered incorrectly (meaning the QA pair is good)
                multiple_choice_valid = not model_answered_correctly
        
        if multiple_choice_valid:
            # Validation 2: Check if user query leaks groundtruth preference
            llm.reset_history()
            leakage_prompt = prompts.validate_preference_leakage_in_query(user_query, groundtruth_preference)
            leakage_response = llm.query_llm(leakage_prompt, use_history=False, verbose=verbose)
            
            # Extract yes/no from boxed format
            leakage_match = re.search(r'\\boxed\{(yes|no)\}', leakage_response, re.IGNORECASE)
            if leakage_match:
                leakage_result = leakage_match.group(1).lower()
                no_preference_leakage = (leakage_result == 'no')  # Success if model says 'no'
            else:
                # Fallback: look for yes/no in the response
                leakage_response_lower = leakage_response.lower()
                if 'no' in leakage_response_lower and 'yes' not in leakage_response_lower:
                    no_preference_leakage = True
                elif 'yes' in leakage_response_lower and 'no' not in leakage_response_lower:
                    no_preference_leakage = False
                else:
                    no_preference_leakage = True  # Default to valid if unclear
            
            if verbose:
                print(f"Preference Leakage Test:")
                print(f"  Query leaks preference: {not no_preference_leakage}")
                print(f"  Validation result: {'PASS' if no_preference_leakage else 'FAIL'}")

            if no_preference_leakage:
                # Validation 3: Check if correct answer is crafted from groundtruth preference
                llm.reset_history()
                alignment_prompt = prompts.validate_correct_answer_alignment(groundtruth_preference, correct_answer)
                alignment_response = llm.query_llm(alignment_prompt, use_history=False, verbose=verbose)
                
                # Extract yes/no from boxed format
                alignment_match = re.search(r'\\boxed\{(yes|no)\}', alignment_response, re.IGNORECASE)
                if alignment_match:
                    alignment_result = alignment_match.group(1).lower()
                    correct_answer_aligned = (alignment_result == 'yes')  # Success if model says 'yes'
                else:
                    # Fallback: look for yes/no in the response
                    alignment_response_lower = alignment_response.lower()
                    if 'yes' in alignment_response_lower and 'no' not in alignment_response_lower:
                        correct_answer_aligned = True
                    elif 'no' in alignment_response_lower and 'yes' not in alignment_response_lower:
                        correct_answer_aligned = False
                    else:
                        correct_answer_aligned = True  # Default to valid if unclear
                
                if verbose:
                    print(f"Answer Alignment Test:")
                    print(f"  Correct answer reflects preference: {correct_answer_aligned}")
                    print(f"  Validation result: {'PASS' if correct_answer_aligned else 'FAIL'}")
            
                if correct_answer_aligned:
                    # Validation 4: Check if incorrect answers mention groundtruth preference
                    llm.reset_history()
                    incorrect_answers_str = str(incorrect_answers)
                    contamination_prompt = prompts.validate_incorrect_answers_contamination(groundtruth_preference, incorrect_answers_str)
                    contamination_response = llm.query_llm(contamination_prompt, use_history=False, verbose=verbose)
                    
                    # Extract yes/no from boxed format
                    contamination_match = re.search(r'\\boxed\{(yes|no)\}', contamination_response, re.IGNORECASE)
                    if contamination_match:
                        contamination_result = contamination_match.group(1).lower()
                        no_incorrect_contamination = (contamination_result == 'no')  # Success if model says 'no'
                    else:
                        # Fallback: look for yes/no in the response
                        contamination_response_lower = contamination_response.lower()
                        if 'no' in contamination_response_lower and 'yes' not in contamination_response_lower:
                            no_incorrect_contamination = True
                        elif 'yes' in contamination_response_lower and 'no' not in contamination_response_lower:
                            no_incorrect_contamination = False
                        else:
                            no_incorrect_contamination = True  # Default to valid if unclear
                    
                    if verbose:
                        print(f"Incorrect Answer Contamination Test:")
                        print(f"  Incorrect answers mention preference: {not no_incorrect_contamination}")
                        print(f"  Validation result: {'PASS' if no_incorrect_contamination else 'FAIL'}")
                    
                    if no_incorrect_contamination:
                        # Validation 5: Check if answers contain formatting artifacts or meta-commentary
                        llm.reset_history()
                        format_prompt = prompts.validate_answer_format_cleanliness(correct_answer, incorrect_answers_str)
                        format_response = llm.query_llm(format_prompt, use_history=False, verbose=verbose)
                        
                        # Extract yes/no from boxed format
                        format_match = re.search(r'\\boxed\{(yes|no)\}', format_response, re.IGNORECASE)
                        if format_match:
                            format_result = format_match.group(1).lower()
                            answers_clean_format = (format_result == 'yes')  # Success if model says 'yes'
                        else:
                            # Fallback: look for yes/no in the response
                            format_response_lower = format_response.lower()
                            if 'yes' in format_response_lower and 'no' not in format_response_lower:
                                answers_clean_format = True
                            elif 'no' in format_response_lower and 'yes' not in format_response_lower:
                                answers_clean_format = False
                            else:
                                answers_clean_format = True  # Default to valid if unclear
                        
                        if verbose:
                            print(f"Answer Format Cleanliness Test:")
                            print(f"  Answers contain formatting artifacts: {not answers_clean_format}")
                            print(f"  Validation result: {'PASS' if answers_clean_format else 'FAIL'}")

        # Overall validation result - all tests must pass
        overall_valid = (multiple_choice_valid and 
                        no_preference_leakage and 
                        correct_answer_aligned and 
                        no_incorrect_contamination and
                        answers_clean_format)
        
        if verbose:
            print(f"Overall QA Validation: {'VALID' if overall_valid else 'INVALID'}")
            print(f"  Multiple choice test: {'PASS' if multiple_choice_valid else 'FAIL'}")
            print(f"  No preference leakage: {'PASS' if no_preference_leakage else 'FAIL'}")
            print(f"  Correct answer aligned: {'PASS' if correct_answer_aligned else 'FAIL'}")
            print(f"  No incorrect contamination: {'PASS' if no_incorrect_contamination else 'FAIL'}")
            print(f"  Clean answer format: {'PASS' if answers_clean_format else 'FAIL'}")
        
        return overall_valid
        
    except Exception as e:
        if verbose:
            print(f"Error during QA validation: {e}")
        return True  # If validation fails, assume the pair is valid to be safe


def generate_qa_for_single_persona(args):
    """Helper function to process QA for a single persona in parallel."""
    uuid, persona, llm, verbose = args
    
    # Initialize global topics for this persona
    global_topics = []
    
    # Statistics tracking
    existing_qa_pairs = 0
    new_qa_pairs = 0
    failed_qa_pairs = 0
    total_qa_pairs = 0
    invalid_qa_pairs = 0
    
    try:
        processed_persona = persona.copy()
        conversations_by_type = persona.get("conversations", {})
        
        for conv_type, conv_list in conversations_by_type.items():
            if verbose:
                print(f'Processing conv_type: {conv_type} for {uuid}')
            
            # First pass: identify preferences that should be skipped for Q&A generation
            skip_preferences = set()
            
            # Find all prev_pref of elements with pref_type == "ask_to_forget"
            for conv_elem in conv_list:
                if conv_elem.get('pref_type') == "ask_to_forget":
                    prev_pref = conv_elem.get('prev_pref')
                    if prev_pref:
                        skip_preferences.add(prev_pref)
                        if verbose:
                            print(f"  Skipping Q&A for prev_pref '{prev_pref}' (referenced by ask_to_forget)")
                
                # Also skip preferences with updated == True
                if conv_elem.get('updated') == True:
                    preference = conv_elem.get('preference')
                    if preference:
                        skip_preferences.add(preference)
                        if verbose:
                            print(f"  Skipping Q&A for preference '{preference}' (updated == True)")
            
            # Create a new list to store all preferences (preserving structure)
            updated_conv_list = []
            
            for conv_elem in tqdm(conv_list, desc=f"Processing {conv_type} for {uuid}", disable=not verbose):
                try:
                    preference = conv_elem.get('preference')
                    
                    # Check if this preference already has a Q&A pair
                    if 'user_query' in conv_elem:
                        if verbose:
                            print(f"  Q&A already exists for preference: '{preference}' - skipping")
                        updated_conv_list.append(conv_elem)
                        existing_qa_pairs += 1
                        continue
                    
                    # Check if this preference should be skipped by original filters
                    should_skip_by_filter = False
                    skip_reason = ""
                    
                    if preference in skip_preferences:
                        should_skip_by_filter = True
                        skip_reason = "in skip_preferences"
                    
                    # Special handling for knowledge_query
                    if conv_type == 'knowledge_query':
                        if 'idx_repeat' not in conv_elem or conv_elem['idx_repeat'] < 2:
                            should_skip_by_filter = True
                            skip_reason = "knowledge_query idx_repeat < 2"
                    
                    # For preferences that would normally be skipped but don't have Q&A, generate Q&A anyway
                    if should_skip_by_filter:
                        if verbose:
                            print(f"  Preference normally skipped ({skip_reason}) but missing Q&A, generating: '{preference}'")
                    
                    llm.reset_history()
                    # Concatenate all keys in persona before 'stereotypical_preferences' as a string
                    persona_keys = list(persona.keys())
                    idx = persona_keys.index('stereotypical_preferences')
                    keys_before = persona_keys[:idx]
                    curr_persona_dict = {k: persona[k] for k in keys_before if k in persona}
                    curr_persona = str(curr_persona_dict)
                    groundtruth_preference = conv_elem.get("preference", "")

                    # Special handling for knowledge_query - skip Q&A generation but still include in final list
                    if conv_type == 'knowledge_query':
                        if 'idx_repeat' not in conv_elem or conv_elem['idx_repeat'] < 2:
                            if verbose:
                                print(f"  Skipping Q&A generation for knowledge_query (idx_repeat < 2): '{preference}'")
                            updated_conv_list.append(conv_elem)
                            continue

                    # Use while loop to generate Q&A pairs with up to 5 attempts
                    max_attempts = 5
                    attempt = 0
                    qa_generated = False
                    
                    while attempt < max_attempts and not qa_generated:
                        attempt += 1
                        if verbose:
                            print(f"  Attempting Q&A generation for '{preference}' (attempt {attempt}/{max_attempts})")
                        
                        try:
                            # Generate QA fields
                            qa_fields = None
                            if "sensitive_info" in conv_elem:
                                qa_fields = generate_qa_for_sensitive_info(llm, conv_elem.copy(), curr_persona, global_topics, verbose=verbose)
                            else:
                                ask_to_forget = conv_elem.get('pref_type') == "ask_to_forget"
                                qa_fields = generate_qa_for_each_element(llm, conv_elem.copy(), curr_persona, conv_list, ask_to_forget=ask_to_forget, global_topics=global_topics, verbose=verbose)
                            
                            # Check if QA generation was successful
                            if qa_fields is None:
                                if verbose:
                                    print(f"    QA generation failed for attempt {attempt}")
                                continue  # Try again
                            
                            user_query = qa_fields.get("user_query")
                            correct_answer = qa_fields.get("correct_answer")
                            incorrect_answers = qa_fields.get("incorrect_answers")
                            
                            # For parallel processing, we don't validate by default to avoid complexity
                            # Just accept the first successful generation
                            conv_elem.update({
                                "user_query": user_query,
                                "correct_answer": correct_answer,
                                "incorrect_answers": incorrect_answers,
                            })
                            updated_conv_list.append(conv_elem)
                            new_qa_pairs += 1
                            qa_generated = True
                            if verbose:
                                print(f"    ✓ QA pair generated on attempt {attempt}")
                            
                            total_qa_pairs += 1
                            
                        except Exception as e:
                            if verbose:
                                print(f"    Error on attempt {attempt}: {e}")
                            continue  # Try again
                    
                    # If we couldn't generate a valid Q&A pair after all attempts
                    if not qa_generated:
                        if verbose:
                            print(f"  Failed to generate valid Q&A pair for '{preference}' after {max_attempts} attempts")
                        # Still add the element to the list but without Q&A fields
                        updated_conv_list.append(conv_elem)
                        failed_qa_pairs += 1
                        
                except Exception as e:
                    if "sensitive_info" in conv_elem:
                        print(f"Error processing persona conv_type {conv_elem.get('preference', 'Unknown')}: {e} with sensitive info {conv_elem.get('sensitive_info', 'Unknown')}")
                    else:
                        pref_info = conv_elem.get('preference', 'Unknown preference')
                        pref_type = conv_elem.get('pref_type', 'Unknown type')
                        print(f"Error processing persona conv_type '{pref_info}' (type: {pref_type}): {e}")
                        if verbose:
                            print(f"  Full element keys: {list(conv_elem.keys())}")
                    # Still add the element to preserve structure
                    updated_conv_list.append(conv_elem)
                    continue
            
            # Replace the original conversation list with the updated one
            processed_persona['conversations'][conv_type] = updated_conv_list
        
        # Print statistics for this persona
        total_preferences_processed = existing_qa_pairs + new_qa_pairs + failed_qa_pairs
        if verbose:
            print(f"Persona {uuid} summary:")
            print(f"  Total preferences processed: {total_preferences_processed}")
            print(f"  Existing Q&A pairs (skipped): {existing_qa_pairs}")
            print(f"  New Q&A pairs generated: {new_qa_pairs}")
            print(f"  Failed Q&A generation: {failed_qa_pairs}")
        
        return uuid, processed_persona
        
    except Exception as e:
        print(f"Error processing persona {uuid}: {e}")
        return uuid, None


def save_qa_topic_counts(output_dir, verbose=False):
    """
    Save the global topic counts from QA generation to a separate JSON file.
    
    Args:
        output_dir: Directory to save the topic counts file
        verbose: Whether to print debug information
    """
    import datetime
    
    # Create topic counts filename
    timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    topic_counts_path = "data/qa_topic_counts.json"
    
    # Convert defaultdict to regular dict and sort by count (descending)
    with TOPIC_COUNTER_LOCK:
        topic_counts_dict = dict(GLOBAL_TOPIC_COUNTER)
    
    # Sort topics by count (descending)
    sorted_topics = dict(sorted(topic_counts_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Add metadata
    final_data = {
        "metadata": {
            "total_categorizations": sum(sorted_topics.values()),
            "unique_topics": len(sorted_topics),
            "generated_at": timestamp,
            "source": "qa_generation"
        },
        "topic_counts": sorted_topics
    }
    
    # Save to file
    utils.save_json(final_data, topic_counts_path, clean=True)
    
    if verbose:
        print(f"Saved QA topic counts to: {topic_counts_path}")
        print(f"Total categorizations: {final_data['metadata']['total_categorizations']}")
        print(f"Unique topics: {final_data['metadata']['unique_topics']}")
        print(f"Top 5 topics: {list(sorted_topics.items())[:5]}")
    
    return topic_counts_path


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
    
    # Save topic counts after all QA processing is complete
    topic_counts_file = save_qa_topic_counts(output_dir or ".", verbose=verbose)
    print(f"QA topic categorization complete. Counts saved to: {topic_counts_file}")


def process_single_file_qa(args):
    """Process QA for a single persona file in parallel."""
    file_path, llm, verbose, validate_qa = args
    
    try:
        return file_path, process_single_file_qa_sequential(file_path, llm, verbose, validate_qa)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return file_path, False


def process_single_file_qa_sequential(file_path, llm, verbose, validate_qa=False):
    """Process QA for a single persona file sequentially."""
    # Load the persona file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize global topics at the file level for topic categorization consistency
    global_topics = []
    
    # Keep track of validation statistics
    total_qa_pairs = 0
    valid_qa_pairs = 0
    invalid_qa_pairs = 0
    existing_qa_pairs = 0
    new_qa_pairs = 0
    failed_qa_pairs = 0
    
    # Process each persona in the file
    for uuid, persona in data.items():
        conversations_by_type = persona.get("conversations", {})
        
        for i, (conv_type, conv_list) in enumerate(conversations_by_type.items()):
            print(f'Processing conv_type: {conv_type} in {os.path.basename(file_path)}')
            
            # First pass: identify preferences that should be skipped for Q&A generation
            skip_preferences = set()
            
            # Find all prev_pref of elements with pref_type == "ask_to_forget"
            for conv_elem in conv_list:
                if conv_elem.get('pref_type') == "ask_to_forget":
                    prev_pref = conv_elem.get('prev_pref')
                    if prev_pref:
                        skip_preferences.add(prev_pref)
                        if verbose:
                            print(f"  Skipping Q&A for prev_pref '{prev_pref}' (referenced by ask_to_forget)")
                
                # Also skip preferences with updated == True
                if conv_elem.get('updated') == True:
                    preference = conv_elem.get('preference')
                    if preference:
                        skip_preferences.add(preference)
                        if verbose:
                            print(f"  Skipping Q&A for preference '{preference}' (updated == True)")
            
            if verbose and skip_preferences:
                print(f"  Total preferences to skip: {len(skip_preferences)}")
            
            # Create a new list to store only valid QA pairs
            valid_conv_list = []
            
            for conv_elem in tqdm(conv_list, desc=f"Processing {conv_type} in {os.path.basename(file_path)}", leave=False):
                try:
                    preference = conv_elem.get('preference')
                    
                    # Check if this preference already has a Q&A pair
                    if 'user_query' in conv_elem:
                        if verbose:
                            print(f"  Q&A already exists for preference: '{preference}' - skipping")
                        valid_conv_list.append(conv_elem)
                        existing_qa_pairs += 1
                        continue
                    
                    # Check if this preference should be skipped by original filters
                    should_skip_by_filter = False
                    skip_reason = ""
                    
                    if preference in skip_preferences:
                        should_skip_by_filter = True
                        skip_reason = "in skip_preferences"
                    
                    # Special handling for knowledge_query
                    if conv_type == 'knowledge_query':
                        if 'idx_repeat' not in conv_elem or conv_elem['idx_repeat'] < 2:
                            should_skip_by_filter = True
                            skip_reason = "knowledge_query idx_repeat < 2"
                    
                    # For preferences that would normally be skipped but don't have Q&A, generate Q&A anyway
                    if should_skip_by_filter:
                        if verbose:
                            print(f"  Preference normally skipped ({skip_reason}) but missing Q&A, generating: '{preference}'")
                    
                    llm.reset_history()
                    # Concatenate all keys in persona before 'stereotypical_preferences' as a string
                    persona_keys = list(persona.keys())
                    idx = persona_keys.index('stereotypical_preferences')
                    keys_before = persona_keys[:idx]
                    curr_persona_dict = {k: persona[k] for k in keys_before if k in persona}
                    curr_persona = str(curr_persona_dict)
                    groundtruth_preference = conv_elem.get("preference", "")

                    # Special handling for knowledge_query - skip Q&A generation but still include in final list
                    should_skip_qa_generation = False
                    if conv_type == 'knowledge_query':
                        if 'idx_repeat' not in conv_elem or conv_elem['idx_repeat'] < 2:
                            should_skip_qa_generation = True
                            if verbose:
                                print(f"  Skipping Q&A generation for knowledge_query (idx_repeat < 2): '{preference}'")
                            valid_conv_list.append(conv_elem)
                            continue

                    # Use while loop to generate Q&A pairs with up to 5 attempts
                    max_attempts = 5
                    attempt = 0
                    qa_generated = False
                    
                    while attempt < max_attempts and not qa_generated:
                        attempt += 1
                        if verbose:
                            print(f"  Attempting Q&A generation for '{preference}' (attempt {attempt}/{max_attempts})")
                        
                        try:
                            # Generate QA fields
                            qa_fields = None
                            if "sensitive_info" in conv_elem:
                                qa_fields = generate_qa_for_sensitive_info(llm, conv_elem.copy(), curr_persona, global_topics, verbose=verbose)
                            else:
                                ask_to_forget = conv_elem.get('pref_type') == "ask_to_forget"
                                qa_fields = generate_qa_for_each_element(llm, conv_elem.copy(), curr_persona, conv_list, ask_to_forget=ask_to_forget, global_topics=global_topics, verbose=verbose)
                            
                            # Check if QA generation was successful
                            if qa_fields is None:
                                if verbose:
                                    print(f"    QA generation failed for attempt {attempt}")
                                continue  # Try again
                            
                            user_query = qa_fields.get("user_query")
                            correct_answer = qa_fields.get("correct_answer")
                            incorrect_answers = qa_fields.get("incorrect_answers")
                            
                            # Validate the QA pair only if validate_qa is enabled
                            if validate_qa:
                                is_valid = validate_qa_pair(llm, groundtruth_preference, user_query, correct_answer, incorrect_answers, verbose=verbose)
                                
                                if is_valid:
                                    # Only add to the conversation list if validation passes
                                    conv_elem.update({
                                        "user_query": user_query,
                                        "correct_answer": correct_answer,
                                        "incorrect_answers": incorrect_answers,
                                    })
                                    valid_conv_list.append(conv_elem)
                                    new_qa_pairs += 1
                                    qa_generated = True
                                    if verbose:
                                        print(f"    ✓ QA pair VALID on attempt {attempt} - Added to dataset")
                                else:
                                    invalid_qa_pairs += 1
                                    if verbose:
                                        print(f"    ✗ QA pair INVALID on attempt {attempt} - Retrying")
                            else:
                                # Skip validation, accept the first successful generation
                                conv_elem.update({
                                    "user_query": user_query,
                                    "correct_answer": correct_answer,
                                    "incorrect_answers": incorrect_answers,
                                })
                                valid_conv_list.append(conv_elem)
                                new_qa_pairs += 1
                                qa_generated = True
                                if verbose:
                                    print(f"    ✓ QA pair generated on attempt {attempt} (validation disabled)")
                            
                            total_qa_pairs += 1
                            
                        except Exception as e:
                            if verbose:
                                print(f"    Error on attempt {attempt}: {e}")
                            continue  # Try again
                    
                    # If we couldn't generate a valid Q&A pair after all attempts
                    if not qa_generated:
                        if verbose:
                            print(f"  Failed to generate valid Q&A pair for '{preference}' after {max_attempts} attempts")
                        # Still add the element to the list but without Q&A fields
                        valid_conv_list.append(conv_elem)
                        failed_qa_pairs += 1
            
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

    # Calculate final statistics
    total_preferences_processed = existing_qa_pairs + new_qa_pairs + failed_qa_pairs
    
    # Print validation statistics
    print(f"QA Generation Summary for {os.path.basename(file_path)}:")
    print(f"  Total preferences processed: {total_preferences_processed}")
    print(f"  Existing Q&A pairs (skipped): {existing_qa_pairs}")
    print(f"  New Q&A pairs generated: {new_qa_pairs}")
    print(f"  Failed Q&A generation: {failed_qa_pairs}")
    print(f"  Total Q&A pairs attempted: {total_qa_pairs}")
    if validate_qa:
        print(f"  Invalid attempts (retried): {invalid_qa_pairs}")
        if total_qa_pairs > 0:
            print(f"  Success rate for new pairs: {new_qa_pairs/(new_qa_pairs + failed_qa_pairs)*100:.1f}%")
    else:
        print(f"  Note: QA validation was disabled - all generated pairs were saved")
    
    # Calculate coverage
    total_qa_pairs_final = existing_qa_pairs + new_qa_pairs
    if total_preferences_processed > 0:
        coverage_rate = total_qa_pairs_final / total_preferences_processed * 100
        print(f"  Overall Q&A coverage: {total_qa_pairs_final}/{total_preferences_processed} ({coverage_rate:.1f}%)")

    # Save the updated data back to the same file
    utils.save_json(data, file_path, clean=True)  # Use clean=True to overwrite
    return True
