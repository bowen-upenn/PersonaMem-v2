
import json
import random
from uuid import uuid4
from tqdm import tqdm
from json_repair import repair_json
import concurrent.futures
import threading
import math
import os
import re
import glob

import prompts
import utils
from qa_generator import generate_qa_for_each_element


def find_max_persona_index(output_path, clean=False):
    """
    Find the maximum existing persona index from existing files to avoid overwriting.
    
    Args:
        output_path: The base output path for persona files
        clean: If True, ignore existing files and start from 0
    
    Returns:
        int: The starting index for new personas (max_existing + 1, or 0 if clean or no files found)
    """
    if clean:
        return 0
    
    # Extract directory and base filename pattern
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    
    # Handle different file extensions
    if base_name.endswith('.json'):
        pattern = base_name.replace('.json', '_*_persona*.json')
    elif base_name.endswith('.jsonl'):
        pattern = base_name.replace('.jsonl', '_*_persona*.jsonl')
    else:
        pattern = f"{base_name}_*_persona*"
    
    # Search for existing persona files
    search_pattern = os.path.join(output_dir, pattern)
    existing_files = glob.glob(search_pattern)
    
    max_index = -1
    
    # Extract persona indices from filenames
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Look for pattern like "interactions_250721_120307_persona0.json"
        match = re.search(r'_persona(\d+)(?:\.|$)', filename)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    
    # Return next available index
    return max_index + 1 if max_index >= 0 else 0


def expand_persona_info(llm, persona_str, image_matcher=None, verbose=False):
    """
    Expand persona with demographic info and generate preferences and background.
    
    Returns:
        tuple: (expanded_persona_str, final_json)
    """
    # 1) demographic info
    prompt = prompts.expand_persona(persona_str)
    llm.reset_history()
    persona_str = llm.query_llm(prompt, use_history=True, verbose=verbose)

    # 2) stereotypical preferences
    prompt = prompts.generate_stereotypical_preferences()
    llm.query_llm(prompt, use_history=True, verbose=verbose)
    print("Done generating stereotypical preferences.")

    # 3) anti-stereotypical preferences
    prompt = prompts.generate_anti_stereotypical_preferences()
    llm.query_llm(prompt, use_history=True, verbose=verbose)
    print("Done generating anti-stereotypical preferences.")

    # 4) verify conflicts
    prompt = prompts.verify_conflicts()
    llm.query_llm(prompt, use_history=True, verbose=verbose)
    print("Done verifying conflicts in preferences.")

    # 5) additional therapy-related personal history
    prompt = prompts.generate_therapy_related_history()
    final_json_temp = llm.query_llm(prompt, use_history=True, verbose=verbose)
    print("Done generating therapy-related personal history.")

    # 6) generate sensitive private information
    prompt = prompts.generate_sensitive_information()
    final_json = llm.query_llm(prompt, use_history=True, verbose=verbose)
    print("Done generating sensitive private information.")
    if 'sorry' in final_json.lower():
        final_json = final_json_temp

    # 7) find images if image_matcher is provided that match the persona
    if image_matcher:
        matched_images = image_matcher.find_most_similar_image(final_json, top_k=6)
        matched_images = [img_path for img_path, _ in matched_images]  # Filter out low similarity images
        if verbose:
            print(f"Matched images: {matched_images}")
        print("Done finding images that match the persona.")

    # Convert final json from a string to a JSON dictionary
    final_json = utils.extract_json_from_response(final_json)

    if verbose:
        print({f"all keys in final_json": list(final_json.keys())})

    return persona_str, final_json, matched_images if image_matcher else []


def verify_preference_alignment(llm, pref, pref_key, self_verify, verbose=False):
    """
    Verify if a preference is actually aligned with the model's believed stereotypes or anti-stereotypes.
    
    Returns:
        str: 'yes' if aligned, 'no' otherwise
    """
    if not self_verify or pref_key == "therapy_background" or pref_key == "sensitive_information":
        return 'yes'
    
    # 1) Guess which persona fits the preference
    prompt_guess = prompts.guess_persona(pref, anti=(pref_key == "anti_stereotypical_pref"))
    guessed_persona = llm.query_llm(prompt_guess, use_history=True, verbose=verbose)

    # 2) Check alignment of guessed persona with actual persona
    prompt_check = prompts.check_alignment_with_population_mean(guessed_persona)
    resp_check = llm.query_llm(prompt_check, use_history=True, verbose=verbose)
    # Extract the final answer after '####Final Answer'
    alignment = utils.extract_after_token(resp_check, '####Final Answer').strip().lower()
    
    return alignment


def generate_knowledge_queries(llm, persona_str, pref, pref_key, conversations, verbose=False):
    """
    Generate repetitive user knowledge queries to the chatbot.
    """
    type = 'knowledge_query'
    repeat = random.randint(1, 6)

    for idx_repeat in range(repeat):
        llm.reset_history()
        prompt = prompts.generate_conversations(persona_str, pref, type, is_others_pref=False)   # Interests shown by repetitive knowledge queries shall always belong to the user's own interests

        conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)
        try:
            conv_turns = utils.extract_json_from_response(conv_turns)
        except json.JSONDecodeError as e:
            try:
                conv_turns = repair_json(conv_turns)
                conv_turns = utils.extract_json_from_response(conv_turns)
            except json.JSONDecodeError as e:
                print(f"Failed to parse knowledge query response: {e}")
                continue

        if conv_turns:
            element = {'preference': pref, 'pref_type': pref_key, 'who': 'self', 'idx_repeat': idx_repeat, 'conversations': conv_turns, 'updated': False}
            conversations[type].append(element)
    
    return element


def get_random_sensitive_info(sensitive_info, verbose=False):
    """
    Get a random sensitive information item.
    
    Returns:
        str or None: Random sensitive info string or None
    """
    key = random.choice(list(sensitive_info.keys()))
    value = str(sensitive_info[key])
    random_sensitive_info = f"{key}: {value}"
    if verbose:
        print("random_sensitive_info", random_sensitive_info)
    return random_sensitive_info


def generate_cross_domain_conversations(llm, persona_str, pref, pref_key, type, conversations, is_others_pref, verbose=False):
    """
    Generate cross-domain user-chatbot conversations that implicitly encode user personas and preferences.
    """
    # Find one random type from implicit_types for each pref
    if verbose:
        print(f'{utils.Colors.OKGREEN}{pref_key}: {utils.Colors.ENDC}{pref}{utils.Colors.OKGREEN} data type: {utils.Colors.ENDC}{type}')

    prompt = prompts.generate_conversations(persona_str, pref, type, is_others_pref)
    conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

    try:
        conv_turns = utils.extract_json_from_response(conv_turns)
    except json.JSONDecodeError as e:
        try:
            conv_turns = repair_json(conv_turns)
            conv_turns = utils.extract_json_from_response(conv_turns)
        except json.JSONDecodeError as e:
            print(f"Failed to parse knowledge query response: {e}")
            return

    conv_turns = utils.merge_consecutive_roles(conv_turns)
    who = 'others' if is_others_pref else 'self'

    if conv_turns:
        element = {'preference': pref, 'pref_type': pref_key, 'who': who, 'conversations': conv_turns, 'updated': False}
        conversations[type].append(element)
        return element


def generate_preference_updates(llm, persona_str, pref, pref_key, type, conversations, updates, is_others_pref, verbose=False):
    """
    Generate preference updates by changing preferences to their opposite along the timeline.
    """
    llm.reset_history()
    prompt = prompts.update_preference(pref)   # Interests shown by repetitive knowledge queries shall always belong to the user's own interests
    updated_pref = llm.query_llm(prompt, use_history=False, verbose=verbose)

    # Record the update
    updates[pref] = updated_pref
    llm.reset_history()

    prompt = prompts.generate_conversations(persona_str, updated_pref, type, is_others_pref, updated=True)
    conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

    try:
        conv_turns = utils.extract_json_from_response(conv_turns)
    except json.JSONDecodeError as e:
        try:
            conv_turns = repair_json(conv_turns)
            conv_turns = utils.extract_json_from_response(conv_turns)
        except json.JSONDecodeError as e:
            print(f"Failed to parse knowledge query response: {e}")
            return

    conv_turns = utils.merge_consecutive_roles(conv_turns)
    if conv_turns:
        conversations[type].append({'preference': updated_pref, 'pref_type': pref_key, 'who': 'self', 'conversations': conv_turns, 'updated': True, 'prev_pref': pref})


def user_ask_to_forget(llm, element, conversations, type, verbose):
    """
    Handle user requests to forget a specific preference.
    """
    llm.reset_history()

    prev_pref = element['preference']
    new_element = generate_qa_for_each_element(llm, element, verbose)
    user_query = new_element['user_query']
    correct_answer = new_element['correct_answer']

    llm.reset_history()
    prompt = prompts.user_ask_to_forget(user_query, prev_pref, correct_answer)
    user_followup = llm.query_llm(prompt, use_history=False, verbose=verbose)
    user_followup = utils.extract_after_token(user_followup, '####').strip()

    conv_turns = [{
        "role": "user",
        "content": user_query
    }, {
        "role": "assistant",
        "content": correct_answer
    }, {
        "role": "user",
        "content": user_followup
    }]

    element = {'preference': f"Do not remember '{prev_pref}' in memory", 'pref_type': "ask_to_forget", 'who': 'self', 'conversations': conv_turns, 'updated': True, 'prev_pref': prev_pref}
    conversations[type].append(element)


def find_preference_from_image_and_generate_conversations(llm, persona_str, image_path, conversations, is_others_pref, verbose=False):
    """
    Find the user's preference based on the content of the image.
    """
    type = 'multimodal'

    prompt = prompts.find_preference_from_image(persona_str, is_others_pref)
    preference = llm.query_llm(prompt, image_path=image_path, use_history=True, verbose=verbose)
    preference = utils.extract_after_token(preference, '####').strip()  # Extract the preference after the special token

    prompt = prompts.generate_conversations(persona_str, preference, type, is_others_pref)
    conv_turns = llm.query_llm(prompt, image_path=image_path, use_history=True, verbose=verbose)

    try:
        conv_turns = utils.extract_json_from_response(conv_turns)
    except json.JSONDecodeError as e:
        try:
            conv_turns = repair_json(conv_turns)
            conv_turns = utils.extract_json_from_response(conv_turns)
        except json.JSONDecodeError as e:
            print(f"Failed to parse knowledge query response: {e}")
            return None, conversations
        
    conv_turns = utils.merge_consecutive_roles(conv_turns)

    # add the image to the user query
    if image_path:
        conv_turns = utils.rewrite_user_query_to_add_image(conv_turns, image_path)

    who = 'others' if is_others_pref else 'self'

    if conv_turns:
        conversations[type].append({'preference': preference, 'pref_type': 'multimodal', 'who': who, 'conversations': conv_turns, 'updated': False, 'image_path': image_path})


def convert_preferences_to_conversations(llm, persona_str, final_json, implicit_types, self_verify, verbose=False):
    """
    Process all preferences and generate conversations for each aligned preference.
    
    Returns:
        tuple: (conversations, updates)
    """
    conversations = {'multimodal': []}
    for type in implicit_types:
        conversations[type] = []
    updates = {}

    # print(f"All keys in final_json: {list(final_json.keys())}")
    sensitive_info = final_json.get('sensitive_information', {})
    # Add conversations with sensitive information
    if sensitive_info:
        random_sensitive_info = get_random_sensitive_info(sensitive_info, verbose)
        type = random.choice(implicit_types)
        if verbose:
            print(f"Sensitive information: {random_sensitive_info} for type {type}")
        prompt = prompts.generate_conversations_sensitive_info(persona_str, random_sensitive_info, type)
        conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

        try:
            conv_turns = utils.extract_json_from_response(conv_turns)
            conv_turns = utils.merge_consecutive_roles(conv_turns)
            if conv_turns:
                conversations[type].append({'sensitive_info': random_sensitive_info, 'who': 'self', 'conversations': conv_turns})
        except json.JSONDecodeError as e:
            try:
                conv_turns = repair_json(conv_turns)
                conv_turns = utils.extract_json_from_response(conv_turns)
                conv_turns = utils.merge_consecutive_roles(conv_turns)
                if conv_turns:
                    conversations[type].append({'sensitive_info': random_sensitive_info, 'who': 'self', 'conversations': conv_turns})
            except json.JSONDecodeError as e:
                print(f"Failed to parse knowledge query response: {e}")
                sensitive_info = {}

    for pref_key, pref_list in [
        ("stereotypical_pref", final_json.get("stereotypical_preferences", [])),
        ("anti_stereotypical_pref", final_json.get("anti_stereotypical_preferences", [])),
        ("therapy_background", final_json.get("therapy_background", []))
    ]:
        for pref_idx, pref in tqdm(enumerate(pref_list), desc=f"Processing {pref_key} preferences", total=len(pref_list)):
            llm.reset_history()

            try:
                # Verify preference alignment
                alignment = verify_preference_alignment(llm, pref, pref_key, self_verify, verbose)

                # Only generate conversations for aligned preferences
                if alignment == 'yes':
                    # Set both the user's own preferences and other people's preferences mentioned by this user, for example, to test the llm
                    is_others_pref = random.random() < 0.33
                    random_type = random.choice(implicit_types)

                    # We assign around 1/3 preferences to induce knowledge-related queries, and assume repetitive queries indicate some interests
                    if random.random() < 0.33 and pref_key != "therapy_background" and 'knowledge_query' in implicit_types:
                        element = generate_knowledge_queries(llm, persona_str, pref, pref_key, conversations, verbose)
                    else:
                        # Generate cross-domain conversations in random types
                        element = generate_cross_domain_conversations(llm, persona_str, pref, pref_key, random_type, conversations, is_others_pref, verbose)

                    has_asked_to_forget = False
                    if random.random() < 0.33 and not is_others_pref and element:
                        user_ask_to_forget(llm, element, conversations, random_type, verbose)
                        has_asked_to_forget = True

                    # Generate preference updates
                    if random.random() < 0.67 and not is_others_pref and pref_key not in ["therapy_background", "knowledge_query"] and not has_asked_to_forget:
                        random_type = random.choice(implicit_types) if pref_key == "therapy_background" or 'knowledge_query' not in implicit_types else 'knowledge_query'
                        generate_preference_updates(llm, persona_str, pref, pref_key, random_type, conversations, updates, is_others_pref, verbose)

            except Exception as e:
                print(f"Error processing preference {pref_key} with value {pref}: {e}")
                continue

    image_paths = final_json.get("matched_images", [])
    for image_idx, image_path in enumerate(image_paths):
        llm.reset_history()
        is_others_pref = image_idx > 0.67 * len(image_paths)   # sorted by the order of relevance
        # Generate preference and conversations as if the user is providing an image to the chatbot
        find_preference_from_image_and_generate_conversations(llm, str(final_json), image_path, conversations, is_others_pref, verbose=verbose)

    return conversations, updates


def update_aligned_preferences(final_json, conversations, updates):
    """
    Update final_json to only include aligned preferences.
    """
    # Update final_json to only include aligned preferences
    aligned_stereo = [c['stereotypical_pref'] for convs in conversations.values() for c in convs if 'stereotypical_pref' in c]
    aligned_anti = [c['anti_stereotypical_pref'] for convs in conversations.values() for c in convs if 'anti_stereotypical_pref' in c]

    final_json['stereotypical_preferences'] = list(set(aligned_stereo))     # the repeated calls on knowledge_query may introduce repeated preferences in the previous step
    final_json['anti_stereotypical_preferences'] = list(set(aligned_anti))
    final_json['preference_updates'] = updates

    return final_json


def process_single_persona(llm, persona, implicit_types, self_verify, image_matcher, verbose=False):
    """
    Process a single persona to generate all its interactions and conversations.
    
    Returns:
        dict: Complete persona data with conversations
    """
    persona_str = json.dumps(persona, ensure_ascii=False)
    if verbose:
        print(f"Persona: {persona_str}")

    # Expand persona info and generate preferences
    persona_str, final_json_response, matched_images = expand_persona_info(llm, persona_str, image_matcher, verbose)
    try:
        final_json_response = utils.extract_json_from_response(final_json_response)
        final_json_response['matched_images'] = matched_images
    except json.JSONDecodeError as e:
        try:
            final_json_response = repair_json(final_json_response)
            final_json_response = utils.extract_json_from_response(final_json_response)
            if 'stereotypical_preferences' not in final_json_response:
                return None
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            return None

    # Process preferences and generate conversations
    conversations, updates = convert_preferences_to_conversations(llm, persona_str, final_json_response, implicit_types, self_verify, verbose)

    # Update aligned preferences
    # if updates is not an empty dict
    if updates:
        final_json_response = update_aligned_preferences(final_json_response, conversations, updates)

    # Attach conversations to the final output
    final_json_response["conversations"] = conversations
    
    return final_json_response


def process_single_persona_thread(args):
    """
    Thread-safe function to process a single persona.
    
    Args:
        args: tuple containing (idx, persona, llm, implicit_types, self_verify, image_matcher, output_path, clean, verbose)
    
    Returns:
        tuple: (idx, persona_id, final_json, full_path) or (idx, None, None, None) if failed
    """
    idx, persona, llm, implicit_types, self_verify, image_matcher, output_path, clean, verbose = args
    
    try:
        # Process single persona
        final_json = process_single_persona(llm, persona, implicit_types, self_verify, image_matcher, verbose)
        
        if final_json is not None:
            persona_id = str(uuid4())
            
            # Modify the output path to include timestamp
            base_path = output_path
            if base_path.endswith('.json'):
                full_path = base_path.replace('.json', f'_persona{idx}.json')
            elif base_path.endswith('.jsonl'):
                full_path = base_path.replace('.jsonl', f'_persona{idx}.jsonl')
            else:
                full_path = f"{base_path}_persona{idx}"
            
            return idx, persona_id, final_json, full_path
        else:
            return idx, None, None, None
            
    except Exception as e:
        print(f"Error processing persona {idx}: {e}")
        return idx, None, None, None


def generate_interactions_from_persona(llm, all_personas, image_matcher, output_path, implicit_types, num_persona=1, self_verify=True, clean=False, verbose=False):
    """
    Load personas and process them in parallel, then sequentially query the Assistant API with three prompts:
      1) Add name and demographic info in JSON for the persona
      2) Propose overly stereotypical and anti-stereotypical preferences
      3) Verify and replace any conflicts
      4) Generate cross-domain user-chatbot conversations that implicitly mention the preferences
    Save the final JSON to output_file.
    """
    output_dict = {}
    
    # Find the starting index to avoid overwriting existing personas
    persona_start_idx = find_max_persona_index(output_path, clean)
    if not clean and persona_start_idx > 0:
        print(f"Found existing persona files. Starting from persona index {persona_start_idx}")
    
    # Prepare arguments for each persona
    persona_args = []
    for i in range(num_persona):
        idx = persona_start_idx + i  # Use the adjusted index
        persona = random.choice(all_personas)
        persona_args.append((idx, persona, llm, implicit_types, self_verify, image_matcher, output_path, clean, verbose))
    
    # Process personas in parallel batches
    max_workers = min(llm.rate_limit_per_min, num_persona)
    batch_size = max_workers
    num_batches = math.ceil(num_persona / batch_size)
    
    for batch_idx in range(num_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min((batch_idx + 1) * batch_size, num_persona)
        batch_args = persona_args[batch_start_idx:batch_end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_args)} personas)")
        
        # Process batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
            # Submit all tasks for this batch
            future_to_args = {executor.submit(process_single_persona_thread, args): args for args in batch_args}
            
            # Collect results with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                             desc=f"Batch {batch_idx + 1} personas", 
                             total=len(batch_args)):
                try:
                    idx, persona_id, final_json, full_path = future.result()
                    
                    if final_json is not None:
                        # Add to output dict
                        output_dict[persona_id] = final_json
                        
                        # Save individual file - only clean on first file if clean=True
                        should_clean = clean and idx == persona_start_idx
                        utils.save_json({persona_id: final_json}, full_path, clean=should_clean)
                        print(f"Saved persona {idx} to {full_path}")
                    
                except Exception as e:
                    args = future_to_args[future]
                    idx = args[0]
                    print(f"Error in future for persona {idx}: {e}")
    
    return output_dict
