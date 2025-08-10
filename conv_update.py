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
    
    # Look for existing persona files in data/raw_data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, "data", "raw_data")
    
    # Extract base name from timestamped path
    base_name = os.path.basename(output_path)
    
    # Remove timestamp and extension to get the clean base name
    timestamp_pattern = r'_\d{6}_\d{6}'
    clean_base_name = re.sub(timestamp_pattern, '', base_name)
    if clean_base_name.endswith('.json'):
        clean_base_name = clean_base_name[:-5]  # Remove .json
    
    # Search for existing persona files in raw_data directory with pattern: {clean_base_name}_*_persona*.json
    search_pattern = os.path.join(raw_data_dir, f"{clean_base_name}*_persona*.json")
    existing_files = glob.glob(search_pattern)
    
    max_index = -1
    
    # Extract persona indices from filenames
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Look for pattern like "interactions_250727_201452_persona607.json"
        match = re.search(r'_persona(\d+)\.json$', filename)
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


def extend_to_multiturns(llm, conv_turns, verbose):
    prompt = prompts.extend_to_multiturns(str(conv_turns))
    conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

    try:
        conv_turns = utils.extract_json_from_response(conv_turns)
    except json.JSONDecodeError as e:
        try:
            conv_turns = repair_json(conv_turns)
            conv_turns = utils.extract_json_from_response(conv_turns)
        except json.JSONDecodeError as e:
            print(f"Failed to parse multi-turn response: {e}")
            return
    return conv_turns


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

        if random.random() < 0.33:
            conv_turns_temp = extend_to_multiturns(llm, conv_turns, verbose)
            if conv_turns_temp:
                conv_turns = conv_turns_temp

        conv_turns = utils.merge_consecutive_roles(conv_turns)

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

    if random.random() < 0.33:
        conv_turns_temp = extend_to_multiturns(llm, conv_turns, verbose)
        if conv_turns_temp:
            conv_turns = conv_turns_temp

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

    if random.random() < 0.33:
        conv_turns_temp = extend_to_multiturns(llm, conv_turns, verbose)
        if conv_turns_temp:
            conv_turns = conv_turns_temp

    conv_turns = utils.merge_consecutive_roles(conv_turns)
    if conv_turns:
        conversations[type].append({'preference': updated_pref, 'pref_type': pref_key, 'who': 'self', 'conversations': conv_turns, 'updated': True, 'prev_pref': pref})


def user_ask_to_forget(llm, element, conversations, type, verbose):
    """
    Handle user requests to forget a specific preference.
    """
    llm.reset_history()

    prev_pref = element['preference']
    new_element = generate_qa_for_each_element(llm, element.copy(), verbose)
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
            return None  # Return None for element, conversations is modified in place

    if random.random() < 0.33:
        conv_turns_temp = extend_to_multiturns(llm, conv_turns, verbose)
        if conv_turns_temp:
            conv_turns = conv_turns_temp
        
    conv_turns = utils.merge_consecutive_roles(conv_turns)

    # add the image to the user query
    if image_path:
        conv_turns = utils.rewrite_user_query_to_add_image(conv_turns, image_path)

    who = 'others' if is_others_pref else 'self'

    if conv_turns:
        element = {'preference': preference, 'pref_type': 'multimodal', 'who': who, 'conversations': conv_turns, 'updated': False, 'image_path': image_path}
        conversations[type].append(element)
        return element
    

def convert_preferences_to_conversations(llm, persona_str, final_json, implicit_types, self_verify, file_path, verbose=False):
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
        persona_id = re.search(r'persona\d+', file_path).group()
        print(f"Processing {pref_key} preferences for {persona_id}")
        for pref_idx, pref in tqdm(enumerate(pref_list), total=len(pref_list)):
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
        # try:
        element = find_preference_from_image_and_generate_conversations(llm, str(final_json), image_path, conversations, is_others_pref, verbose=verbose)
        # except Exception as e:
        #     continue
        
    return conversations, updates


def update_aligned_preferences(final_json, conversations, updates):
    """
    Update final_json to only include aligned preferences.
    """
    # Update final_json to only include aligned preferences
    aligned_stereo = [c['preference'] for convs in conversations.values() for c in convs if c.get('pref_type') == 'stereotypical_pref']
    aligned_anti = [c['preference'] for convs in conversations.values() for c in convs if c.get('pref_type') == 'anti_stereotypical_pref']

    final_json['stereotypical_preferences'] = list(set(aligned_stereo))     # the repeated calls on knowledge_query may introduce repeated preferences in the previous step
    final_json['anti_stereotypical_preferences'] = list(set(aligned_anti))
    final_json['preference_updates'] = updates

    return final_json


def process_single_persona(llm, persona, implicit_types, self_verify, image_matcher, file_path, verbose=False):
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
    conversations, updates = convert_preferences_to_conversations(llm, persona_str, final_json_response, implicit_types, self_verify, file_path, verbose)

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
    idx, persona, llm, implicit_types, self_verify, image_matcher, output_path, clean, file_path, verbose = args
    
    # try:
    # Process single persona
    final_json = process_single_persona(llm, persona, implicit_types, self_verify, image_matcher, file_path, verbose)
    
    if final_json is not None:
        persona_id = str(uuid4())
        
        # Create individual persona file path
        output_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        
        # Keep timestamp but remove extension from base name
        if base_name.endswith('.json'):
            base_name_no_ext = base_name[:-5]  # Remove .json
        else:
            base_name_no_ext = base_name
        
        full_path = os.path.join(output_dir, f"{base_name_no_ext}_persona{idx}.json")
        
        return idx, persona_id, final_json, full_path
    else:
        return idx, None, None, None
            
    # except Exception as e:
    #     print(f"Error processing persona {idx}: {e}")
    #     return idx, None, None, None


def generate_interactions_from_persona(llm, all_personas, image_matcher, output_path, implicit_types, num_persona=1, self_verify=True, clean=False, parallel=False, verbose=False):
    """
    Load personas and process them in parallel or sequentially, then query the Assistant API with three prompts:
      1) Add name and demographic info in JSON for the persona
      2) Propose overly stereotypical and anti-stereotypical preferences
      3) Verify and replace any conflicts
      4) Generate cross-domain user-chatbot conversations that implicitly mention the preferences
    Save the final JSON to output_file.
    """
    output_dict = {}
    
    # Find the starting index to avoid overwriting existing personas
    persona_start_idx = find_max_persona_index(output_path, clean)
    if persona_start_idx > 0:
        print(f"Found existing persona files. Starting from persona index {persona_start_idx}.")
    else:
        print(f"Starting from persona index 0.")
    
    if parallel:
        # Parallel processing
        # Prepare arguments for each persona
        persona_args = []
        for i in range(num_persona):
            idx = persona_start_idx + i  # Use the adjusted index
            persona = random.choice(all_personas)
            persona_args.append((idx, persona, llm, implicit_types, self_verify, image_matcher, output_path, clean, file_path, verbose))
        
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
                    # try:
                    idx, persona_id, final_json, full_path = future.result()
                    
                    if final_json is not None:
                        # Add to output dict
                        output_dict[persona_id] = final_json
                        
                        # Save individual file - only clean on first file if clean=True
                        should_clean = clean and idx == persona_start_idx
                        utils.save_json({persona_id: final_json}, full_path, clean=should_clean)
                        print(f"Saved persona {idx} to {full_path}")
                        
                    # except Exception as e:
                    #     args = future_to_args[future]
                    #     idx = args[0]
                    #     print(f"Error in future for persona {idx}: {e}")
    else:
        # Sequential processing
        for i in tqdm(range(num_persona), desc="Processing personas sequentially"):
            idx = persona_start_idx + i  # Use the adjusted index
            persona = random.choice(all_personas)
            
            try:
                # Process single persona
                final_json = process_single_persona(llm, persona, implicit_types, self_verify, image_matcher, verbose)
                
                if final_json is not None:
                    persona_id = str(uuid4())
                    
                    # Create individual persona file path
                    output_dir = os.path.dirname(output_path)
                    base_name = os.path.basename(output_path)
                    
                    # Keep timestamp but remove extension from base name
                    if base_name.endswith('.json'):
                        base_name_no_ext = base_name[:-5]  # Remove .json
                    else:
                        base_name_no_ext = base_name
                    
                    full_path = os.path.join(output_dir, f"{base_name_no_ext}_persona{idx}.json")
                    
                    # Add to output dict
                    output_dict[persona_id] = final_json
                    
                    # Save individual file - only clean on first file if clean=True
                    should_clean = clean and idx == persona_start_idx
                    utils.save_json({persona_id: final_json}, full_path, clean=should_clean)
                    print(f"Saved persona {idx} to {full_path}")
                    
            except Exception as e:
                print(f"Error processing persona {idx}: {e}")
    
    return output_dict


def load_persona_data_from_file(file_path, llm=None, persona_keys_to_add=None, verbose=False):
    """
    Load persona data from a JSON file and separate persona_str and final_json_response.
    Optionally add new persona keys (e.g., health_and_medical_conditions) by querying LLM.

    Args:
        file_path: Path to the persona JSON file
        llm: QueryLLM instance (required if persona_keys_to_add is not None)
        persona_keys_to_add: list of keys to add (currently supports ["health_and_medical_conditions"])
        verbose: verbosity
    Returns:
        tuple: (persona_str, final_json_response, persona_id, conversations) or (None, None, None, None) if failed
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        persona_id = list(data.keys())[0]
        persona_data = data[persona_id]

        # Build persona_str (content before stereotypical_preferences as before)
        persona_dict = {}
        for key, value in persona_data.items():
            if key == 'stereotypical_preferences':
                break
            persona_dict[key] = value
        persona_str = json.dumps(persona_dict, ensure_ascii=False)

        # Build final_json_response (content before conversations)
        final_json_response = {}
        for key, value in persona_data.items():
            if key == 'conversations':
                break
            final_json_response[key] = value

        conversations = persona_data.get('conversations', {})

        # Add new persona keys if requested
        if persona_keys_to_add and llm is not None:
            for key_to_add in persona_keys_to_add:
                if key_to_add == 'health_and_medical_conditions':
                    # Construct prompt by prepending existing persona_str
                    llm.reset_history()
                    prompt = persona_str + "\n" + prompts.generate_health_and_medical_conditions()
                    try:
                        resp = llm.query_llm(prompt, use_history=False, verbose=verbose)
                        resp_json = utils.extract_json_from_response(resp)
                        # Merge only the new key if present
                        if 'health_and_medical_conditions' in resp_json:
                            final_json_response['health_and_medical_conditions'] = resp_json['health_and_medical_conditions']
                            if verbose:
                                print(f"Added health_and_medical_conditions to {file_path}")
                    except Exception as e:
                        if verbose:
                            print(f"Failed adding {key_to_add} for {file_path}: {e}")

        return persona_str, final_json_response, persona_id, conversations
    except Exception as e:
        print(f"Error loading persona data from {file_path}: {e}")
        return None, None, None, None


def regenerate_conversations_for_data_types(llm, persona_str, final_json_response, data_types_to_update, conversations, image_matcher, self_verify, file_path, persona_keys_to_add=None, verbose=False):
    """
    Regenerate conversations for specific data types while preserving all other data.
    
    Args:
        llm: Language model instance
        persona_str: String representation of the persona
        final_json_response: Persona data before conversations
        data_types_to_update: List of data types to regenerate
        conversations: Existing conversations dict
        image_matcher: Image matcher instance
        self_verify: Whether to verify preferences
        verbose: Whether to print verbose output
    
    Returns:
        dict: Updated conversations
    """
    if not persona_keys_to_add:
        # Clear conversations for data types we're updating
        for data_type in data_types_to_update:
            if data_type in conversations:
                conversations[data_type] = []
            else:
                conversations[data_type] = []
    
    # Process preferences and generate new conversations only for specified data types
    new_conversations, updates = convert_preferences_to_conversations_selective(
        llm, persona_str, final_json_response, data_types_to_update, conversations, self_verify, file_path, persona_keys_to_add, verbose
    )
    
    # Update only the specified data types
    for data_type in data_types_to_update:
        if data_type in new_conversations:
            conversations[data_type] = new_conversations[data_type]
    
    return conversations, updates


def collect_existing_preferences_from_conversations(conversations, data_types_to_update):
    """
    Collect all preferences that are already used in existing conversation types.
    Only looks at conversation types that are NOT being updated.
    
    Args:
        conversations: Existing conversations dict
        data_types_to_update: List of data types being updated (preferences from these will be ignored)
    
    Returns:
        set: Set of preference strings already used in existing conversation types
    """
    existing_preferences = set()
    
    for conversation_type, conversation_list in conversations.items():
        # Skip multimodal and types being updated
        if conversation_type == 'multimodal' or conversation_type in data_types_to_update:
            continue
            
        # Collect preferences from this conversation type
        for conversation_obj in conversation_list:
            if 'preference' in conversation_obj:
                existing_preferences.add(conversation_obj['preference'])
                
            # Also check for preferences in 'prev_pref' field (for updated preferences)
            if 'prev_pref' in conversation_obj:
                existing_preferences.add(conversation_obj['prev_pref'])
    
    return existing_preferences


def convert_preferences_to_conversations_selective(llm, persona_str, final_json, data_types_to_update, existing_conversations, self_verify, file_path, persona_keys_to_add=None, verbose=False):
    """
    Process preferences and generate conversations only for specified data types.
    Modified version of convert_preferences_to_conversations that only generates specific data types.
    Only works on preferences not already covered in existing conversation types.
    
    Args:
        llm: Language model instance
        persona_str: String representation of the persona
        final_json: Persona data
        data_types_to_update: List of data types to generate
        existing_conversations: Existing conversations dict to check for already used preferences
        self_verify: Whether to verify preferences
        verbose: Whether to print verbose output
    
    Returns:
        tuple: (conversations, updates)
    """
    conversations = {}
    for type in data_types_to_update:
        conversations[type] = []
    updates = {}

    # Collect preferences that are already used in existing conversation types (excluding those being updated)
    existing_preferences = collect_existing_preferences_from_conversations(existing_conversations, data_types_to_update)
    
    if verbose:
        print(f"Found {len(existing_preferences)} existing preferences to avoid: {list(existing_preferences)[:5]}..." if existing_preferences else "No existing preferences found to avoid")

    sensitive_info = final_json.get('sensitive_information', {})
    # Add conversations with sensitive information only if relevant data type is requested
    if sensitive_info and any(dt in data_types_to_update for dt in data_types_to_update):
        random_sensitive_info = get_random_sensitive_info(sensitive_info, verbose)
        # Only use data types that are being updated
        available_types = [dt for dt in data_types_to_update if dt != 'multimodal']
        if available_types:
            type = random.choice(available_types)
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

    if persona_keys_to_add is None:
        persona_features = [
            ("stereotypical_pref", final_json.get("stereotypical_preferences", [])),
            ("anti_stereotypical_pref", final_json.get("anti_stereotypical_preferences", [])),
            ("therapy_background", final_json.get("therapy_background", []))
        ]
    else:
        if isinstance(persona_keys_to_add, str):
            persona_keys_to_add = [persona_keys_to_add]
        persona_features = []
        for key in persona_keys_to_add:
            persona_features.append((key, final_json.get(key, [])))

    for pref_key, pref_list in persona_features:
        # Extract persona id from the full path
        persona_id = re.search(r'persona\d+', file_path).group()
        for pref_idx, pref in tqdm(enumerate(pref_list), desc=f"Processing {pref_key} for {persona_id}", total=len(pref_list)):
            llm.reset_history()
            try:
                # Skip preferences that are already used in existing conversation types
                if pref in existing_preferences:
                    if verbose:
                        print(f"Skipping preference already used: {pref[:50]}...")
                    continue
                
                # Verify preference alignment
                alignment = verify_preference_alignment(llm, pref, pref_key, self_verify, verbose)

                # Only generate conversations for aligned preferences
                if alignment == 'yes':
                    # Generate conversations only for data types we're updating
                    available_types = [dt for dt in data_types_to_update if dt != 'multimodal']
                    if available_types:
                        # Randomly choose whether this is others' preference
                        is_others_pref = random.random() < 0.33
                        
                        # Each preference is randomly assigned to ONLY ONE of the available types
                        selected_type = random.choice(available_types)
                        
                        if random.random() < 0.33 and pref_key != "therapy_background" and 'knowledge_query' in available_types:
                            # For knowledge queries, we might override the selected_type
                            selected_type = 'knowledge_query'
                            element = generate_knowledge_queries(llm, persona_str, pref, pref_key, conversations, verbose)
                        else:
                            # Generate conversations for the selected data type only
                            element = generate_cross_domain_conversations(llm, persona_str, pref, pref_key, selected_type, conversations, is_others_pref, verbose)

                        has_asked_to_forget = False
                        if random.random() < 0.33 and not is_others_pref and element and available_types:
                            # Use the same selected_type for consistency
                            user_ask_to_forget(llm, element, conversations, selected_type, verbose)
                            has_asked_to_forget = True

                        # Generate preference updates
                        if random.random() < 0.67 and not is_others_pref and pref_key not in ["therapy_background", "knowledge_query"] and not has_asked_to_forget and available_types:
                            # Use the same selected_type for consistency
                            generate_preference_updates(llm, persona_str, pref, pref_key, selected_type, conversations, updates, is_others_pref, verbose)

            except Exception as e:
                print(f"Error processing preference {pref_key} with value {pref}: {e}")
                continue

    # Handle multimodal if requested
    if 'multimodal' in data_types_to_update:
        conversations['multimodal'] = []
        image_paths = final_json.get("matched_images", [])
        for image_idx, image_path in enumerate(image_paths):
            llm.reset_history()
            is_others_pref = image_idx > 0.67 * len(image_paths)   # sorted by the order of relevance
            # Generate preference and conversations as if the user is providing an image to the chatbot
            element = find_preference_from_image_and_generate_conversations(llm, persona_str, image_path, conversations, is_others_pref, verbose=verbose)
        
    return conversations, updates


def update_single_persona_file(llm, file_path, data_types_to_update, persona_keys_to_add, image_matcher, self_verify, verbose=False):
    """
    Update conversations for specific data types in a single persona file, or add new persona keys.
    
    Args:
        llm: Language model instance
        file_path: Path to the persona JSON file
        data_types_to_update: List of data types to regenerate
        persona_keys_to_add: list of persona keys to add (e.g., ["health_and_medical_conditions"])
        image_matcher: Image matcher instance
        self_verify: Whether to verify preferences
        verbose: Whether to print verbose output
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Load existing persona data
    persona_str, final_json_response, persona_id, conversations = load_persona_data_from_file(file_path, llm, persona_keys_to_add, verbose)
    
    if persona_str is None:
        print(f"Failed to load persona data from {file_path}")
        return False
    
    # If persona_keys_to_add is given, just add those keys and skip conversation regeneration
    if persona_keys_to_add:
        if verbose:
            print(f"Adding persona keys {persona_keys_to_add} to {file_path}")
        
        # The keys were already added by load_persona_data_from_file, so we just need to save
        # Ensure conversations are preserved at the end
        final_json_response['conversations'] = conversations
        
        # Save the updated data back to the file
        try:
            utils.save_json({persona_id: final_json_response}, file_path, clean=False)
            if verbose:
                print(f"Successfully added persona keys to {file_path}")
        except Exception as e:
            print(f"Error saving updated data to {file_path}: {e}")
    
    # Otherwise, regenerate conversations for specified data types
    if verbose:
        print(f"Updating conversations for data types {data_types_to_update} in {file_path}")
    
    # Regenerate conversations for specified data types
    updated_conversations, updates = regenerate_conversations_for_data_types(
        llm, persona_str, final_json_response, data_types_to_update, conversations, image_matcher, self_verify, file_path, persona_keys_to_add, verbose
    )
    
    # Update the final_json_response with new conversations
    final_json_response['conversations'] = updated_conversations
    
    # If there were preference updates, add them
    if updates:
        final_json_response['preference_updates'] = updates
    
    # Save the updated data back to the file
    try:
        utils.save_json({persona_id: final_json_response}, file_path, clean=False)
        if verbose:
            print(f"Successfully updated {file_path}")
        return True
    except Exception as e:
        print(f"Error saving updated data to {file_path}: {e}")
        return False


def update_single_persona_file_thread(args):
    """
    Thread-safe function to update a single persona file.
    
    Args:
        args: tuple containing (file_path, llm, data_types_to_update, persona_keys_to_add, image_matcher, self_verify, verbose)
    
    Returns:
        tuple: (file_path, success)
    """
    file_path, llm, data_types_to_update, persona_keys_to_add, image_matcher, self_verify, verbose = args
    
    try:
        success = update_single_persona_file(llm, file_path, data_types_to_update, persona_keys_to_add, image_matcher, self_verify, verbose)
        return file_path, success
    except Exception as e:
        print(f"Error updating persona file {file_path}: {e}")
        return file_path, False


def update_conversations_for_data_types(llm, persona_files, data_types_to_update, image_matcher, self_verify, persona_keys_to_add=None, parallel=False, verbose=False):
    """
    Update conversations for specific data types across multiple persona files.
    
    Args:
        llm: Language model instance
        persona_files: List of persona file paths
        data_types_to_update: List of data types to regenerate
        image_matcher: Image matcher instance
        self_verify: Whether to verify preferences
        persona_keys_to_add: list of persona keys to add (e.g., ["health_and_medical_conditions"])
        parallel: Whether to process files in parallel
        verbose: Whether to print verbose output
    """
    print(f"Updating conversations for data types: {data_types_to_update}")
    print(f"Processing {len(persona_files)} persona files")
    
    if parallel:
        # Parallel processing
        file_args = []
        for file_path in persona_files:
            file_args.append((file_path, llm, data_types_to_update, persona_keys_to_add, image_matcher, self_verify, verbose))
        
        # Process files in parallel batches
        max_workers = min(llm.rate_limit_per_min, len(persona_files))
        batch_size = max_workers
        num_batches = math.ceil(len(persona_files) / batch_size)
        
        successful_updates = 0
        failed_updates = 0
        
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, len(persona_files))
            batch_args = file_args[batch_start_idx:batch_end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_args)} files)")
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
                # Submit all tasks for this batch
                future_to_args = {executor.submit(update_single_persona_file_thread, args): args for args in batch_args}
                
                # Collect results with progress bar
                for future in tqdm(concurrent.futures.as_completed(future_to_args), 
                                 desc=f"Batch {batch_idx + 1} files", 
                                 total=len(batch_args)):
                    try:
                        file_path, success = future.result()
                        if success:
                            successful_updates += 1
                            if verbose:
                                print(f"Successfully updated {file_path}")
                        else:
                            failed_updates += 1
                            print(f"Failed to update {file_path}")
                    except Exception as e:
                        args = future_to_args[future]
                        file_path = args[0]
                        failed_updates += 1
                        print(f"Error in future for file {file_path}: {e}")
    else:
        # Sequential processing
        successful_updates = 0
        failed_updates = 0
        
        for file_path in tqdm(persona_files, desc="Updating persona files"):
            try:
                success = update_single_persona_file(llm, file_path, data_types_to_update, persona_keys_to_add, image_matcher, self_verify, verbose)
                if success:
                    successful_updates += 1
                else:
                    failed_updates += 1
            except Exception as e:
                print(f"Error updating {file_path}: {e}")
                failed_updates += 1
    
    print(f"\nUpdate complete:")
    print(f"Successfully updated: {successful_updates} files")
    print(f"Failed to update: {failed_updates} files")