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
from collections import defaultdict
import time
import threading

import prompts
import utils
from qa_generator import generate_qa_for_each_element

# Global topic counter for tracking all categorizations across all personas
GLOBAL_TOPIC_COUNTER = defaultdict(int)
TOPIC_COUNTER_LOCK = threading.Lock()

# File saving lock to prevent race conditions during parallel file saves
FILE_SAVE_LOCK = threading.Lock()


def save_topic_counts(output_path, verbose=False):
    """
    Save the global topic counts to a separate JSON file.
    
    Args:
        output_path: Base output path to derive the topic counts filename
        verbose: Whether to print debug information
    """
    # Create topic counts filename based on output path
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    
    # Remove extension and add topic_counts suffix
    if base_name.endswith('.json'):
        base_name_no_ext = base_name[:-5]
    else:
        base_name_no_ext = base_name
    
    topic_counts_path = os.path.join(output_dir, f"{base_name_no_ext}_topic_counts.json")
    
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
            "generated_at": utils.create_timestamped_filename("", "", timestamp=None).split('_')[-1]  # Get just timestamp
        },
        "topic_counts": sorted_topics
    }
    
    # Save to file
    utils.save_json(final_data, topic_counts_path, clean=True)
    
    if verbose:
        print(f"Saved topic counts to: {topic_counts_path}")
        print(f"Total categorizations: {final_data['metadata']['total_categorizations']}")
        print(f"Unique topics: {final_data['metadata']['unique_topics']}")
        print(f"Top 5 topics: {list(sorted_topics.items())[:5]}")
    
    return topic_counts_path


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


def categorize_single_item(llm, text, global_topics, verbose=False):
    """
    Categorize a single text item (preference or query) into topics using the LLM.
    
    Args:
        llm: QueryLLM instance
        text: The text to categorize
        global_topics: List of existing global topics
        verbose: Whether to print debug information
        
    Returns:
        str: The topic name
    """
    # Generate categorization prompt
    prompt = prompts.categorize_preference_topic(text, global_topics)
    
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
            # Record topic count in global counter (thread-safe)
            with TOPIC_COUNTER_LOCK:
                GLOBAL_TOPIC_COUNTER[topic] += 1
            
            if verbose:
                print(f"Categorized '{text}' as topic: '{topic}'")
            return topic
        else:
            # Record "Uncategorized" count
            with TOPIC_COUNTER_LOCK:
                GLOBAL_TOPIC_COUNTER["Uncategorized"] += 1
                
            if verbose:
                print(f"Failed to categorize '{text}', using 'Uncategorized'")
            return "Uncategorized"
            
    except Exception as e:
        # Record "Uncategorized" count for errors
        with TOPIC_COUNTER_LOCK:
            GLOBAL_TOPIC_COUNTER["Uncategorized"] += 1
            
        if verbose:
            print(f"Error categorizing '{text}': {e}, using 'Uncategorized'")
        return "Uncategorized"


def verify_conflicts_structured(llm, final_json_response, persona_id, verbose=False):
    """
    Verify conflicts and duplicates in preferences using a structured approach.
    Processes each preference individually against existing lists.
    """    
    # Get initial preference lists
    stereotypical_prefs = final_json_response.get("stereotypical_preferences", [])
    anti_stereotypical_prefs = final_json_response.get("anti_stereotypical_preferences", [])
    
    if verbose:
        print(f"Initial counts - Stereotypical: {len(stereotypical_prefs)}, Anti-stereotypical: {len(anti_stereotypical_prefs)}")
    
    # Clean stereotypical preferences first
    cleaned_stereotypical = []
    for pref in stereotypical_prefs:
        if not pref or not pref.strip():
            continue
            
        # Check against existing cleaned preferences and anti-stereotypical list
        existing_stereo_str = "\n".join([f"- {p}" for p in cleaned_stereotypical])
        existing_anti_str = "\n".join([f"- {p}" for p in anti_stereotypical_prefs])
        
        prompt = prompts.verify_stereotypical_preference(pref, existing_stereo_str, existing_anti_str)
        try:
            response = llm.query_llm(prompt, use_history=False, verbose=verbose)
            decision = utils.extract_after_token(response, '###Decision').strip().upper()
        except Exception as e:
            if verbose:
                print(f"Error verifying stereotypical preference '{pref}': {e}")
            continue

        if "KEEP" in decision or not decision:  # Default to keep if unclear
            cleaned_stereotypical.append(pref)
            if verbose:
                print(f"✓ Keeping stereotypical: '{pref[:50]}...'")
        elif verbose:
            print(f"✗ Removing stereotypical: '{pref[:50]}...'")
    
    # Clean anti-stereotypical preferences against cleaned stereotypical list
    cleaned_anti_stereotypical = []
    for pref in anti_stereotypical_prefs:
        if not pref or not pref.strip():
            continue
            
        existing_anti_str = "\n".join([f"- {p}" for p in cleaned_anti_stereotypical])
        existing_stereo_str = "\n".join([f"- {p}" for p in cleaned_stereotypical])

        prompt = prompts.verify_anti_stereotypical_preference(pref, existing_stereo_str, existing_anti_str)
        try:
            response = llm.query_llm(prompt, use_history=False, verbose=verbose)
            decision = utils.extract_after_token(response, '###Decision').strip().upper()
        except Exception as e:
            if verbose:
                print(f"Error verifying anti-stereotypical preference '{pref}': {e}")
            continue

        if "KEEP" in decision or not decision:  # Default to keep if unclear
            cleaned_anti_stereotypical.append(pref)
            if verbose:
                print(f"✓ Keeping anti-stereotypical: '{pref[:50]}...'")
        elif verbose:
            print(f"✗ Removing anti-stereotypical: '{pref[:50]}...'")
    
    # Update the response
    final_json_response["stereotypical_preferences"] = cleaned_stereotypical
    final_json_response["anti_stereotypical_preferences"] = cleaned_anti_stereotypical
    
    if verbose:
        removed_stereo = len(stereotypical_prefs) - len(cleaned_stereotypical)
        removed_anti = len(anti_stereotypical_prefs) - len(cleaned_anti_stereotypical)
        print(f"Final counts - Stereotypical: {len(cleaned_stereotypical)}, Anti-stereotypical: {len(cleaned_anti_stereotypical)}")
        print(f"Removed {removed_stereo} stereotypical and {removed_anti} anti-stereotypical preferences")
    
    return final_json_response


def expand_persona_info(llm, persona_str, persona_id, image_matcher=None, verbose=False):
    """
    Expand persona with demographic info and generate preferences and background.
    
    Returns:
        tuple: (expanded_persona_str, final_json)
    """
    while True:
        llm.reset_history()
        # 1) demographic info
        prompt = prompts.expand_persona(persona_str)
        persona_str = llm.query_llm(prompt, use_history=True, verbose=verbose)
        
        # 2) stereotypical and anti-stereotypical preferences
        prompt = prompts.generate_stereotypical_and_antistereotypical_preferences()
        final_json_temp = llm.query_llm(prompt, use_history=True, verbose=verbose)
        print(f"Done generating stereotypical and anti-stereotypical preferences for persona {persona_id}")

        # 3) verify conflicts - structured approach
        final_json_temp = utils.extract_json_from_response(final_json_temp)
        final_json_temp = verify_conflicts_structured(llm, final_json_temp, persona_id, verbose)
        print(f"Done verifying conflicts in preferences for persona {persona_id}")

        # 4) additional therapy-related personal history
        final_json_temp = json.dumps(final_json_temp)
        prompt = prompts.generate_therapy_related_history(final_json_temp)
        llm.query_llm(prompt, use_history=True, verbose=verbose)
        print(f"Done generating therapy-related personal history for persona {persona_id}")

        # 5) additional health and medical-related personal history
        prompt = prompts.generate_health_and_medical_conditions()
        final_json_temp = llm.query_llm(prompt, use_history=True, verbose=verbose)
        print(f"Done generating health and medical-related personal history for persona {persona_id}")

        # 6) generate sensitive private information
        prompt = prompts.generate_sensitive_information()
        final_json = llm.query_llm(prompt, use_history=True, verbose=verbose)
        print(f"Done generating sensitive private information for persona {persona_id}")
        if 'sorry' in final_json.lower():
            final_json = final_json_temp

        # Convert final json from a string to a JSON dictionary
        final_json = utils.extract_json_from_response(final_json)

        # Sanity check: ensure required fields have non-empty values
        required_fields = ["stereotypical_preferences", "anti_stereotypical_preferences", "therapy_background", "health_and_medical_conditions"]
        missing_fields = []
        
        for field in required_fields:
            if field not in final_json or not final_json[field] or len(final_json[field]) < 5:
                missing_fields.append(field)
        
        if len(missing_fields) > 0:
            print(f"Missing or empty required fields: {missing_fields}. Retrying expand_persona_info...")
        else:
            break

    # 8) find images if image_matcher is provided that match the persona
    matched_images = []
    if image_matcher:
        random_num_images = random.randint(3, 8)
        matched_images = image_matcher.find_most_similar_image(persona_str, top_k=random_num_images)
        matched_images = [img_path for img_path, _ in matched_images]  # Filter out low similarity images
        if verbose:
            print(f"Matched images: {len(matched_images)}: {matched_images}")
        print("Done finding images that match the persona.")
    final_json['matched_images'] = matched_images

    if verbose:
        print({f"all keys in final_json": list(final_json.keys())})

    # 9) Categorize all preferences for global topic counting only
    global_topics = []  # Track topics within this persona for consistency
    
    # Categorize stereotypical preferences (for global counting only)
    if 'stereotypical_preferences' in final_json:
        for pref in final_json['stereotypical_preferences']:
            if pref:  # Only categorize non-empty preferences
                topic = categorize_single_item(llm, pref, global_topics, verbose)
                if topic not in global_topics:
                    global_topics.append(topic)

    # Categorize anti-stereotypical preferences (for global counting only)
    if 'anti_stereotypical_preferences' in final_json:
        for pref in final_json['anti_stereotypical_preferences']:
            if pref:  # Only categorize non-empty preferences
                topic = categorize_single_item(llm, pref, global_topics, verbose)
                if topic not in global_topics:
                    global_topics.append(topic)
    
    # Categorize therapy background (for global counting only)
    if 'therapy_background' in final_json:
        for pref in final_json['therapy_background']:
            if pref:  # Only categorize non-empty preferences
                topic = categorize_single_item(llm, pref, global_topics, verbose)
                if topic not in global_topics:
                    global_topics.append(topic)
    
    # Categorize health and medical conditions (for global counting only)
    if 'health_and_medical_conditions' in final_json:
        for pref in final_json['health_and_medical_conditions']:
            if pref:  # Only categorize non-empty preferences
                topic = 'health_and_medical'
                if topic not in global_topics:
                    global_topics.append(topic)
                # Add to global counter
                with TOPIC_COUNTER_LOCK:
                    GLOBAL_TOPIC_COUNTER[topic] += 1

    return persona_str, final_json


def verify_preference_alignment(llm, pref, pref_key, self_verify, verbose=False):
    """
    Verify if a preference is actually aligned with the model's believed stereotypes or anti-stereotypes.
    
    Returns:
        str: 'yes' if aligned, 'no' otherwise
    """
    if not self_verify or pref_key not in ["stereotypical_pref", "anti_stereotypical_pref"]:
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


def generate_knowledge_queries(llm, persona_str, pref, pref_key, conversations, topic_preference=None, verbose=False):
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
            # Add topic information if available
            if topic_preference:
                element['topic_preference'] = topic_preference
            else:
                element['topic_preference'] = "Uncategorized"
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


def generate_cross_domain_conversations(llm, persona_str, pref, pref_key, type, conversations, is_others_pref, topic_preference=None, verbose=False):
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
        # Add topic information if available
        if topic_preference:
            element['topic_preference'] = topic_preference
        else:
            element['topic_preference'] = "Uncategorized"
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
    if new_element:
        user_query = new_element['user_query']
        correct_answer = new_element['correct_answer']

        if len(user_query) == 0:
            user_query = new_element['conversations'][0]['content']

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
    

def convert_preferences_to_conversations(llm, persona_str, persona_id, final_json, implicit_types, self_verify, verbose=False):
    """
    Process all preferences and generate conversations for each aligned preference.
    
    Returns:
        tuple: (conversations, updates)
    """
    conversations = {'multimodal': []}
    for curr_type in implicit_types:
        conversations[curr_type] = []
    updates = {}

    image_paths = final_json.get("matched_images", [])
    for image_idx, image_path in enumerate(image_paths):
        llm.reset_history()
        is_others_pref = image_idx > 0.67 * len(image_paths)   # sorted by the order of relevance
        # Generate preference and conversations as if the user is providing an image to the chatbot
        try:
            element = find_preference_from_image_and_generate_conversations(llm, str(final_json), image_path, conversations, is_others_pref, verbose=verbose)
        except Exception as e:
            continue

    # print(f"All keys in final_json: {list(final_json.keys())}")
    sensitive_info = final_json.get('sensitive_information', {})
    # Add conversations with sensitive information
    if sensitive_info:
        random_sensitive_info = get_random_sensitive_info(sensitive_info, verbose)
        random_type = random.choice(implicit_types)
        if verbose:
            print(f"Sensitive information: {random_sensitive_info} for type {random_type}")
        prompt = prompts.generate_conversations_sensitive_info(persona_str, random_sensitive_info, random_type)
        conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

        try:
            conv_turns = utils.extract_json_from_response(conv_turns)
            conv_turns = utils.merge_consecutive_roles(conv_turns)
            if conv_turns:
                conversations[random_type].append({'sensitive_info': random_sensitive_info, 'who': 'self', 'conversations': conv_turns})
        except json.JSONDecodeError as e:
            try:
                conv_turns = repair_json(conv_turns)
                conv_turns = utils.extract_json_from_response(conv_turns)
                conv_turns = utils.merge_consecutive_roles(conv_turns)
                if conv_turns:
                    conversations[random_type].append({'sensitive_info': random_sensitive_info, 'who': 'self', 'conversations': conv_turns})
            except json.JSONDecodeError as e:
                print(f"Failed to parse knowledge query response: {e}")
                sensitive_info = {}
    
    # Global topic list for this persona's conversation generation
    global_topics = []
    
    for pref_key, pref_list in [
        ("stereotypical_pref", final_json.get("stereotypical_preferences", [])),
        ("anti_stereotypical_pref", final_json.get("anti_stereotypical_preferences", [])),
        ("therapy_background", final_json.get("therapy_background", [])),
        ("health_and_medical_conditions", final_json.get("health_and_medical_conditions", []))
    ]:
        for pref_idx, pref in tqdm(enumerate(pref_list), desc=f"Processing {pref_key} preferences for persona {persona_id}", total=len(pref_list)):
            llm.reset_history()

            try:
                # Categorize the preference inline during conversation generation
                if pref_key == "health_and_medical_conditions":
                    topic_preference = "health_and_medical"
                    # Add to global counter for health/medical
                    with TOPIC_COUNTER_LOCK:
                        GLOBAL_TOPIC_COUNTER[topic_preference] += 1
                else:
                    # Categorize this preference dynamically
                    topic_preference = categorize_single_item(llm, pref, global_topics, verbose)
                    if topic_preference not in global_topics:
                        global_topics.append(topic_preference)
                
                # Verify preference alignment
                alignment = verify_preference_alignment(llm, pref, pref_key, self_verify, verbose)

                # Only generate conversations for aligned preferences
                if alignment == 'yes':
                    # Set both the user's own preferences and other people's preferences mentioned by this user, for example, to test the llm
                    is_others_pref = random.random() < 0.33
                    
                    # Check if we're in the last num(implicit_types) preferences and if any type is empty
                    remaining_prefs = len(pref_list) - pref_idx
                    if remaining_prefs <= len(implicit_types):
                        # Find empty conversation types
                        empty_types = [t for t in implicit_types if len(conversations[t]) == 0]
                        if empty_types:
                            # Use the first empty type instead of random
                            random_type = empty_types[0]
                        else:
                            # All types have conversations, use random
                            random_type = random.choice(implicit_types)
                    else:
                        # Not in the final stretch, use random type
                        random_type = random.choice(implicit_types)

                    # We assign around 15 percent of preferences to induce knowledge-related queries, and assume repetitive queries (1 to 6) indicate some interests
                    if random.random() < 0.15 and pref_key != "therapy_background" and 'knowledge_query' in implicit_types:
                        element = generate_knowledge_queries(llm, persona_str, pref, pref_key, conversations, topic_preference, verbose)
                    else:
                        # Generate cross-domain conversations in selected type
                        element = generate_cross_domain_conversations(llm, persona_str, pref, pref_key, random_type, conversations, is_others_pref, topic_preference, verbose)

                    has_asked_to_forget = False
                    if random.random() < 0.33 and not is_others_pref and element:
                        try:
                            user_ask_to_forget(llm, element, conversations, random_type, verbose)
                            has_asked_to_forget = True
                        except Exception as e:
                            print(f"Error asking user to forget: {e}")

                    # Generate preference updates
                    if random.random() < 0.67 and not is_others_pref and pref_key not in ["therapy_background", "knowledge_query"] and not has_asked_to_forget:
                        random_type = random.choice(implicit_types) if pref_key == "therapy_background" or 'knowledge_query' not in implicit_types else 'knowledge_query'
                        generate_preference_updates(llm, persona_str, pref, pref_key, random_type, conversations, updates, is_others_pref, verbose)

            except Exception as e:
                print(f"Error processing preference {pref_key} with value {pref}: {e}")
                continue
        
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


def process_single_persona(llm, persona, persona_id, implicit_types, self_verify, image_matcher, verbose=False):
    """
    Process a single persona to generate all its interactions and conversations.
    
    Returns:
        dict: Complete persona data with conversations
    """
    # Expand persona info and generate preferences
    persona_str = json.dumps(persona, ensure_ascii=False)
    if verbose:
        print(f"Persona: {persona_str}")

    persona_str, final_json_response = expand_persona_info(llm, persona_str, persona_id, image_matcher, verbose)

    # Process preferences and generate conversations
    conversations, updates = convert_preferences_to_conversations(llm, persona_str, persona_id, final_json_response, implicit_types, self_verify, verbose)

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
    
    # Set a unique random seed for this thread to ensure different random choices
    random.seed(int(time.time() * 1000000) + idx)
    
    # try:
    # Process single persona
    final_json = process_single_persona(llm, persona, idx, implicit_types, self_verify, image_matcher, verbose)

    if final_json is not None:
        # Create individual persona file path
        output_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        
        # Keep timestamp but remove extension from base name
        if base_name.endswith('.json'):
            base_name_no_ext = base_name[:-5]  # Remove .json
        else:
            base_name_no_ext = base_name
        
        full_path = os.path.join(output_dir, f"{base_name_no_ext}_persona{idx}.json")
        
        return idx, final_json, full_path
    else:
        return idx, None, None
            
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
        import time
        # Prepare arguments for each persona
        persona_args = []
        for i in range(num_persona):
            idx = persona_start_idx + i  # Use the adjusted index
            # Set a unique seed for this specific persona selection
            temp_seed = int(time.time() * 1000000) + idx
            random.seed(temp_seed)
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
                    # try:
                    persona_id, final_json, full_path = future.result()
                    
                    if final_json is not None:
                        # Add to output dict
                        output_dict[persona_id] = final_json
                        
                        # Thread-safe file saving - only clean on first file if clean=True
                        with FILE_SAVE_LOCK:
                            should_clean = clean and persona_id == persona_start_idx
                            utils.save_json({persona_id: final_json}, full_path, clean=should_clean)
                            print(f"Saved persona {persona_id} to {full_path}")
                        
                    # except Exception as e:
                    #     args = future_to_args[future]
                    #     idx = args[0]
                    #     print(f"Error in future for persona {idx}: {e}")
    else:
        # Sequential processing
        import time
        for i in tqdm(range(num_persona), desc="Processing personas sequentially"):
            idx = persona_start_idx + i  # Use the adjusted index
            # Set unique random seed for each iteration
            random.seed(int(time.time() * 1000000) + idx)
            persona = random.choice(all_personas)
            try:
                # Process single persona
                final_json = process_single_persona(llm, persona, idx, implicit_types, self_verify, image_matcher, verbose)

                if final_json is not None:
                    # Generate a truly unique persona ID
                    persona_id = f"{uuid4()}-{int(time.time() * 1000000)}-{idx}"

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
    
    # Save topic counts after all processing is complete
    topic_counts_file = save_topic_counts('data/preference_topic_counts.json', verbose=verbose)
    print(f"Topic categorization complete. Counts saved to: {topic_counts_file}")
    
    return output_dict