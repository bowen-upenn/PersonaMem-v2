import json
import random
from uuid import uuid4
from tqdm import tqdm

import prompts
import utils


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

    # 3) anti-stereotypical preferences
    prompt = prompts.generate_anti_stereotypical_preferences()
    llm.query_llm(prompt, use_history=True, verbose=verbose)

    # 4) verify conflicts
    prompt = prompts.verify_conflicts()
    llm.query_llm(prompt, use_history=True, verbose=verbose)

    # 5) additional therapy-related personal history
    prompt = prompts.generate_therapy_related_history()
    llm.query_llm(prompt, use_history=True, verbose=verbose)

    # 6) generate sensitive private information
    prompt = prompts.generate_sensitive_information()
    final_json = llm.query_llm(prompt, use_history=True, verbose=verbose)

    # 7) find images if image_matcher is provided that match the persona
    if image_matcher:
        matched_images = image_matcher.find_most_similar_image(final_json, top_k=6)
        matched_images = [img_path for img_path, _ in matched_images]  # Filter out low similarity images
        if verbose:
            print(f"Matched images: {matched_images}")

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
        conv_turns = utils.extract_json_from_response(conv_turns)
        if conv_turns:
            conversations[type].append({pref_key: pref, 'who': 'self', 'idx_repeat': idx_repeat, 'conversations': conv_turns, 'updated': False})


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
    print('prompt', prompt)
    conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

    conv_turns = utils.extract_json_from_response(conv_turns)
    conv_turns = utils.merge_consecutive_roles(conv_turns)
    who = 'others' if is_others_pref else 'self'

    if conv_turns:
        conversations[type].append({pref_key: pref, 'who': who, 'conversations': conv_turns, 'updated': False})


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

    prompt = prompts.generate_conversations(persona_str, updated_pref, type, is_others_pref)
    conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

    conv_turns = utils.extract_json_from_response(conv_turns)
    conv_turns = utils.merge_consecutive_roles(conv_turns)
    if conv_turns:
        conversations[type].append({pref_key: updated_pref, 'who': 'self', 'conversations': conv_turns, 'updated': True, 'prev_pref': pref})


def find_preference_from_image_and_generate_conversations(llm, persona_str, image_path, conversations, is_others_pref, verbose=False):
    """
    Find the user's preference based on the content of the image.
    """
    type = 'multimodal'

    # Encode image to base64
    base64_image = utils.encode_image_to_base64(image_path)
    if not base64_image:
        return None, conversations

    prompt = prompts.find_preference_from_image(persona_str, is_others_pref)
    preference = llm.query_llm(prompt, image=base64_image, use_history=True, verbose=verbose)
    preference = utils.extract_after_token(preference, '####').strip()  # Extract the preference after the special token

    prompt = prompts.generate_conversations(persona_str, preference, type, is_others_pref)
    conv_turns = llm.query_llm(prompt, image=base64_image, use_history=True, verbose=verbose)

    conv_turns = utils.extract_json_from_response(conv_turns)
    conv_turns = utils.merge_consecutive_roles(conv_turns)

    # add the image to the user query
    if base64_image:
        conv_turns = utils.rewrite_user_query_to_add_image(conv_turns, base64_image)

    who = 'others' if is_others_pref else 'self'

    if conv_turns:
        conversations[type].append({'multimodal': preference, 'who': who, 'conversations': conv_turns, 'updated': False, 'image_path': image_path})


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

    image_paths = final_json.get("matched_images", [])
    for image_idx, image_path in enumerate(image_paths):
        is_others_pref = image_idx > 0.67 * len(image_paths)   # sorted by the order of relevance
        # Generate preference and conversations as if the user is providing an image to the chatbot
        find_preference_from_image_and_generate_conversations(llm, str(final_json), image_path, conversations, is_others_pref, verbose=verbose)

    try:
        sensitive_info = final_json['sensitive_information']
        # Add conversations with sensitive information
        if sensitive_info:
            random_sensitive_info = get_random_sensitive_info(sensitive_info, verbose)
            type = random.choice(implicit_types)
            if verbose:
                print(f"Sensitive information: {random_sensitive_info} for type {type}")
            prompt = prompts.generate_conversations_sensitive_info(persona_str, random_sensitive_info, type)
            conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

            conv_turns = utils.extract_json_from_response(conv_turns)
            conv_turns = utils.merge_consecutive_roles(conv_turns)
            if conv_turns:
                conversations[type].append({'sensitive_info': random_sensitive_info, 'who': 'self', 'conversations': conv_turns})
    except (json.JSONDecodeError, KeyError) as e:
        print("Failed to parse sensitive_information:", e)
        sensitive_info = {}

    for pref_key, pref_list in [
        ("stereotypical_pref", final_json.get("stereotypical_preferences", [])),
        ("anti_stereotypical_pref", final_json.get("anti_stereotypical_preferences", [])),
        ("therapy_background", final_json.get("therapy_background", []))
    ]:
        for pref_idx, pref in tqdm(enumerate(pref_list)):
            llm.reset_history()

            try:
                # Verify preference alignment
                alignment = verify_preference_alignment(llm, pref, pref_key, self_verify, verbose)

                # Only generate conversations for aligned preferences
                if alignment == 'yes':
                    # Set both the user's own preferences and other people's preferences mentioned by this user, for example, to test the llm
                    is_others_pref = random.random() < 0.33

                    # We assign around 1/3 preferences to induce knowledge-related queries, and assume repetitive queries indicate some interests
                    if random.random() < 0.33 and pref_key != "therapy_background" and 'knowledge_query' in implicit_types:
                        generate_knowledge_queries(llm, persona_str, pref, pref_key, conversations, verbose)
                    else:
                        # Generate cross-domain conversations in random types
                        type = random.choice(implicit_types)
                        generate_cross_domain_conversations(llm, persona_str, pref, pref_key, type, conversations, is_others_pref, verbose)

                    # Generate preference updates
                    if random.random() < 0.67 and not is_others_pref and pref_key not in ["therapy_background", "knowledge_query"]:
                        type = random.choice(implicit_types) if pref_key == "therapy_background" or 'knowledge_query' not in implicit_types else 'knowledge_query'
                        generate_preference_updates(llm, persona_str, pref, pref_key, type, conversations, updates, is_others_pref, verbose)

            except Exception as e:
                print(f"Error processing preference {pref_key} with value {pref}: {e}")
                continue

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


def generate_interactions_from_persona(llm, all_personas, image_matcher, output_path, implicit_types, num_persona=1, self_verify=True, clean=False, verbose=False):
    """
    Load one random persona from the JSONL file, then sequentially query the Assistant API with three prompts:
      1) Add name and demographic info in JSON for the persona
      2) Propose overly stereotypical and anti-stereotypical preferences
      3) Verify and replace any conflicts
      4) Generate cross-domain user-chatbot conversations that implicitly mention the preferences
    Save the final JSON to output_file.
    """
    output_dict = {}

    for _ in tqdm(range(num_persona)):
        persona = random.choice(all_personas)
        
        # Process single persona
        final_json = process_single_persona(llm, persona, implicit_types, self_verify, image_matcher, verbose)
        
        if final_json is not None:
            persona_id = str(uuid4())
            output_dict[persona_id] = final_json

    # Save the full dictionary with persona_id as keys
    utils.save_json(output_dict, output_path, clean=clean)
    print(f"Saved to {output_path}")
    return output_dict