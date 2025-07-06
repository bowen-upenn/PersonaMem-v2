import json
import random
import openai
import re
from uuid import uuid4
from datasets import load_dataset

import prompts
import utils


def generate_interactions_from_persona(llm, all_personas, output_path, implicit_types, num_persona=1, self_verify=True, clean=False, verbose=False):
    """
    Load one random persona from the JSONL file, then sequentially query the Assistant API with three prompts:
      1) Add name and demographic info in JSON for the persona
      2) Propose overly stereotypical and anti-stereotypical preferences
      3) Verify and replace any conflicts
      4) Generate cross-domain user-chatbot conversations that implicitly mention the preferences
    Save the final JSON to output_file.
    """
    output_dict = {}

    for _ in range(num_persona):
        persona = random.choice(all_personas)
        persona_str = json.dumps(persona, ensure_ascii=False)
        if verbose:
            print(f"Persona: {persona_str}")

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

        # # 6) generate sensitive private information
        prompt = prompts.generate_sensitive_information()
        final_json = llm.query_llm(prompt, use_history=True, verbose=verbose)

        # 7) curate conversations
        # parse JSON part from the response
        try:
            final_json = utils.extract_json_from_response(final_json)
            # print("Parsed JSON:", final_json)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            continue

        conversations = {}
        for type in implicit_types:
            conversations[type] = []
        updates = {}

        try:
            sensitive_info = final_json['sensitive_information']
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            sensitive_info = {}

        for pref_key, pref_list in [
            ("stereotypical_pref", final_json.get("stereotypical_preferences", [])),
            ("anti_stereotypical_pref", final_json.get("anti_stereotypical_preferences", [])),
            ("therapy_background", final_json.get("therapy_background", [])),
        ]:
            for pref in pref_list:
                llm.reset_history()
                # We verify if a preference is actually aligned with the model's believed stereotypes or anti-stereotypes
                if not self_verify or pref_key == "therapy_background" or pref_key == "sensitive_information":
                    alignment = 'yes'
                else:
                    # 1) Guess which persona fits the preference
                    prompt_guess = prompts.guess_persona(pref, anti=(pref_key == "anti_stereotypical_pref"))
                    guessed_persona = llm.query_llm(prompt_guess, use_history=True, verbose=verbose)

                    # 2) Check alignment of guessed persona with actual persona
                    prompt_check = prompts.check_alignment_with_population_mean(guessed_persona)
                    resp_check = llm.query_llm(prompt_check, use_history=True, verbose=verbose)
                    # Extract the final answer after '####Final Answer'
                    alignment = utils.extract_after_token(resp_check, '####Final Answer').strip().lower()

                # Only generate email conversations for aligned preferences
                if alignment == 'yes':
                    # Set both the user's own preferences and other people's preferences mentioned by this user, for example, to test the llm
                    is_others_pref = random.random() < 0.33

                    # We assign around 1/3 preferences to induce knowledge-related queries, and assume repetitive queries indicate some interests
                    if random.random() < 0.33 and pref_key != "therapy_background" and 'knowledge_query' in implicit_types:
                        """ 
                        This block of code generates repetitive user knowledge queries to the chatbot
                        """
                        repeat = random.randint(1, 6)
                        type = 'knowledge_query'
                        if verbose:
                            print(f'{utils.Colors.OKGREEN}{pref_key}: {utils.Colors.ENDC}{pref}{utils.Colors.OKGREEN} data type: {utils.Colors.ENDC}{type}')

                        for idx_repeat in range(repeat):
                            llm.reset_history()
                            prompt = prompts.generate_conversations(persona_str, pref, type, is_others_pref=False)   # Interests shown by repetitive knowledge queries shall always belong to the user's own interests

                            conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)
                            conv_turns = utils.extract_json_from_response(conv_turns)
                            if conv_turns:
                                conversations[type].append({pref_key: pref, 'who': 'self', 'idx_repeat': idx_repeat, 'conversations': conv_turns, 'updated': False})
                    else:
                        """
                        This is the default block that implicitly encode user personas and preferences into random cross-domain scenarios
                        """
                        # Find one random type from implicit_types for each pref
                        type = random.choice(implicit_types)
                        if verbose:
                            print(f'{utils.Colors.OKGREEN}{pref_key}: {utils.Colors.ENDC}{pref}{utils.Colors.OKGREEN} data type: {utils.Colors.ENDC}{type}')

                        # Access if the chatbot can recognize sensitive private information and avoid them in the responses
                        random_sensitive_info = None
                        if random.random() < 0.33:
                            key = random.choice(list(sensitive_info.keys()))
                            value = str(sensitive_info[key])
                            random_sensitive_info = f"{key}: {value}"
                            if verbose:
                                print("random_sensitive_info", random_sensitive_info)

                        prompt = prompts.generate_conversations(persona_str, pref, type, is_others_pref, random_sensitive_info)
                        conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

                        conv_turns = utils.extract_json_from_response(conv_turns)
                        conv_turns = utils.merge_consecutive_roles(conv_turns)
                        who = 'others' if is_others_pref else 'self'

                        if conv_turns:
                            if random_sensitive_info:
                                conversations[type].append({pref_key: pref, 'who': who, 'conversations': conv_turns, 'updated': False, 'sensitive_info': random_sensitive_info})
                            else:
                                conversations[type].append({pref_key: pref, 'who': who, 'conversations': conv_turns, 'updated': False})

                    # We update 2/3 preferences to their opposite along the timeline
                    """
                    This block of code generates preference updates
                    """
                    if random.random() < 0.67 and not is_others_pref and pref_key not in ["therapy_background", "knowledge_query"]:
                        llm.reset_history()
                        prompt = prompts.update_preference(pref)   # Interests shown by repetitive knowledge queries shall always belong to the user's own interests
                        updated_pref = llm.query_llm(prompt, use_history=False, verbose=verbose)

                        # Record the update
                        updates[pref] = updated_pref
                        llm.reset_history()

                        random_sensitive_info = None
                        if random.random() < 0.33:
                            key = random.choice(list(sensitive_info.keys()))
                            value = sensitive_info[key]
                            random_sensitive_info = f"{key}: {value}"
                            if verbose:
                                print("random_sensitive_info", random_sensitive_info)

                        prompt = prompts.generate_conversations(persona_str, updated_pref, type, is_others_pref, random_sensitive_info)
                        conv_turns = llm.query_llm(prompt, use_history=False, verbose=verbose)

                        conv_turns = utils.extract_json_from_response(conv_turns)
                        conv_turns = utils.merge_consecutive_roles(conv_turns)
                        if conv_turns:
                            if random_sensitive_info:
                                conversations[type].append({pref_key: updated_pref, 'who': 'self', 'conversations': conv_turns, 'updated': True, 'prev_pref': pref, 'sensitive_info': random_sensitive_info})
                            else:
                                conversations[type].append({pref_key: updated_pref, 'who': 'self', 'conversations': conv_turns, 'updated': True, 'prev_pref': pref})

            # Update final_json to only include aligned preferences
            aligned_stereo = [c['stereotypical_pref'] for convs in conversations.values() for c in convs if 'stereotypical_pref' in c]
            aligned_anti = [c['anti_stereotypical_pref'] for convs in conversations.values() for c in convs if 'anti_stereotypical_pref' in c]

            final_json['stereotypical_preferences'] = list(set(aligned_stereo))     # the repeated calls on knowledge_query may introduce repeated preferences in the previous step
            final_json['anti_stereotypical_preferences'] = list(set(aligned_anti))
            final_json['preference_updates'] = updates

        # 8) attach to the parsed output and save
        final_json["conversations"] = conversations
        persona_id = str(uuid4())
        output_dict[persona_id] = final_json

    # Save the full dictionary with persona_id as keys
    utils.save_json(output_dict, output_path, clean=clean)
    print(f"Saved to {output_path}")
    return output_dict