import json
import random
import openai
import re
from uuid import uuid4

from json_repair import repair_json
import prompts
import utils


def generate_interactions_from_persona(llm, all_personas, output_path, implicit_types, num_persona=1, verbose=False):
    """
    Load one random persona from the JSONL file, then sequentially query the Assistant API with three prompts:
      1) Add name and demographic info in JSON for [PERSONA]
      2) Propose 10 overly stereotypical preferences
      3) Propose 10 anti-stereotypical preferences avoiding conflicts
      4) Verify and replace any conflicts
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
        demo_json = llm.query_llm(prompt, use_history=True, verbose=verbose)

        # 2) stereotypical preferences
        prompt = prompts.generate_stereotypical_preferences()
        stereotypical_json = llm.query_llm(prompt, use_history=True, verbose=verbose)

        # 3) anti-stereotypical preferences
        prompt = prompts.generate_anti_stereotypical_preferences()
        anti_json = llm.query_llm(prompt, use_history=True, verbose=verbose)

        # 4) verify conflicts
        prompt = prompts.verify_conflicts()
        final_json = llm.query_llm(prompt, use_history=True, verbose=verbose)

        # 5) curate conversations
        # parse JSON part from the response
        try:
            final_json = utils.extract_json_from_response(final_json)
            print("Parsed JSON:", final_json)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            continue

        # collect conversations for stereotypical_preferences
        conversations = {}
        for type in implicit_types:
            conversations[type] = []

            for pref_key, pref_list in [
                ("stereotypical_pref", final_json.get("stereotypical_preferences", [])),
                ("anti_stereotypical_pref", final_json.get("anti_stereotypical_preferences", []))
            ]:
                for pref in pref_list:
                    # 1) Guess which persona fits the preference
                    prompt_guess = prompts.guess_persona(pref, anti=(pref_key == "anti_stereotypical_pref"))
                    llm.reset_history()
                    guessed_persona = llm.query_llm(prompt_guess, use_history=True, verbose=verbose)

                    # 2) Check alignment of guessed persona with actual persona
                    prompt_check = prompts.check_alignment_with_population_mean(guessed_persona)
                    resp_check = llm.query_llm(prompt_check, use_history=True, verbose=verbose)
                    # Extract the final answer after '####Final Answer'
                    alignment = utils.extract_after_token(resp_check, '####Final Answer').strip().lower()

                    if alignment == 'yes':
                        # Only generate email conversations for aligned preferences
                        email_prompt = prompts.generate_emails(persona_str, pref)
                        conv_turns = llm.query_llm(email_prompt, use_history=False, verbose=verbose)
                        conv_turns = utils.extract_json_from_response(conv_turns)
                        if conv_turns:
                            conversations[type].append({pref_key: pref, 'conversations': conv_turns})

            # Update final_json to only include aligned preferences
            aligned_stereo = [c['stereotypical_pref'] for convs in conversations.values() for c in convs if 'stereotypical_pref' in c]
            aligned_anti = [c['anti_stereotypical_pref'] for convs in conversations.values() for c in convs if 'anti_stereotypical_pref' in c]

            final_json['stereotypical_preferences'] = aligned_stereo
            final_json['anti_stereotypical_preferences'] = aligned_anti

        print('conversations', conversations)
        # 6) attach to the parsed output and save
        final_json["conversations"] = conversations
        persona_id = str(uuid4())
        output_dict[persona_id] = final_json

    # Save the full dictionary with persona_id as keys
    print('output_dict', output_dict)
    utils.save_json(output_dict, output_path)
    return output_dict