import json
import random
import openai
import re
from uuid import uuid4

from data_manager import save_json
from json_repair import repair_json
import prompts
import utils


def generate_interactions_from_persona(llm, all_personas, output_path, num_persona=1, verbose=False):
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

        # # collect conversations for stereotypical_preferences
        # conversations = []
        # for pref in final_json.get("stereotypical_preferences", []):
        #     prompt = prompts.generate_conversation(pref)
        #     conv_json = llm.query_llm(prompt, use_assistant=False, verbose=verbose)
        #     # parse the returned JSON array
        #     conv_json = repair_json(conv_json)
        #     conversation = json.loads(conv_json)
        #     conversations.append({
        #         "preference": pref,
        #         "conversation": conversation
        #     })
        #
        # # same for anti_stereotypical_preferences
        # for pref in final_json.get("anti_stereotypical_preferences", []):
        #     prompt = prompts.generate_conversation(pref)
        #     conv_json = llm.query_llm(prompt, use_assistant=False, verbose=verbose)
        #     # parse the returned JSON array
        #     conv_json = repair_json(conv_json)
        #     conversation = json.loads(conv_json)
        #     conversations.append({
        #         "preference": pref,
        #         "conversation": conversation
        #     })
    #
    #     # 6) attach to the parsed output and save
    #     final_json["conversations"] = conversations
    #     persona_id = str(uuid4())
    #     output_dict[persona_id] = final_json
    #
    # # Save the full dictionary with persona_id as keys
    # print('output_dict', output_dict)
    # save_json(output_dict, output_path)
    return output_dict