import json
from pathlib import Path
import argparse
from json_repair import repair_json
from tqdm import tqdm

import prompts
import utils


def generate_qa_for_each_element(llm, element, conv_list, verbose=False):
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
    if element['updated']:
        prev_pref = element['prev_pref']
        for curr_element in conv_list:
            if (('stereotypical_pref' in curr_element and curr_element['stereotypical_pref'] == prev_pref) or
                    ('anti_stereotypical_pref' in curr_element and curr_element['anti_stereotypical_pref'] == prev_pref)):
                # Previous preference must already have correct answers generated
                prev_correct = curr_element['correct_answer']
            else:
                continue

    if who == 'self':
        element['correct_answer'] = answers.get('correct')
        incorrect = []
        if prev_correct:    # only happen with who == self
            for key in ('random1', 'generic'):
                if key in answers:
                    incorrect.append(answers[key])
            incorrect.append(prev_correct)
        else:
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


def generate_qa(llm, input_path, output_path, verbose=False):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for uuid, persona in data.items():
        conversations_by_type = persona.get("conversations", {})
        for conv_type, conv_list in conversations_by_type.items():
            print('conv_type', conv_type)
            for conv_elem in tqdm(conv_list):
                llm.reset_history()

                if conv_type != 'knowledge_query':
                    continue

                if conv_type == 'knowledge_query':
                    # print("conv_elem['idx_repeat']", conv_elem['idx_repeat'])
                    if 'idx_repeat' not in conv_elem or conv_elem['idx_repeat'] < 2:
                        continue # we assume the user asking a topic >= 3 times indicates user interests

                qa_fields = generate_qa_for_each_element(llm, conv_elem, conv_list, verbose=verbose)
                conv_elem.update({
                    "user_query": qa_fields.get("user_query"),
                    "correct_answer": qa_fields.get("correct_answer"),
                    "incorrect_answers": qa_fields.get("incorrect_answers"),
                })

    utils.save_json(data, output_path)
    print(f"Saved to {output_path}")

