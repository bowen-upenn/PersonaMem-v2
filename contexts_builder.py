import random
import tiktoken
import json
from uuid import uuid4

import utils

ENCODER = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text) -> int:
    if isinstance(text, str):
        return len(ENCODER.encode(text))
    elif isinstance(text, list):
        return sum(len(ENCODER.encode(msg['content'])) for msg in text)
    elif isinstance(text, dict):
        return len(ENCODER.encode(text['content']))
    else:
        raise ValueError("Input must be a string or a list of strings.")


def build_context(interactions, irrelevant_file, context_len):
    """
    Constructs a ~32k-token context by sampling from 'irrelevant_file'
    and interleaving with conversations from 'interactions_file'.
    Returns a dict {context_id: messages} and saves to JSON.
    """
    irrelevant = utils.load_json(irrelevant_file)
    random.shuffle(irrelevant)

    # load data of each persona
    all_uuids = interactions.keys()
    for uuid in all_uuids:
        interaction_curr_persona = interactions[uuid].get("conversations", {})

        messages = []
        total = 0
        idx = 0

        # add ~80% irrelevant messages first
        limit_irrelevant = context_len * 0.8
        while idx < len(irrelevant) and total < limit_irrelevant:
            original_idx = list(irrelevant[idx].keys())[0]
            conv = irrelevant[idx][original_idx]
            total += count_tokens(conv)
            messages.append(conv)
            idx += 1

        num_irrelevant_tokens = total

        # interleave all conversations from interactions
        data_types = interaction_curr_persona.keys()
        for data_type in data_types:
            conv_curr_type = interaction_curr_persona[data_type]
            # randomly shuffle this list
            random.shuffle(conv_curr_type)

            for conv in conv_curr_type:
                conv = conv.get("conversations", [])
                messages.append(conv)
                total += count_tokens(conv)

        print(f"Total tokens in {uuid}: {total}. Number of irrelevant tokens: {num_irrelevant_tokens}.")

        # shuffle the messages as a list of lists
        random.shuffle(messages)

        # flatten the messages as a single list
        messages = [msg for sublist in messages for msg in sublist]

        # save to JSON with unique ID
        utils.save_json(messages, f'data/contexts/context_{uuid}.json')
        print(f"Saved context for {uuid} with {len(messages)} messages.")
