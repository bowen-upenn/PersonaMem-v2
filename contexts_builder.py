import random
import tiktoken
import json
from uuid import uuid4

from data_manager import load_json, save_json

MAX_TOKENS = 1024
ENCODER = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def build_context(interactions_file: str, irrelevant_file: str):
    """
    Constructs a ~32k-token context by sampling from 'irrelevant_file'
    and interleaving with conversations from 'interactions_file'.
    Returns a dict {context_id: messages} and saves to JSON.
    """
    # load data
    interactions = load_json(interactions_file)  # list of personas with conversation data
    irrelevant = load_json(irrelevant_file)

    # flatten irrelevant messages
    flat = []
    for item in irrelevant:
        for seq in item.values():
            flat.extend(seq)
    random.shuffle(flat)

    messages = []
    total = 0
    idx = 0

    # add ~80% irrelevant messages first
    limit_irrelevant = MAX_TOKENS * 0.8
    while idx < len(flat) and total < limit_irrelevant:
        msg = flat[idx]
        total += count_tokens(msg['content'])
        messages.append(msg)
        idx += 1

    # interleave all conversations from interactions
    for inter in interactions:
        for kind in ["stereotypical", "anti_stereotypical"]:
            convs = inter.get("conversations", {}).get(kind, [])
            for conv_obj in convs:
                for msg in conv_obj.get("conversation", []):
                    messages.append(msg)

    # truncate to max tokens if needed
    final_messages = []
    token_sum = 0
    for msg in messages:
        token_count = count_tokens(msg['content'])
        if token_sum + token_count > MAX_TOKENS:
            break
        final_messages.append(msg)
        token_sum += token_count

    # save to JSON with unique ID
    context_id = str(uuid4())
    save_json({context_id: final_messages}, f'context_{context_id}.json')
    return {context_id: final_messages}
