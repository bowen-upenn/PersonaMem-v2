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
        raise ValueError("Input must be a string, dict, or list of dicts.")


def build_context(interactions, context_len=None):
    """
    Constructs a multi-turn conversation list for each persona by interleaving
    their conversation blocks with irrelevant data. Inserts updated blocks
    at random positions. Saves flattened messages to JSON and returns a list of dictionaries:
        [ {role, content}, ... ]
    """
    for uuid, persona in interactions.items():
        # separate blocks
        not_updated = []      # list of message lists
        updated_blocks = []   # list of message lists to insert randomly

        for conv_type, blocks in persona.get("conversations", {}).items():
            for block in blocks:
                msgs = block.get("conversations", [])
                if block.get("updated", False):
                    updated_blocks.append(msgs)
                else:
                    not_updated.append(msgs)

        # shuffle non-updated base blocks
        random.shuffle(not_updated)

        # start ordered list
        ordered = list(not_updated)

        # insert each updated block at a random position
        for msgs in updated_blocks:
            insert_idx = random.randint(0, len(ordered))
            ordered.insert(insert_idx, msgs)

        # flatten to single list
        all_messages = [msg for block in ordered for msg in block]

        # compute total tokens
        total_tokens = count_tokens(all_messages)
        print(f"Total tokens: {total_tokens}.")

        # # optional trimming by token budget
        # if context_len is not None:
        #     trimmed = []
        #     tokens = 0
        #     for msg in all_messages:
        #         tcount = count_tokens(msg)
        #         if tokens + tcount > context_len:
        #             break
        #         trimmed.append(msg)
        #         tokens += tcount
        #     all_messages = trimmed

        # save raw messages
        filename = f"data/contexts/context_{uuid}.json"
        utils.save_json(all_messages, filename)
        print(f"Saved contexts to {filename}.")