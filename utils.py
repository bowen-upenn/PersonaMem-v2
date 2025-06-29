import json
import csv
import os
import re
from json_repair import repair_json


class Colors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filename, clean=False):
    """
    Save dictionary or a list of dictionaries to a JSON file.
    If clean=True and data is not empty, merge with existing JSON (if any).
    """
    if clean and data:
        # Try to load existing data
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, dict):
                        existing = {}
                except Exception:
                    existing = {}
        else:
            existing = {}
        # Update dict with new key(s)
        existing.update(data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    else:
        # Overwrite with new data
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows: list, filename: str):
    if not rows:
        return
    with open(filename, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def extract_json_from_response(response_str):
    """
    Extracts JSON array or object from a response string enclosed in triple backticks with 'json',
    and parses it into a Python dictionary or list.
    """
    # Match content inside ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
    if not match:
        raise ValueError("No JSON content found in response.")

    json_str = match.group(1).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        json_str = repair_json(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}\nExtracted content:\n{json_str}")


def merge_consecutive_roles(messages):
    if not messages:
        return []
    merged = [messages[0].copy()]
    for msg in messages[1:]:
        if msg['role'] == merged[-1]['role']:
            # Merge content with two newlines for readability
            merged[-1]['content'] += "\n\n" + msg['content']
        else:
            merged.append(msg.copy())
    return merged


def extract_after_token(text: str, token: str) -> str:
    """
    Extracts and returns the portion of `text` that follows the first occurrence of `token`.
    If the token is not found, returns an empty string.

    Args:
        text: The input string to search within.
        token: The delimiter token after which the content should be returned.

    Returns:
        The substring of `text` after `token`, stripped of leading/trailing whitespace,
        or an empty string if `token` is not present.
    """
    try:
        # Find the starting index of the token
        start = text.index(token) + len(token)
        # Return everything after the token, stripped of whitespace
        return text[start:].strip()
    except ValueError:
        # Token not found
        return ""
