import json


class Colors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color


def extract_json_from_response(response_text):
    """
    Extracts the first JSON object found in a string and returns it as a Python dict.
    Raises ValueError if no valid JSON is found.
    """
    decoder = json.JSONDecoder()
    start_idx = response_text.find('{')
    if start_idx == -1:
        return None

    try:
        obj, _ = decoder.raw_decode(response_text[start_idx:])
        return obj
    except json.JSONDecodeError as e:
        return None


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
