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

