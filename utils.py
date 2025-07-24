import json
import csv
import os
import re
import base64
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
    if clean:
        # Overwrite with new data
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
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
    # Check if response_str is already a valid JSON or just a string
    if isinstance(response_str, str):
        # Match content inside ```json ... ```
        try:
            match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
            json_str = match.group(1).strip()
        except:
            # If no match, assume the whole response is JSON
            json_str = response_str.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            json_str = repair_json(json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON: {e}\nExtracted content:\n{json_str}")
    else:
        return response_str


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


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string for API calls.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image, or None if error
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def rewrite_user_query_to_add_image(conv_turns, image_path):
    """
    Rewrite the user query in conversation turns to include the base64 image.

    Args:
        conv_turns: List of conversation turns
        image_path: Path to the image file

    Returns:
        List of rewritten conversation turns

    We save image_path as an efficient placeholder.
    In context_builder.py, we will rewrite the image_path with the actual base64 string,
    so that we won't save the base64 string redundantly.
    """
    if not conv_turns:
        return []

    rewritten = []
    for turn in conv_turns:
        if turn['role'] == 'user':
            # Add the image to the user query
            turn['content'] = [
                {"type": "text", "text": turn['content']},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_path}"
                    }
                }
            ]
        rewritten.append(turn)
    return rewritten


def create_timestamped_filename(base_dir, base_name, extension='.json', timestamp=None):
    """
    Create a timestamped filename in the format: {base_dir}/{base_name}_{timestamp}.{extension}
    
    Args:
        base_dir (str): Base directory path
        base_name (str): Base filename without extension
        extension (str): File extension (including the dot)
        timestamp (str, optional): Timestamp string. If None, generates Pacific time timestamp
    
    Returns:
        str: Full timestamped file path
    """
    import pytz
    from datetime import datetime
    
    if timestamp is None:
        pacific_tz = pytz.timezone('US/Pacific')
        now = datetime.now(pacific_tz)
        timestamp = now.strftime('%y%m%d_%H%M%S')
    
    timestamped_name = f"{base_name}_{timestamp}{extension}"
    return os.path.join(base_dir, timestamped_name)


def get_persona_files_in_range(base_dir, base_name, start_idx=-1, end_idx=-1):
    """
    Get persona files within a specified index range.
    
    Args:
        base_dir (str): Directory containing persona files
        base_name (str): Base filename pattern
        start_idx (int): Starting persona index (-1 means all)
        end_idx (int): Ending persona index (-1 means all)
    
    Returns:
        list: List of matching file paths
    """
    import glob
    
    # Create pattern to match persona files
    pattern = os.path.join(base_dir, f"{base_name}_*_persona*.json")
    all_files = glob.glob(pattern)
    
    if start_idx == -1 and end_idx == -1:
        return sorted(all_files)

    if end_idx == -1:
        end_idx = len(all_files)
    
    # Filter files by persona index
    filtered_files = []
    for file_path in all_files:
        # Extract persona index from filename
        filename = os.path.basename(file_path)
        match = re.search(r'_persona(\d+)\.json', filename)
        if match:
            persona_idx = int(match.group(1))
            if start_idx <= persona_idx <= end_idx:
                filtered_files.append(file_path)
    
    return sorted(filtered_files)