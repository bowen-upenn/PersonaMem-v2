def compute_score(solution_str, ground_truth: dict) -> float: 
    """
    For implicit_persona, ground_truth is a dict containing:
    - correct_answer: str in format "(a) some text"
    - all_answers: list of answer options
    - etc.
    """
    
    # Extract correct answer from ground_truth dict
    if isinstance(ground_truth, dict):
        correct_answer = ground_truth.get("correct_answer", "")
    elif isinstance(ground_truth, list) and len(ground_truth) > 0:
        # Fallback for backward compatibility
        correct_answer = ground_truth[0] if ground_truth else ""
    else:
        correct_answer = str(ground_truth) if ground_truth else ""
    
    # Extract the correct letter from format "(a) some text"
    correct_letter = extract_letter_from_answer(correct_answer)
    
    retval = 0.
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            generated_answer = remove_boxed(string_in_last_boxed)
            if is_equiv(generated_answer, correct_letter):
                retval = 1.
    except Exception as e:
        print(f"Error in compute_score: {e}")
    
    return retval


def extract_letter_from_answer(answer_text):
    """
    Extract the letter from format "(a) some text" -> "a"
    """
    answer_text = str(answer_text).strip()
    
    # Look for pattern like "(a)" at the beginning
    if answer_text.startswith("(") and len(answer_text) > 2 and answer_text[2] == ")":
        return answer_text[1].lower()  # Extract the letter and convert to lowercase
    
    # Fallback: return the first character if it's a letter
    if answer_text and answer_text[0].isalpha():
        return answer_text[0].lower()
    
    return ""


# string normalization adapted for MCQ answers (a, b, c, d)
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(f"Comparing: '{ss1}' == '{ss2}'")
        return ss1 == ss2
    except Exception:
        # Fallback to direct string comparison
        s1 = str(str1).strip().lower()
        s2 = str(str2).strip().lower()
        if verbose:
            print(f"Fallback comparing: '{s1}' == '{s2}'")
        return s1 == s2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def strip_string(string):
    # remove spaces and normalize for MCQ answers
    string = string.replace(" ", "")
    
    # normalize common variations
    string = string.replace("\n", "")
    
    # convert to lowercase for case-insensitive comparison
    string = string.lower()
    
    return string