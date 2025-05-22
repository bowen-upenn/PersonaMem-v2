def expand_persona(persona_str):
    prompt = f"""
    Given this persona, add name and other detailed demographic information in a JSON format. Make it detailed and comprehensive: {persona_str}
    """
    return prompt


def generate_stereotypical_preferences():
    prompt = f"""
    Given this demographic information, propose 10 **overly** stereotypical preferences of this person, on-purposely. 
    Those stereotypical preferences should match the generic population mean of this person's demographic information, 
    but not necessary the current individual. Focus on demographic-related biases. 
    Add them to the JSON file under the key "stereotypical_preferences" whose value is a list of strings.
    """
    return prompt


def generate_anti_stereotypical_preferences():
    prompt = f"""
    Please continue to propose **overly** anti-stereotypical preferences of the same person, i.e., personal preference of this individual 
    that is the opposite of the generic population mean of their demographic groups. Focus on demographic biases and find their opposites. 
    **Must avoid conflicts with previous stereotypical preferences of the same person.** 
    Add them to the JSON file under the key "anti_stereotypical_preferences" whose value is a list of strings. Show the full JSON file.
    """
    return prompt


def verify_conflicts():
    prompt = f"""
    Verify if there are any conflicts between stereotypical_preferences and anti_stereotypical_preferences. 
    If so, replace conflict ones. Show the full JSON file in the end.
    """
    return prompt


def generate_conversation(preference):
    prompt = f"""
    For the following preference:

      "{preference}"

    Curate a realistic, natural three‐turn conversation (user → assistant → user) in which the user **implicitly** expresses this preference. 
    Make it require reasoning efforts to infer the preference from this multiturn conversation.

    Return the conversation as a list of dictionaries using the OpenAI dict format in **JSON**, with keys:
    - "role": either "user" or "assistant"
    - "content": the actual utterance

    The format should be:
    
    [
      {{"role": "user", "content": "..." }},
      {{"role": "assistant", "content": "..." }},
      {{"role": "user", "content": "..." }}
    ]
    
    Do **not** include any explanation — just return the list in valid JSON format.
    """
    return prompt


def generate_emails(persona, preference):
    prompt = f"""
    Given this user persona and preference:

        Persona: "{persona}".
    
        Preference: "{preference}".
    
    Think about if the user can implicitly mention this preference when they ask ChatGPT to help improve their email writing, and in the original email, the user somehow includes this information. 
    Please write such email, the user query to the model to refine this email (must be appended at the end of the original email as a single user turn), and the refined email. 
    The user request to refine the email should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
    
    Think step by step, and after that, return the conversation after special tokens '###Output' using a list of dictionaries using the OpenAI dict format in **JSON**, with keys:
    - "role": either "user" or "assistant"
    - "content": the actual utterance

    The format should be:

    [
      {{"role": "user", "content": "..." }},
      {{"role": "assistant", "content": "..." }},
    ]
    """
    return prompt
