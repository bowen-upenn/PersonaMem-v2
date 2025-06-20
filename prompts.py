import random

def expand_persona(persona_str):
    prompt = f"""
    Given this persona, add name and other detailed demographic information in a JSON format. Make it detailed and comprehensive: {persona_str}
    """
    return prompt


def generate_stereotypical_preferences():
    prompt = f"""
    Given this demographic information, propose 20 **overly** stereotypical preferences of this person, on-purposely. 
    Those stereotypical preferences should match the generic population mean of this person's demographic information, 
    but not necessary the current individual. Focus on demographic-related biases. 
    Add them to the JSON file under the key "stereotypical_preferences" whose value is a list of strings. Let us think step by step and output the full JSON in the end.
    """
    return prompt


def generate_anti_stereotypical_preferences():
    prompt = f"""
    Please continue to propose 20 **overly** anti-stereotypical preferences of the same person, i.e., personal preference of this individual 
    that is the opposite of the generic population mean of their demographic groups. Focus on demographic biases and find their opposites. 
    **Must avoid conflicts with previous stereotypical preferences of the same person.** 
    Add them to the JSON file under the key "anti_stereotypical_preferences" whose value is a list of strings. Let us think step by step and output the **full** JSON in the end.
    """
    return prompt


def verify_conflicts():
    prompt = f"""
    Verify if there are (1) any conflicts between stereotypical_preferences and anti_stereotypical_preferences. 
    (2) any redundant or repetitive preferences within each one of the two lists.
    If so, replace conflict ones and remove redundant ones. Show the full JSON file in the end.
    """
    return prompt


def generate_conversations(persona, preference, type, is_others_pref=False):
    who = "the user" if is_others_pref else "this person"
    prompt = f"""
    Given {who}'s persona and preference:

        Persona: "{persona}".

        Preference: "{preference}".
    """

    if type == 'email' or type == 'creative_writing' or type == 'professional_writing' or type == 'chat_message':
        if type == 'professional_writing':
            type = 'professional writing related to their work'
        if type == 'email' and is_others_pref:
            prompt += f"""
            Think about if the owner if this persona and preference can somehow implicitly mention this preference in an {type}. 
            Please pick a random name as if they are the owner of this {type} send to this user (check the user's name in the user persona above). 
            Please write such {type}, the user query to the model to explain this {type} (must be appended at the end of the original {type} as a single user turn), and an explanations. 
            The user request to explain the {type} received should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
            """
        else:
            whose = "the user's first perspective" if is_others_pref else f"a third person's perspective and pick a random name as the author of this {type}, such that this {type} is not written by this user"
            prompt += f"""
            Think about if the user can implicitly mention this preference when they ask ChatGPT to help improve the language in this {type}, and in the original {type}, the user somehow includes this information. 
            The {type} should use {whose}. Please write such {type}, the user query to the model to refine this {type} (must be appended at the end of the original {type} as a single user turn), and the refined {type}. 
            The user request to refine the {type} should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
            """
    elif type == 'translation':
        random_languages = random.choice(['English', 'Hindi', 'Chinese', 'Japanese', 'Korean', 'French', 'Germany', 'Spanish', 'Arabic', 'Vietnamese', 'Italian', 'Thai', 'Portuguese', 'Hebrew', 'Ukrainian'])
        if is_others_pref:
            prompt += f"""
            Think about if the user can implicitly mention this preference when they ask ChatGPT to help translate in a {type} written by others from {random_languages} into their native language. 
            If these two languages are the same, just choose a different source language yourself, without saying it in the formatted output.
            In the original {type}, the user somehow includes this preference information, and mention where the user found this piece of {type}.
            Please write such {type}, the user query to the model to translate this {type} (must be appended at the end of the original {type} as a single user turn), and the translated {type}. 
            The user request to translate the {type} should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
            """
        else:
            prompt += f"""
            First please figure out the native language of this person. Next,
            think about if the user can implicitly mention this preference when they ask ChatGPT to help translate in their own {type} written in their native language to {random_languages} for other readers.
            If these two languages are the same, just choose a different target language yourself, without saying it in the formatted output.
            In the original {type}, the user somehow includes this preference information, and mention that this is written by the user themselves.
            Please write such {type}, the user query to the model to translate this {type} (must be appended at the end of the original {type} as a single user turn), and the translated {type}. 
            The user request to translate the {type} should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
            """
    else:
        raise ValueError(f"Unknown type {type}")

    prompt += f"""
    Think step by step, and after that, return the conversation after special tokens '###Output' using a list of dictionaries using the OpenAI dict format in **JSON**, with keys:
    - "role": either "user" or "assistant"
    - "content": the actual utterance

    The format should be:

    ###Output
    ```json
    [
      {{"role": "user", "content": "..." }},
      {{"role": "assistant", "content": "..." }},
    ]
    ```
    """
    return prompt


def guess_persona(preference, anti=False):
    label = "anti-" if anti else ""
    prompt = f"""
    You see this {label}stereotypical preference:

        "{preference}"

    What single, concise user persona label would most likely go with that {label}stereotypical preference?  
    """
    return prompt


def check_alignment_with_population_mean(persona):
    prompt = f"""
    Given this actual user persona:

        "{persona}"

    Do you think your previous guess is roughly aligned with or fit this actual user persona? 
    Answer yes or no in the end after special tokens ####Final Answer.
    """
    return prompt


def generate_therapy_related_history():
    prompt = f"""
    Given the persona and preferences above, propose 20 personal histories of this person that might result in this person seeking AI chatbot for therapy consultations around them in the future. 
    Be very specific and personal. Add them to the JSON file under the key "therapy_background" whose value is a list of strings. 
    Let us think step by step and output the full JSON in the end.
    """
    return prompt