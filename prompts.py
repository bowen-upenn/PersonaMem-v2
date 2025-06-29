import random

def expand_persona(persona_str):
    prompt = f"""
    Given this persona, add name and other detailed demographic information in a JSON format. Make it detailed and comprehensive: {persona_str}
    """
    return prompt


def generate_stereotypical_preferences():
    prompt = f"""
    Given this demographic information, propose 30 **overly** stereotypical preferences of this person, on-purposely. 
    Those stereotypical preferences should match the generic population mean of this person's demographic information, 
    but not necessary the current individual. Focus on demographic-related biases. 
    Add them to the JSON file under the key "stereotypical_preferences" whose value is a list of strings. Let us think step by step and output the full JSON in the end.
    """
    return prompt


def generate_anti_stereotypical_preferences():
    prompt = f"""
    Please continue to propose 30 **overly** anti-stereotypical preferences of the same person, i.e., personal preference of this individual 
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


def update_preference(pref):
    prompt = f"""
    Assume the user later changes their previous preference "{pref}" to its opposite. Please give me the new, opposite preference here. No additional text.
    """
    return prompt


def generate_conversations(persona, preference, type, is_others_pref=False):
    who = "the user" if is_others_pref else "this person"
    prompt = f"""
    Given {who}'s persona and preference:

        Persona: "{persona}".

        Preference: "{preference}".
    """

    if type == 'personal_email' or type == 'professional_email' or type == 'creative_writing' or type == 'professional_writing' or type == 'chat_message':
        if type == 'professional_writing':
            type = 'professional writing related to their work'
        if (type == 'personal_email' or type == 'professional_email') and is_others_pref:
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
        random_languages = random.choice(['Chinese', 'Japanese', 'Hindi', 'Korean', 'French', 'Germany', 'Spanish', 'Arabic', 'Vietnamese', 'Italian', 'Thai', 'Portuguese', 'Hebrew', 'Ukrainian'])
        target_language = 'English' if random.random() > 0.67 else 'their native language'
        if is_others_pref:
            prompt += f"""
            Think about if the user can implicitly mention this preference when they ask ChatGPT to help translate in a {type} written by others from {random_languages} into {target_language}. 
            If these two languages are the same, just choose a different source language yourself, without saying it in the formatted output.
            In the original {type}, the user somehow includes this preference information, and mention where the user found this piece of {type}.
            Please write such {type}, the user query to the model to translate this {type} (must be appended at the end of the original {type} as a single user turn), and the translated {type}. 
            The user request to translate the {type} should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
            """
        else:
            prompt += f"""
            First please figure out the native language of this person. Next,
            think about if the user can implicitly mention this preference when they ask ChatGPT to help translate in their {type} written in {target_language} to {random_languages} for other readers.
            If these two languages are the same, just choose a different target language yourself, without saying it in the formatted output.
            In the original {type}, the user somehow includes this preference information, and mention that this is written by the user themselves.
            Please write such {type}, the user query to the model to translate this {type} (must be appended at the end of the original {type} as a single user turn), and the translated {type}. 
            The user request to translate the {type} should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
            """
    elif type == 'trouble_consult':
        if is_others_pref:
            prompt += f"""
            Think about if the user can implicitly mention this preference when they consult ChatGPT about some troubles this user knows has in their lives. 
            The trouble topic can be diverse like relations, health, romantics, politics, family, study, work, safety, identity, personal character, philosophy, destiny, and etc, 
            and the person who experienced this trouble can be anyone this user knows, but not themselves. The user query needs to mention who has this concern.
            However, "{preference}" is NOT what troubles the person. The user should talk about this person having other concerns while unintentionally and naturally mentions this preference.
            Please write such user query, and the chatbot's answers. 
            The user query should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
            """
        else:
            prompt += f"""
            Think about if the user can implicitly mention this preference when they consult ChatGPT about some troubles the user has in their lives. 
            The trouble topic can be diverse like relations, health, romantics, politics, family, study, work, safety, identity, personal character, philosophy, destiny, and etc.
            However, "{preference}" is NOT what troubles the user. The user should talk about other concerns while unintentionally and naturally mentions this preference.
            Please write such user query, and the chatbot's answers. 
            The user query should be simple and realistic, without on purposely mentioning the specific preference we are testing here. 
            """
    elif type == 'knowledge_query':
        prompt += f"""
        Generate a random question that {who} might ask a chatbot, related to this preference. 
        The question should reflect a request for explanation or clarification of some detailed or nuanced knowledge of "{preference}, which should indicate some hidden curiosity implicitly.".
        Please write such user query and a high-quality, long, and detailed model response. The user query should be short, simple, and realistic.
        """
    else:
        raise ValueError(f"Unknown type {type}")

    prompt += f"""
    **Importantly, please make the user preference implicit and requires some reasoning to interpret.**
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


def generate_user_question(element):
    """
    Generates a natural first-person user query that would *elicit* a personalized response
    based on the element's preference/background, but does NOT reference the preference/background.
    """
    if "stereotypical_pref" in element or "anti_stereotypical_pref" in element:
        if "stereotypical_pref" in element:
            context = f"Given this hidden ground-truth user preference: {element['stereotypical_pref']}"
        else:
            context = f"Given this hidden ground-truth user preference: {element['anti_stereotypical_pref']}"
        if 'idx_repeat' in element:
            context += "This user has previously asked some detailed questions related to this topic, which indicates some hidden interests."
        instruct = (
            """
            Write a first-person, natural question (for a chat with an assistant) 
            that could plausibly be asked by this user. 
            The question should invite recommendations, ideas, advice, or explanations. 
            It should be phrased so that a personalized answer would use this ground-truth preference specific to this user, 
            but the question itself should not mention or allude to it, so the preference remain hidden to the chatbot. 
            Do NOT refer to 'my preferences', 'what I like', 'recall my preference', 'recall your memory', or anything similar, 
            since we want to robustly evaluate the chatbot's actual ability to interpret and remember this user preferences, 
            and therefore, give this user unexpected personalized responses to enhance user engagement.
            Do NOT leak the user's hidden ground-truth preference in the user query to make the test useless.
            """
        )
    elif "therapy_background" in element:
        context = f"The user has a hidden personal background: {element['therapy_background']}"
        instruct = (
            """
            Write a first-person, natural question seeking therapeutic advice, coping strategies, or support.
            The question should be influenced by the user's background, but not mention or reference it, so the preference remain hidden to the chatbot.
            Do NOT refer to 'my preferences', 'what I like', 'recall my preference', 'recall your memory', or anything similar,
            since we want to robustly evaluate the chatbot's actual ability to interpret and remember this user preferences.
            and therefore, give this user unexpected personalized responses to enhance user engagement.
            Do NOT leak the user's hidden ground-truth preference in the user query to make the test useless.
            """
        )
    else:
        raise NotImplementedError("Unknown scenarios")

    return (
        f"{context}\n"
        f"{instruct}\n"
        f"Think step by step and return the user question after ###Output."
    )


def generate_answer_options(element, user_query, who):
    """
    Prompts the LLM to generate 4 answers to the user_query, all with the same wording/structure,
    but each reflecting a different use of personalization:
      - 'correct': personalized to the user's background or preference (contextually, not explicitly).
      - 'opposite': same structure, but uses the *opposite* background/preference.
      - 'random': uses a random background/preference unrelated to the user.
      - 'generic': a generic, non-personalized answer.
    All should be plausible, same length, and not leak the background directly.
    We always assume the preference belongs to the user themselves in this prompt,
    and will adjust this assumption in qa_generator.py.
    """
    # Construct backgrounds for each answer type
    if "stereotypical_pref" in element:
        user_bg = element["stereotypical_pref"]
    elif "anti_stereotypical_pref" in element:
        user_bg = element["anti_stereotypical_pref"]
    elif "therapy_background" in element:
        user_bg = element["therapy_background"]
    else:
        raise NotImplementedError("Unknown scenarios")

    if who == 'self':
        prompt = (
            f"Given this user background and preference: {user_bg}\n"
            f"User question: {user_query}\n\n"
            "You are creating a multiple-choice benchmark."
            "Generate four different, one-to-three sentence answers to the user's question, as follows:\n"
            "1. 'correct': The answer should be appropriately personalized to the user's background and preference.\n"
            "2. 'random': The answer should be identical in structure to 'correct' but a random preference.\n"
            "3. 'random': The answer should be identical in structure to 'correct' but another random preference.\n"
            "4. 'generic': The answer should be identical in structure but generic, suitable for anyone.\n\n"
            "Each answer must have the same tone and length. Be natural and realistic.\n"
            "Think step by step and return the final four answers in JSON format at the end:\n"
            "```json"
            "{\n"
            "  \"correct\": <answer>,\n"
            "  \"random1\": <answer>,\n"
            "  \"random2\": <answer>,\n"
            "  \"generic\": <answer>\n"
            "}\n"
            "```"
        )
    else:
        prompt = (
            f"Given this user background and preference: {user_bg}\n"
            f"User question: {user_query}\n\n"
            "You are creating a multiple-choice benchmark. We need to prepare more than one correct answers for diversity."
            "Generate four different, one-to-three sentence answers to the user's question, as follows:\n"
            "1. 'correct': The answer should be appropriately personalized to the user's background and preference.\n"
            "2. 'correct': The answer should mention the same 'correct' preference but rephrased.\n"
            "3. 'correct': The answer should mention the same 'correct' preference but rephrased in another way.\n"
            "4. 'incorrect': The answer should be identical in structure but generic, suitable for anyone.\n\n"
            "Each answer must have the same tone and length. Be natural and realistic.\n"
            "Think step by step and return the final four answers in JSON format at the end:\n"
            "```json"
            "{\n"
            "  \"correct1\": <answer>,\n"
            "  \"correct2\": <answer>,\n"
            "  \"correct3\": <answer>,\n"
            "  \"incorrect\": <answer>\n"
            "}\n"
            "```"
        )
    return prompt
