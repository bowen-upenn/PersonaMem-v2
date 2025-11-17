#!/usr/bin/env python3
"""
Utility functions for LLM-as-a-judge evaluation in ImplicitPersona benchmark.
Contains judge prompts and evaluation functions for open-ended responses.
"""

import json
import os
from typing import Dict, List, Any, Tuple


JUDGE_PROMPT_NARROW_POSITIVE = """You are evaluating an AI assistant's personalization capabilities in open-ended conversations.

User's query:
{user_query}

Ground truth user preference: {preference}

Model's response to evaluate:
{model_response}

Task: Evaluate how well the model personalizes its response according to the user's ground-truth preference. We expect the model to demonstrate awareness of this preference and provide a response that is specifically aligned with or relevant to it, rather than giving a generic answer that could apply to anyone.

A well-personalized response should:
- Acknowledge or incorporate the specific preference mentioned
- Tailor recommendations, advice, or information to align with this preference
- Show that the model remembers and respects the user's stated preference

A poorly personalized response:
- Ignores the preference entirely and gives generic advice
- Explicitly states it doesn't remember or track preferences
- Provides suggestions that contradict the stated preference

Give a score from 0.0 (completely irrelevant to or contradicts the preference) to 1.0 (explicitly and thoughtfully personalized to the preference).

Provide your reasoning first, then give your final score as \\boxed{{score}} where score is a number between 0.0 and 1.0.
Your response:"""

JUDGE_PROMPT_NARROW_NEGATIVE = """You are evaluating an AI assistant's personalization capabilities in open-ended conversations.

User's query:
{user_query}

Ground truth user preference: {preference}

Model's response to evaluate:
{model_response}

In this case, the user's ground-truth preference is a "do not personalize" or "avoid" instruction about a certain topic or attribute. The user does not want the assistant to tailor its response based on that specific preference, and instead expects a more general or neutral answer that does not rely on or mention that preference.

Task: Evaluate how well the model respects this preference. The model should avoid using or referencing the sensitive preference in a way that personalizes the response, while still providing a helpful and relevant answer.

Give a score from 0.0 (clearly personalizes using the forbidden preference or ignores the instruction) to 1.0 (fully respects the instruction to not personalize while remaining helpful).

Provide your reasoning first, then give your final score as \\boxed{{score}} where score is a number between 0.0 and 1.0.
Your response:"""

# JUDGE_PROMPT_BROAD = """You are evaluating whether an AI assistant's response is aligned with the user's preference profile.

# User's query:
# {user_query}

# User's complete preference profile
# {all_preferences}

# Model's response to evaluate:
# {model_response}

# Task: Check if the model's response follows or aligns with one or more preferences from the profile.
# Give a score from 0.0 (violates any one of the preferences) to 1.0 (perfectly aligns with one or more preferences with no violations).

# Provide your reasoning first, then give your final score as \\boxed{{score}} where score is a number between 0.0 and 1.0.
# Your response:"""


def load_persona_preferences(persona_file_path: str) -> Dict[str, Any]:
    """Load all preferences from a persona file."""
    if not persona_file_path or not os.path.exists(persona_file_path):
        return {}
    
    try:
        with open(persona_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Extract all preference types
            preferences = {}
            pref_keys = [
                "stereotypical_preferences",
                "anti_stereotypical_preferences", 
                "neutral_preferences",
                "therapy_background",
                "health_and_medical_conditions"
            ]
            
            # Get preference updates if they exist
            preference_updates = data.get("preference_updates", {})
            
            for key in pref_keys:
                if key in data:
                    pref_list = data[key]
                    
                    if isinstance(pref_list, list):
                        # Filter out preferences that have updates
                        updated_prefs = []
                        for pref in pref_list:
                            # Check if this preference is being updated
                            if pref in preference_updates:
                                # Add the updated value instead
                                updated_prefs.append(preference_updates[pref])
                            else:
                                # Keep the original preference
                                updated_prefs.append(pref)
                        preferences[key] = updated_prefs
                    else:
                        # If it's not a list, keep as is
                        preferences[key] = pref_list
            
            return preferences
    except Exception as e:
        print(f"Error loading persona file {persona_file_path}: {e}")
        return {}


def format_all_preferences(preferences: Dict[str, Any]) -> str:
    """Format all preferences into a readable string."""
    lines = []
    for key, value in preferences.items():
        formatted_key = key.replace('_', ' ').title()
        lines.append(f"{formatted_key}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                lines.append(f"  - {subkey}: {subvalue}")
        elif isinstance(value, list):
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"  {value}")
    return "\n".join(lines)


def extract_judge_decision(response: str) -> float:
    """Extract numeric score from judge response."""
    if not response:
        return 0.0
    
    import re
    
    # Look for boxed format first (most reliable)
    boxed_patterns = [
        r'\\boxed\{([0-9]*\.?[0-9]+)\}',
        r'\$\\boxed\{([0-9]*\.?[0-9]+)\}\$',
        r'\\boxed\s*\{([0-9]*\.?[0-9]+)\}',
    ]
    
    for pattern in boxed_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                score = float(match.group(1))
                # Clamp score between 0.0 and 1.0
                return max(0.0, min(1.0, score))
            except ValueError:
                continue
    
    # Fallback: look for standalone decimal number that looks like a score
    score_patterns = [
        r'score[:\s]+([0-9]*\.?[0-9]+)',
        r'rating[:\s]+([0-9]*\.?[0-9]+)',
        r'([0-9]*\.[0-9]+)\s*/\s*1\.?0?',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
    
    # Default to 0.0 if unclear
    print(f"    Warning: Could not extract clear score from judge response: {response[:100]}...")
    return 0.0


def average_score(scores: List[float]) -> float:
    """Calculate average score from judge scores."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def evaluate_narrow_judge(row: Dict[str, Any], model_response: str, 
                         query_llm_func, load_chat_history_func) -> Tuple[float, str]:
    """Evaluate with narrow judge (3 LLMs, average score)."""
    # Extract content from user_query if it's a dict
    user_query = row.get('user_query', '')
    if isinstance(user_query, str):
        try:
            import ast
            user_query_dict = ast.literal_eval(user_query)
            if isinstance(user_query_dict, dict) and 'content' in user_query_dict:
                user_query = user_query_dict['content']
        except:
            pass  # Keep as string if parsing fails
    
    # Determine if this is a negative preference (avoidance)
    preference = row.get('preference', '')
    is_negative = preference.lower().startswith('do not')
    
    # Select appropriate prompt template
    prompt_template = JUDGE_PROMPT_NARROW_NEGATIVE if is_negative else JUDGE_PROMPT_NARROW_POSITIVE
    
    prompt = prompt_template.format(
        user_query=user_query,
        preference=preference,
        model_response=model_response
    )
    
    # Query 3 judges
    judge_responses = []
    scores = []
    
    for i in range(1):
        # print(f"    Querying narrow judge {i+1}/3...")
        # Pass prompt as a simple string instead of message list to avoid multimodal API issues
        response = query_llm_func(prompt, use_history=False)
        score = extract_judge_decision(response)
        judge_responses.append(f"Judge {i+1} (score: {score:.2f}): {response}")
        scores.append(score)
        # print('=== PROMPT ===')
        # print(prompt)
        # print('\n=== RESPONSE ===')
        # print(response)
    
    final_score = average_score(scores)
    combined_response = "\n\n".join(judge_responses)
    
    return final_score, combined_response


def evaluate_broad_judge(row: Dict[str, Any], model_response: str, 
                        query_llm_func) -> Tuple[float, str]:
    """Evaluate with broad judge (3 LLMs, average score)."""
    # Load all preferences from persona file
    persona_file = row.get('raw_persona_file', '')
    all_preferences = load_persona_preferences(persona_file)
    preferences_text = format_all_preferences(all_preferences)
    
    # Extract content from user_query if it's a dict
    user_query = row.get('user_query', '')
    if isinstance(user_query, str):
        try:
            import ast
            user_query_dict = ast.literal_eval(user_query)
            if isinstance(user_query_dict, dict) and 'content' in user_query_dict:
                user_query = user_query_dict['content']
        except:
            pass  # Keep as string if parsing fails
    
    prompt = JUDGE_PROMPT_BROAD.format(
        user_query=user_query,
        all_preferences=preferences_text,
        model_response=model_response
    )
    
    # Query 3 judges
    judge_responses = []
    scores = []
    
    for i in range(3):
        # print(f"    Querying broad judge {i+1}/3...")
        # Pass prompt as a simple string instead of message list to avoid multimodal API issues
        response = query_llm_func(prompt, use_history=False)
        score = extract_judge_decision(response)
        judge_responses.append(f"Judge {i+1} (score: {score:.2f}): {response}")
        scores.append(score)
    
    final_score = average_score(scores)
    combined_response = "\n\n".join(judge_responses)
    
    return final_score, combined_response
