#!/usr/bin/env python3
"""
Utility functions for LLM-as-a-judge evaluation in ImplicitPersona benchmark.
Contains judge prompts and evaluation functions for open-ended responses.
"""

import json
import os
from typing import Dict, List, Any, Tuple


JUDGE_PROMPT_NARROW = """You are evaluating whether an AI assistant's response correctly addresses a user's personalized preferences based on their conversation history.

Context from conversation:
{related_conversation_snippet}

User's query:
{user_query}

Ground truth user preference: {preference}

Example correct answer: {correct_answer}

Model's response to evaluate:
{model_response}

Task: Determine if the model's response correctly personalizes the answer according to the user's preference shown in the ground truth. The response doesn't need to be identical to the example answer, but it should demonstrate understanding and incorporation of the user's preference.

Important: If the model's response goes AGAINST or contradicts the ground truth preference, you MUST answer No.

Provide your reasoning first, then give your final answer as \\boxed{{Yes}} or \\boxed{{No}}.
Your response:"""

JUDGE_PROMPT_BROAD = """You are evaluating whether an AI assistant's response correctly addresses a user's personalized preferences.

User's query:
{user_query}

User's complete preference profile:
{all_preferences}

Model's response to evaluate:
{model_response}

Task: Determine if the model's response correctly relates to and incorporates at least one of the user's preferences from their profile to provide a personalized response.

Important: If the model's response goes AGAINST or contradicts ANY of the user's preferences in their profile, you MUST answer No.

Provide your reasoning first, then give your final answer as \\boxed{{Yes}} or \\boxed{{No}}.
Your response:"""


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


def extract_judge_decision(response: str) -> bool:
    """Extract yes/no decision from judge response."""
    if not response:
        return False
    
    import re
    
    # Look for boxed format first (most reliable)
    boxed_patterns = [
        r'\\boxed\{(yes|no)\}',
        r'\$\\boxed\{(yes|no)\}\$',
        r'\\boxed\s*\{(yes|no)\}',
    ]
    
    for pattern in boxed_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
            return answer == 'yes'
    
    # Fallback: look for yes/no at the beginning of the response
    response_lower = response.lower().strip()
    if response_lower.startswith('yes'):
        return True
    elif response_lower.startswith('no'):
        return False
    
    # Second fallback: check for yes/no in first sentence
    first_sentence = response.split('.')[0].lower()
    if 'yes' in first_sentence and 'no' not in first_sentence:
        return True
    elif 'no' in first_sentence and 'yes' not in first_sentence:
        return False
    
    # Default to False if unclear
    print(f"    Warning: Could not extract clear yes/no from judge response: {response[:100]}...")
    return False


def majority_vote(decisions: List[bool]) -> bool:
    """Take majority vote from judge decisions."""
    return sum(decisions) > len(decisions) / 2


def evaluate_narrow_judge(row: Dict[str, Any], model_response: str, 
                         query_llm_func, load_chat_history_func) -> Tuple[bool, str]:
    """Evaluate with narrow judge (3 LLMs, majority vote)."""
    # Get related conversation snippet - use a portion of chat history
    chat_history_path = row.get('chat_history_32k_link', row.get('chat_history_link', ''))
    related_snippet = ""
    if chat_history_path and os.path.exists(chat_history_path):
        chat_history = load_chat_history_func(chat_history_path)
        # Take last 5 messages as context
        recent_messages = chat_history[-5:] if len(chat_history) > 5 else chat_history
        related_snippet = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
                                    for msg in recent_messages])
    
    prompt = JUDGE_PROMPT_NARROW.format(
        related_conversation_snippet=related_snippet,
        user_query=row.get('user_query', ''),
        preference=row.get('preference', ''),
        correct_answer=row.get('correct_answer', ''),
        model_response=model_response
    )
    
    # Query 3 judges
    judge_responses = []
    decisions = []
    
    for i in range(3):
        print(f"    Querying narrow judge {i+1}/3...")
        messages = [{"role": "user", "content": prompt}]
        response = query_llm_func(messages, use_history=False)
        judge_responses.append(f"Judge {i+1}: {response}")
        decision = extract_judge_decision(response)
        decisions.append(decision)
    
    final_decision = majority_vote(decisions)
    combined_response = "\n\n".join(judge_responses)
    
    return final_decision, combined_response


def evaluate_broad_judge(row: Dict[str, Any], model_response: str, 
                        query_llm_func) -> Tuple[bool, str]:
    """Evaluate with broad judge (3 LLMs, majority vote)."""
    # Load all preferences from persona file
    persona_file = row.get('raw_persona_file', '')
    all_preferences = load_persona_preferences(persona_file)
    preferences_text = format_all_preferences(all_preferences)
    
    prompt = JUDGE_PROMPT_BROAD.format(
        user_query=row.get('user_query', ''),
        all_preferences=preferences_text,
        model_response=model_response
    )
    
    # Query 3 judges
    judge_responses = []
    decisions = []
    
    for i in range(3):
        print(f"    Querying broad judge {i+1}/3...")
        messages = [{"role": "user", "content": prompt}]
        response = query_llm_func(messages, use_history=False)
        judge_responses.append(f"Judge {i+1}: {response}")
        decision = extract_judge_decision(response)
        decisions.append(decision)
    
    final_decision = majority_vote(decisions)
    combined_response = "\n\n".join(judge_responses)
    
    return final_decision, combined_response
