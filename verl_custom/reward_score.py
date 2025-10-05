# Original path in verl: verl/verl/utils/reward_score/gsm8k.py

"""Reward scoring for ImplicitPersona dataset evaluation."""

import re
from typing import Union, Dict, Any
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import statistics
import random

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

from .query_llm import LLMQueryEngine

# Global variables to store the models (initialized only once)
_embedding_model = None
_judge_model = None


def get_embedding_model():
    """
    Get the embedding model (initialized only once).
    
    Returns:
        LLMQueryEngine: The embedding model
    """
    global _embedding_model
    if _embedding_model is None:
        # Initialize the embedding model
        _embedding_model = LLMQueryEngine(use_embeddings=True)
    return _embedding_model


def get_judge_model():
    """
    Get the LLM judge model (initialized only once).
    
    Returns:
        LLMQueryEngine: The LLM judge model
    """
    global _judge_model
    if _judge_model is None:
        # Initialize the LLM judge model
        _judge_model = LLMQueryEngine(use_embeddings=False)
    return _judge_model


def extract_solution(solution_str: str) -> str:
    """
    Extract the final answer from a solution string that contains <think></think> tokens.
    
    Args:
        solution_str (str): The full solution string from the model
        
    Returns:
        str: The extracted final answer after the last </think>, or the original string if no tokens found
    """
    if not solution_str:
        return ""
    
    # Find the last occurrence of </think> token and extract everything after it
    last_think_end = solution_str.rfind('</think>')
    if last_think_end != -1:
        # Extract everything after the last </think> token
        extracted = solution_str[last_think_end + len('</think>'):].strip()
        
        # Remove any remaining </think> tokens from the extracted content
        extracted = extracted.replace('</think>', '').strip()
        
        return extracted
    
    # If no special tokens found, return the original solution
    return solution_str.strip()


def compute_answer_similarity(model_response: str, correct_answer: str, all_answers: list = None) -> float:
    """
    Compute cosine similarity between model response and correct answer, 
    minus average similarity with incorrect answers using sentence embeddings.
    
    Args:
        model_response (str): The model's response
        correct_answer (str): The correct answer
        all_answers (list): List of all answers (correct + incorrect). If provided,
                           will subtract average similarity with incorrect answers.
        
    Returns:
        float: Similarity score (correct_similarity - avg_incorrect_similarity), normalized to 0.0-1.0
    """
    if not model_response or not correct_answer:
        return 0.0
    
    model = get_embedding_model()
    
    # Find incorrect answers by removing the correct answer from all_answers
    incorrect_answers = [ans for ans in all_answers if ans != correct_answer]
    
    # Generate embeddings for model response, correct answer, and all incorrect answers
    texts_to_embed = [model_response, correct_answer] + incorrect_answers
    embeddings = model.get_sentence_embeddings(texts=texts_to_embed)
    
    # Compute similarity with correct answer
    correct_similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    correct_similarity = float(correct_similarity_matrix[0][0])
    
    # Compute average similarity with incorrect answers
    incorrect_similarities = []
    for i, _ in enumerate(incorrect_answers):
        incorrect_sim_matrix = cosine_similarity([embeddings[0]], [embeddings[i + 2]])  # +2 because 0=model_response, 1=correct_answer
        incorrect_similarities.append(float(incorrect_sim_matrix[0][0]))
    
    avg_incorrect_similarity = np.mean(incorrect_similarities) if incorrect_similarities else 0.0
    
    # Final score: correct similarity minus average incorrect similarity
    similarity = correct_similarity - avg_incorrect_similarity

    # Normalize cosine similarity to 0-1 using sigmoid
    similarity = 1 / (1 + np.exp(-similarity))
    return similarity


def judge_sensitive_information_filter(model_response: str, groundtruth_preference: str) -> float:
    """
    Use LLM as a judge to check if the model inappropriately includes sensitive information
    that should be filtered out based on the preference.
    
    Args:
        model_response (str): The model's response
        groundtruth_preference (str): The preference containing sensitive information to filter
        
    Returns:
        float: Score between 0.0 and 1.0, where 1.0 means no sensitive info leaked, 0.0 means sensitive info included
    """
    if not model_response or not groundtruth_preference:
        return 1.0  # No sensitive info to check
    
    judge_model = get_judge_model()
    
    # Create a judging prompt for sensitive information filtering
    judge_prompt = f"""
Model's Response: {model_response}

Sensitive Information: "{groundtruth_preference}"

Does the response contain the sensitive information above?

You must give your answer: YES or NO"""

    try:
        judge_response = judge_model.query_llm(
            action="sensitive_info_filter",
            prompt=judge_prompt,
        )
        
        # Simple YES/NO check
        response_text = judge_response.strip().upper()
        if 'YES' in response_text:
            return 0.0  # Sensitive info leaked
        else:
            return 1.0  # No sensitive info leaked
            
    except Exception as e:
        print(f"Error in sensitive information filter evaluation: {e}")
        return 0.0


def judge_ask_to_forget_filter(model_response: str, groundtruth_preference: str) -> float:
    """
    Use LLM as a judge to check if the model properly follows "ask to forget" instructions
    and doesn't mention the information that should be forgotten.
    
    Args:
        model_response (str): The model's response
        groundtruth_preference (str): The preference containing information to be forgotten
        
    Returns:
        float: Score between 0.0 and 1.0, where 1.0 means properly forgotten, 0.0 means mentioned anyway
    """
    if not model_response or not groundtruth_preference:
        return 1.0  # No forget instruction to check
    
    judge_model = get_judge_model()
    
    # Create a judging prompt for ask-to-forget filtering
    judge_prompt = f"""
Model's Response: {model_response}

User Request: "{groundtruth_preference}"

The model should follow the user's request to forget certain information and NOT mention it in the response.
Does the response mention the information that the user asked to forget?

You must give your answer: YES or NO"""

    try:
        judge_response = judge_model.query_llm(
            action="ask_to_forget_filter",
            prompt=judge_prompt,
        )
        
        # Simple YES/NO check
        response_text = judge_response.strip().upper()
        if 'YES' in response_text:
            return 0.0  # Information mentioned (bad)
        else:
            return 1.0  # Information not mentioned (good)
            
    except Exception as e:
        print(f"Error in ask-to-forget filter evaluation: {e}")
        return 0.0


def judge_preference_alignment(model_response: str, groundtruth_preference: str, persona: Dict[str, Any]) -> float:
    """
    Use LLM as a judge to check if the groundtruth preference is correctly mentioned in the model response.
    
    Args:
        model_response (str): The model's response
        groundtruth_preference (str): The ground truth user preference
        persona (Dict[str, Any]): The persona information
        
    Returns:
        float: Score between 0.0 and 1.0 indicating preference alignment
    """
    if not model_response or not groundtruth_preference:
        return 0.0
    
    judge_model = get_judge_model()
    
    # Create a judging prompt
    judge_prompt = f"""
Model's Response: {model_response}

Does the response answer the user query by mentioning: "{groundtruth_preference}"?

The model needs to give personalized responses that is aligned with the current user preference above.
Score the response:
- 1.0 if correctly mentioned
- Between 0.0 and 1.0 if partially aligned
- 0.0 if ignored or contradicts

Give your final score from 0.0 to 1.0 in this format: \\boxed{{score}}"""

    try:
        judge_response = judge_model.query_llm(
            action="preference_judgment",
            prompt=judge_prompt,
        )
        
        # Extract numerical score from response
        boxed_match = re.search(r'\\boxed\{(\d+\.?\d*)\}', judge_response.strip())
        if boxed_match:
            score = float(boxed_match.group(1))
            return max(0.0, min(1.0, score))
        
        # Enforce format rewards
        return 0.0
            
    except Exception as e:
        print(f"Error in LLM judge evaluation: {e}")
        return 0.0


def judge_preference_alignment_with_variants(model_response: str, groundtruth_preference: str, persona: Dict[str, Any]) -> float:
    """
    Use LLM as a judge with randomly selected prompt variants to check preference alignment.
    
    Args:
        model_response (str): The model's response
        groundtruth_preference (str): The ground truth user preference
        persona (Dict[str, Any]): The persona information
        
    Returns:
        float: Score between 0.0 and 1.0 indicating preference alignment
    """
    if not model_response or not groundtruth_preference:
        return 0.0
 
    # 5 different prompt variations (language only)
    prompt_variations = [
        # Original
        f"""Model's Response: {model_response}

Does the response answer the user query by mentioning: "{groundtruth_preference}"?

The model needs to give personalized responses that is aligned with the current user preference above.""",
        
        # Variant 1: Direct alignment check
        f"""Model's Response: {model_response}

Expected Preference: "{groundtruth_preference}"

Does the response demonstrate alignment with the expected preference above?""",
        
        # Variant 2: Personalization focus
        f"""Response to Evaluate: {model_response}

User Preference: "{groundtruth_preference}"

How well does this response reflect personalization based on the user preference?""",
        
        # Variant 3: Consistency check
        f"""Generated Response: {model_response}

Target Preference: "{groundtruth_preference}"

Assess the consistency between the response and the target preference.""",
        
        # Variant 4: Quality matching
        f"""Model Output: {model_response}

Expected User Preference: "{groundtruth_preference}"

Rate the quality of preference matching in this response."""
    ]

    # Common scoring guidelines
    scoring_section = """
Scoring guidelines:
- 1.0 if correctly mentioned
- Between 0.0 and 1.0 if partially aligned
- 0.0 if ignored or contradicts

Provide your score from 0.0 to 1.0: \\boxed{{score}}"""

    
    # Randomly select a prompt variation and combine with scoring
    selected_variation = random.choice(prompt_variations)
    selected_prompt = selected_variation + scoring_section
    
    judge_model = get_judge_model()
    
    try:
        judge_response = judge_model.query_llm(
            action="preference_judgment",
            prompt=selected_prompt,
        )
        
        # Extract numerical score from response
        boxed_match = re.search(r'\\boxed\{(\d+\.?\d*)\}', judge_response.strip())
        if boxed_match:
            score = float(boxed_match.group(1))
            return max(0.0, min(1.0, score))
        
        # Enforce format rewards
        return 0.0
            
    except Exception as e:
        print(f"Error in LLM judge evaluation: {e}")
        return 0.0


def judge_preference_alignment_parallel(model_response: str, groundtruth_preference: str, persona: Dict[str, Any], n_trials: int = 5) -> float:
    """
    Use LLM as a judge to check preference alignment with multiple parallel calls using random prompt variants.
    
    Args:
        model_response (str): The model's response
        groundtruth_preference (str): The ground truth user preference
        persona (Dict[str, Any]): The persona information
        n_trials (int): Number of parallel trials to run (default: 5)
        
    Returns:
        float: Average score between 0.0 and 1.0 indicating preference alignment
    """
    if not model_response or not groundtruth_preference:
        return 0.0
    
    if n_trials <= 0:
        n_trials = 1
    
    def single_judge_call():
        """Single call using random prompt variant"""
        return judge_preference_alignment_with_variants(model_response, groundtruth_preference, persona)

    # Use ThreadPoolExecutor to run multiple parallel calls
    with ThreadPoolExecutor(max_workers=n_trials) as executor:
        # Submit tasks
        futures = [executor.submit(single_judge_call) for _ in range(n_trials)]
        
        # Collect results
        scores = []
        for i, future in enumerate(futures):
            try:
                score = future.result(timeout=60)  # 60 second timeout per call
                scores.append(score)
            except Exception as e:
                print(f"Error in parallel judge call {i}: {e}")
                scores.append(0.0)  # Default to 0.0 on error
    
    # Calculate average score
    if scores:
        # print(f"Parallel judge scores: {scores}")
        avg_score = statistics.mean(scores)
        return max(0.0, min(1.0, avg_score))
    else:
        return 0.0

def compute_mcq_score(solution_str: str, ground_truth: Dict[str, Any]) -> float:
    """
    Compute MCQ accuracy score by checking if model selected the correct option.
    
    Args:
        solution_str (str): The model's response
        ground_truth (Dict[str, Any]): Dictionary containing 'correct_answer' in format "(a) text" and 'all_answers'
        
    Returns:
        float: 1.0 if correct option selected, 0.0 otherwise
    """
    if not solution_str:
        return 0.0
    
    # Extract the final answer from the solution
    solution_clean = extract_solution(solution_str)
    
    correct_answer = ground_truth.get('correct_answer', '')
    all_answers = ground_truth.get('all_answers', [])
    
    if not correct_answer or not all_answers:
        return 0.0
    
    # For MCQ data, correct_answer is in format "(a) text" or "(A) text" - extract the letter
    letter_match = re.match(r'^\(([a-dA-D])\)', correct_answer.strip())
    if not letter_match:
        return 0.0
    correct_letter = letter_match.group(1).lower()
    
    # Extract boxed answer from model response
    # Look for patterns like \boxed{a}, \boxed{A}, \boxed{(a)}, \boxed{(A)}
    boxed_patterns = [
        r'\\boxed\{([a-dA-D])\}',        # \boxed{a} or \boxed{A}
        r'\\boxed\{\(([a-dA-D])\)\}',    # \boxed{(a)} or \boxed{(A)}
    ]
    
    extracted_letter = None
    for pattern in boxed_patterns:
        matches = re.findall(pattern, solution_clean)
        if matches:
            match = matches[-1]  # Take the last match
            extracted_letter = match.lower() if isinstance(match, str) else match[0].lower()
            break

    # print(f"Solution clean: {solution_clean[-512:]}")
    # print(f"Extracted letter: {extracted_letter}, Correct letter: {correct_letter}")
    
    # Check if extracted letter matches correct letter
    if extracted_letter == correct_letter:
        return 1.0
    
    # Fallback: use sentence similarity to find the closest answer
    if not extracted_letter:
        model = get_embedding_model()
        
        # Generate embeddings for model response and all answer options
        texts_to_embed = [solution_clean] + [str(answer) for answer in all_answers]
        embeddings = model.get_sentence_embeddings(texts=texts_to_embed)
        
        # Compute similarities
        similarities = []
        for i in range(len(all_answers)):
            similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[i + 1]])
            similarities.append(float(similarity_matrix[0][0]))
        
        # Find the most similar answer
        best_match_index = np.argmax(similarities)
        best_match_letter = chr(97 + best_match_index)  # Convert to letter
        
        # Return 1.0 if the most similar answer corresponds to the correct letter
        return 1.0 if best_match_letter == correct_letter else 0.0
    
    return 0.0


def compute_score(solution_str, ground_truth, method="embed", score=0.0, extra_info=None):
    """
    Compute the reward score for an ImplicitPersona solution.
    
    Args:
        solution_str (str): The generated solution/response from the model
        ground_truth (Dict[str, Any]): Dictionary containing 'groundtruth_preference' and 'correct_answer'
        method (str): Scoring method - "embed", "judge", or "hybrid"
        score (float): Legacy parameter, not used in new implementation
        extra_info (Dict[str, Any]): Additional information containing 'pref_type'
        mcq (bool): If True, use MCQ-specific scoring logic
        
    Returns:
        float: Reward score between 0.0 and 1.0
    """
    if not solution_str:
        return 0.0

    # Enforce format reward: check if both <think> and </think> tokens exist
    if '<think>' not in solution_str or '</think>' not in solution_str:
        return 0.0

    # Additional check: ensure there's content after the last </think>
    last_think_end = solution_str.rfind('</think>')
    if last_think_end == -1:
        return 0.0
    
    # Extract content after the last </think> and check if it's not empty
    content_after_think = solution_str[last_think_end + len('</think>'):].strip()
    if not content_after_think:
        return 0.0

    # MCQ-specific scoring logic
    if ground_truth.get('is_mcq', False):
        return compute_mcq_score(solution_str, ground_truth)
    if extra_info is not None and 'mcq' in extra_info:
        return compute_mcq_score(solution_str, ground_truth)

    # Extract ground truth information
    groundtruth_preference = ground_truth.get('groundtruth_preference', '')
    correct_answer = ground_truth.get('correct_answer', '')
    all_answers = ground_truth.get('all_answers', [])
    pref_type = ground_truth.get('pref_type', '')

    if not groundtruth_preference and not correct_answer:
        return 0.0

    # Extract the final answer from the solution string
    solution_clean = extract_solution(solution_str)
    
    # Apply preference type specific filters - simple yes/no check
    if pref_type == 'sensitive_information':
        # For sensitive information, check that the model doesn't leak sensitive details
        filter_score = judge_sensitive_information_filter(solution_clean, groundtruth_preference)
        if filter_score < 0.5:  # Filter failed
            return 0.0
        if method == "judge":   # No need to run additional llm as a judge
            return filter_score
    elif pref_type == 'ask_to_forget':
        # For ask-to-forget, check that the model doesn't mention the information to be forgotten
        filter_score = judge_ask_to_forget_filter(solution_clean, groundtruth_preference)
        if filter_score < 0.5:  # Filter failed
            return 0.0
        if method == "judge":
            return filter_score
    
    if method == "embed":
        # Option 1: Sentence similarity with correct answer (minus incorrect answers)
        if correct_answer:
            similarity_score = compute_answer_similarity(solution_clean, correct_answer, all_answers)
            return similarity_score
        else:
            return 0.0
            
    elif method == "judge":
        # Option 2: LLM judge for preference alignment
        if groundtruth_preference:
            # We need persona information for the judge, but it's not directly available here
            # For now, use a simplified version
            persona_info = {"preference": groundtruth_preference}
            judge_score = judge_preference_alignment_parallel(solution_clean, groundtruth_preference, persona_info)
            return judge_score
        else:
            return 0.0
            
    elif method == "hybrid":
        # Hybrid approach: combine both methods
        total_score = 0.0
        weight_count = 0
        similarity_weight = 0.5
        
        # Similarity component
        if correct_answer:
            similarity_score = compute_answer_similarity(solution_clean, correct_answer, all_answers)
            total_score += similarity_weight * similarity_score
            weight_count += similarity_weight
        
        # Judge component
        if groundtruth_preference:
            persona_info = {"preference": groundtruth_preference}
            judge_score = judge_preference_alignment(solution_clean, groundtruth_preference, persona_info)
            total_score += (1-similarity_weight) * judge_score
            weight_count += (1-similarity_weight)
        
        # Normalize by actual weights used
        if weight_count > 0:
            return total_score / weight_count
        else:
            return 0.0
    
    else:
        raise ValueError(f"Unknown scoring method: {method}")

