# Original path in verl: verl/verl/utils/reward_score/gsm8k.py

"""Reward scoring for ImplicitPersona dataset evaluation."""

import re
from typing import Union, Dict, Any
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

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


def compute_answer_similarity(model_response: str, correct_answer: str) -> float:
    """
    Compute cosine similarity between model response and correct answer using sentence embeddings.
    
    Args:
        model_response (str): The model's response
        correct_answer (str): The correct answer
        
    Returns:
        float: Cosine similarity score between 0.0 and 1.0
    """
    if not model_response or not correct_answer:
        return 0.0
    
    model = get_embedding_model()
        
    # Generate embeddings for both texts
    embeddings = model.get_sentence_embeddings(texts=[model_response, correct_answer])
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    similarity = float(similarity_matrix[0][0])

    # Ensure the similarity is between 0 and 1
    return max(0.0, min(1.0, similarity))


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
    judge_prompt = f"""Evaluate if this model response is personalized to the user's preference.

User's Ground-truth Preference: {groundtruth_preference}
Model's Response: {model_response}

Does the model's response acknowledge and align with the user's ground-truth preference above? 

Provide your reasoning, then give your final score from 0.0 to 1.0 in this format: \\boxed{{score}}"""

    try:
        judge_response = judge_model.query_llm(
            action="preference_judgment",
            prompt=judge_prompt,
        )
        
        # Extract numerical score from response
        # First try to find score in \boxed{} format
        boxed_match = re.search(r'\\boxed\{(\d+\.?\d*)\}', judge_response.strip())
        if boxed_match:
            score = float(boxed_match.group(1))
            return max(0.0, min(1.0, score))
        
        # Fallback to finding score in [score] format
        bracket_match = re.search(r'\[(\d+\.?\d*)\]', judge_response.strip())
        if bracket_match:
            score = float(bracket_match.group(1))
            return max(0.0, min(1.0, score))
        
        # Enforce format rewards
        return 0.0
            
    except Exception as e:
        print(f"Error in LLM judge evaluation: {e}")
        return 0.0


def compute_score(solution_str, ground_truth, method="embed", score=0.0, extra_info=None):
    """
    Compute the reward score for an ImplicitPersona solution.
    
    Args:
        solution_str (str): The generated solution/response from the model
        ground_truth (Dict[str, Any]): Dictionary containing 'groundtruth_preference' and 'correct_answer'
        method (str): Scoring method - "embed", "judge", or "hybrid"
        score (float): Legacy parameter, not used in new implementation
        
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

    # Extract ground truth information
    groundtruth_preference = ground_truth.get('groundtruth_preference', '')
    correct_answer = ground_truth.get('correct_answer', '')

    if not groundtruth_preference and not correct_answer:
        return 0.0

    # Extract the final answer from the solution string
    solution_clean = extract_solution(solution_str)
    
    if method == "embed":
        # Option 1: Sentence similarity with correct answer
        if correct_answer:
            similarity_score = compute_answer_similarity(solution_clean, correct_answer)
            return similarity_score
        else:
            return 0.0
            
    elif method == "judge":
        # Option 2: LLM judge for preference alignment
        if groundtruth_preference:
            # We need persona information for the judge, but it's not directly available here
            # For now, use a simplified version
            persona_info = {"preference": groundtruth_preference}
            judge_score = judge_preference_alignment(solution_clean, groundtruth_preference, persona_info)
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
            similarity_score = compute_answer_similarity(solution_clean, correct_answer)
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

