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


def judge_preference_alignment(model_response: str, groundtruth_preference: str, persona: Dict[str, Any]) -> float:
    """
    Use LLM as a judge with a focused prompt to check preference alignment.
    
    Args:
        model_response (str): The model's response
        groundtruth_preference (str): The ground truth user preference
        persona (Dict[str, Any]): The persona information (should contain 'preference' and optionally 'user_query')
        
    Returns:
        float: Score between 0.0 and 1.0 indicating preference alignment
    """
    if not model_response or not groundtruth_preference:
        return 0.0
 
    # Extract user query from persona if available, otherwise use a placeholder
    user_query = persona.get('user_query', '')
    
    # Determine if this is a negative preference (avoidance) - same logic as inference_utils.py
    is_negative = groundtruth_preference.lower().startswith('do not')
    
    # Focused prompt template for narrow positive personalization evaluation
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

Give your final score as \\boxed{{score}} where score is a number between 0.0 and 1.0.
Your response:"""
    
    # Prompt template for narrow negative personalization evaluation (avoidance preferences)
    JUDGE_PROMPT_NARROW_NEGATIVE = """You are evaluating an AI assistant's personalization capabilities in open-ended conversations.

User's query:
{user_query}

Ground truth user preference: {preference}

Model's response to evaluate:
{model_response}

In this case, the user's ground-truth preference is a "do not personalize" or "avoid" instruction about a certain topic or attribute. The user does not want the assistant to tailor its response based on that specific preference, and instead expects a more general or neutral answer that does not rely on or mention that preference.

Task: Evaluate how well the model respects this preference. The model should avoid using or referencing the sensitive preference in a way that personalizes the response, while still providing a helpful and relevant answer.

Give a score from 0.0 (clearly personalizes using the forbidden preference or ignores the instruction) to 1.0 (fully respects the instruction to not personalize while remaining helpful).

Give your final score as \\boxed{{score}} where score is a number between 0.0 and 1.0.
Your response:"""
    
    # Select appropriate prompt template based on preference type
    prompt_template = JUDGE_PROMPT_NARROW_NEGATIVE if is_negative else JUDGE_PROMPT_NARROW_POSITIVE
    
    # Format the prompt with actual values
    judge_prompt = prompt_template.format(
        user_query=user_query,
        preference=groundtruth_preference,
        model_response=model_response
    )
    
    judge_model = get_judge_model()
    
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


def judge_external_benchmark(model_response: str, groundtruth: str, benchmark_type: str, persona: Dict[str, Any] = None) -> float:
    """
    Simple LLM judge for external benchmarks (PrefEval and LongMemEval).
    
    Args:
        model_response (str): The model's response
        groundtruth (str): The ground truth preference/answer
        benchmark_type (str): Either 'prefeval' or 'longmemeval'
        persona (Dict[str, Any]): Unused, kept for API compatibility
        
    Returns:
        float: Score between 0.0 and 1.0
    """
    if not model_response or not groundtruth:
        return 0.0
    
    judge_model = get_judge_model()
    
    # Different prompts for different benchmarks
    if benchmark_type == 'prefeval':
        judge_prompt = f"""Model's Response: {model_response}

Ground Truth User Preference: {groundtruth}

Is the model's response aligned with the ground truth user preference above?

Scoring guidelines:
- 1.0 if fully aligned with the preference
- Between 0.0 and 1.0 if partially aligned
- 0.0 if completely misaligned or contradicts the preference

Think step by step and provide your score from 0.0 to 1.0 in the format: \\boxed{{score}}. Replace score with your numerical score."""
    
    elif benchmark_type == 'longmemeval':
        judge_prompt = f"""Model's Response: {model_response}

Ground Truth Answer: {groundtruth}

Has the model correctly answered the query? The response should cover the ground truth answer, though it may contain additional reasoning or thinking, and doesn't need to be word-for-word identical.

Scoring guidelines:
- 1.0 if the answer is fully correct and covers the ground truth
- Between 0.0 and 1.0 if the answer is partially correct
- 0.0 if the answer is completely incorrect or missing

Think step by step and provide your score from 0.0 to 1.0 in the format: \\boxed{{score}}. Replace score with your numerical score."""
    
    else:
        return 0.0
    
    try:
        judge_response = judge_model.query_llm(
            action=f"{benchmark_type}_judgment",
            prompt=judge_prompt,
        )
        # print(f"{benchmark_type} judge response: {judge_response}")
        
        # Extract boxed numerical score
        boxed_match = re.search(r'\\boxed\{(\d+\.?\d*)\}', judge_response.strip())
        if boxed_match:
            score = float(boxed_match.group(1))
            return max(0.0, min(1.0, score))
        
        # Fallback: return 0.0 if no valid score found
        return 0.0
            
    except Exception as e:
        print(f"Error in {benchmark_type} judge evaluation: {e}")
        return 0.0


def judge_preference_alignment_parallel(model_response: str, groundtruth_preference: str, persona: Dict[str, Any], n_trials: int = 1, judge_fn=None) -> float:
    """
    Use LLM as a judge to check preference alignment with multiple parallel calls.
    
    Args:
        model_response (str): The model's response
        groundtruth_preference (str): The ground truth user preference
        persona (Dict[str, Any]): The persona information (or benchmark_type for external benchmarks)
        n_trials (int): Number of parallel trials to run (default: 5)
        judge_fn: Optional judge function to use (default: judge_preference_alignment)
        
    Returns:
        float: Average score between 0.0 and 1.0 indicating preference alignment
    """
    if not model_response or not groundtruth_preference:
        return 0.0
    
    if n_trials <= 0:
        n_trials = 1
    
    # Use provided judge function or default to variants
    if judge_fn is None:
        judge_fn = judge_preference_alignment
    
    # For single trial, skip thread pool overhead
    if n_trials == 1:
        try:
            score = judge_fn(model_response, groundtruth_preference, persona)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Error in judge call: {e}")
            return 0.0
    
    # Use ThreadPoolExecutor for multiple parallel calls
    scores = []
    with ThreadPoolExecutor(max_workers=n_trials) as executor:
        futures = [executor.submit(judge_fn, model_response, groundtruth_preference, persona) for _ in range(n_trials)]
        
        for i, future in enumerate(futures):
            try:
                score = future.result(timeout=60)
                scores.append(score)
            except Exception as e:
                print(f"Error in parallel judge call {i}: {e}")
                scores.append(0.0)
    
    # Calculate average score
    if scores:
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
    # Note: Using raw strings with single backslash to match the literal \boxed in text
    boxed_patterns = [
        r'\\boxed\{([a-dA-D])\}',        # \boxed{a} or \boxed{A}
        r'\\boxed\{\(([a-dA-D])\)\}',    # \boxed{(a)} or \boxed{(A)}
        r'boxed\{([a-dA-D])\}',          # boxed{a} without backslash (fallback)
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
        extra_info (Dict[str, Any]): Additional information containing 'data_source' and 'pref_type'
        
    Returns:
        float: Reward score between 0.0 and 1.0
    """
    if not solution_str:
        return 0.0
    
    # Get data source from extra_info
    data_source = extra_info.get('data_source', 'implicit_persona') if extra_info else 'implicit_persona'
    
    # Check if this is benchmark evaluation mode (validation)
    eval_benchmark = extra_info.get('eval_benchmark', False) if extra_info else False
    
    # MCQ-specific scoring logic - check this first before format enforcement
    is_mcq = ground_truth.get('is_mcq', False) or (extra_info is not None and 'mcq' in extra_info)
    
    # For PrefEval and LongMemEval: different evaluation logic
    # Also apply this logic to our own data when eval_benchmark=True
    if data_source in ['prefeval', 'longmemeval'] or (eval_benchmark and not is_mcq):
        # No format enforcement for external benchmarks or validation mode
        # Extract final response after </think> if it exists, but no penalty if missing
        solution_clean = extract_solution(solution_str) if '</think>' in solution_str else solution_str.strip()
        
        # Extract ground truth - for these datasets, groundtruth_preference contains the answer/preference
        groundtruth = ground_truth.get('groundtruth_preference', '')
        
        if not groundtruth:
            return 0.0
        
        # For our own data in validation mode, use judge_preference_alignment_with_variants
        # For external benchmarks, use judge_external_benchmark
        if eval_benchmark and data_source not in ['prefeval', 'longmemeval']:
            # Our own data in validation mode - use standard judge without filtering
            persona_info = {"preference": groundtruth}
            judge_score = judge_preference_alignment_parallel(
                solution_clean,
                groundtruth,
                persona=persona_info,
                n_trials=1,
                judge_fn=judge_preference_alignment
            )
        else:
            # External benchmarks - use external benchmark judge
            judge_score = judge_preference_alignment_parallel(
                solution_clean, 
                groundtruth, 
                persona=data_source,  # Pass benchmark type as persona
                n_trials=1,
                judge_fn=judge_external_benchmark
            )
        return judge_score
    
    else:
        # For ImplicitPersona: original evaluation logic with format enforcement
        
        if is_mcq:
            # For MCQ, we don't enforce <think> tags - the \boxed{letter} format is sufficient
            return compute_mcq_score(solution_str, ground_truth)
        
        # For non-MCQ ImplicitPersona: enforce format with <think> and </think> tokens
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

