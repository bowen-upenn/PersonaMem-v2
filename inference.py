#!/usr/bin/env python3
"""
Simple evaluation script for the ImplicitPersona benchmark.
Runs evaluation on benchmark.csv using chat history and evaluates responses.
"""

import csv
import json
import os
import argparse
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml
from collections import defaultdict
import time
from tqdm import tqdm
from datetime import datetime
import re
import random
import tiktoken

from query_llm import QueryLLM
from inference_utils import (
    evaluate_narrow_judge,
    evaluate_broad_judge
)


class PersonaBenchmarkEvaluator:
    def __init__(self, config_path: str = "config.yaml", model_name: str = None, result_path: str = "results/", verbose: bool = False):
        """Initialize the evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.verbose = verbose
        
        # Override model name if specified
        if model_name and model_name in self._map_model_name(model_name):
            self.config['models']['llm_model'] = self._map_model_name(model_name)
        
        self.query_llm = QueryLLM(self.config)
        self.results_dir = Path(result_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for conversations: {file_path: conversations}
        self.chat_history_cache = {}
        

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _map_model_name(self, model_name: str) -> str:
        """Map user-friendly model names to deployment names."""
        # Only map models that need aliasing
        model_mapping = {
            'gpt-4o': 'gpt-4o-0806',
            'gemini-pro': 'gemini-2.5-pro',
            'gemini-flash': 'gemini-2.5-flash',
            'claude-sonnet': 'claude-3-5-sonnet-20241022',
            'claude-haiku': 'claude-3-5-haiku-20241022'
        }
        
        return model_mapping.get(model_name, model_name)
    

    def load_chat_history(self, chat_history_path: str, size: str = '32k', use_cache: bool = True) -> List[Dict[str, str]]:
        """Load chat history from JSON file. Cache key is the file path (which contains size info)."""
        # Check cache if enabled - use file path as key (already contains size: .../32k/... or .../128k/...)
        if use_cache and chat_history_path in self.chat_history_cache:
            return self.chat_history_cache[chat_history_path]
        
        try:
            with open(chat_history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract conversation history - handle different formats
                if isinstance(data, list):
                    conversations = data
                elif isinstance(data, dict) and 'conversations' in data:
                    conversations = data['conversations']
                elif isinstance(data, dict):
                    # Try to find conversation data in nested structure
                    conversations = []
                    for key, value in data.items():
                        if isinstance(value, dict) and 'conversations' in value:
                            conversations = value['conversations']
                            break
                        elif isinstance(value, list):
                            conversations = value
                            break
                else:
                    conversations = []
                
                # Cache the loaded conversation with file path as key
                # Only cache if key doesn't exist already
                if chat_history_path not in self.chat_history_cache:
                    # Check if cache is at max capacity (2 entries)
                    if len(self.chat_history_cache) >= 2:
                        # Clear cache before adding new entry
                        self.chat_history_cache.clear()

                    self.chat_history_cache[chat_history_path] = conversations
                
                return conversations
                
        except Exception as e:
            print(f"Error loading chat history from {chat_history_path}: {e}")
            return []
    
    def _reduce_context_length(self, conversations: List[Dict[str, str]], tokens_to_remove: int = 2000) -> List[Dict[str, str]]:
        """Reduce context length by removing code-related conversations."""
        enc = tiktoken.encoding_for_model("gpt-4o")
        
        print(f"Reducing context by removing ~{tokens_to_remove} tokens...")
        
        # Pattern to detect code-related content
        code_pattern = re.compile(r'\b(code|python|buggy)\b', re.IGNORECASE)

        # Sample 1000 random conversations to find code-related content
        sample_size = min(1000, len(conversations))
        sample_indices = random.sample(range(len(conversations)), sample_size)
        
        code_indices = []
        for idx in sample_indices:
            content = str(conversations[idx].get('content', ''))
            if code_pattern.search(content):
                code_indices.append(idx)

        # Remove conversation pairs and count tokens removed
        random.shuffle(code_indices)
        indices_to_remove = set()
        tokens_removed = 0
        
        for idx in code_indices:
            if idx in indices_to_remove or tokens_removed >= tokens_to_remove:
                continue
            
            role = conversations[idx].get('role', '')
            
            # Identify pair indices
            if role == 'user' and idx + 1 < len(conversations):
                pair_indices = [idx, idx + 1]
            elif role == 'assistant' and idx - 1 >= 0:
                pair_indices = [idx - 1, idx]
            else:
                pair_indices = [idx]
            
            # Count tokens in this pair
            pair_tokens = sum(
                len(enc.encode(str(conversations[i].get('content', ''))))
                for i in pair_indices
            )
            
            # Add to removal set
            indices_to_remove.update(pair_indices)
            tokens_removed += pair_tokens
        
        # Create final list without removed indices
        final_conversations = [
            conv for i, conv in enumerate(conversations)
            if i not in indices_to_remove
        ]
        
        print(f"Removed {len(indices_to_remove)} messages (~{tokens_removed} tokens)")
        # Update the cache
        self.chat_history_cache[self._current_chat_history_path] = final_conversations

        return final_conversations
    

    def create_mcq_options(self, correct_answer: str, incorrect_answers: List[str], 
                          seed: int = None) -> Tuple[str, Dict[str, str]]:
        """Create MCQ options string and mapping."""
        # Combine all options and shuffle with consistent seed
        import random
        if seed is not None:
            random.seed(seed)
        
        options = [correct_answer] + incorrect_answers
        random.shuffle(options)
        
        # Create mapping of letters to answers
        option_mapping = {}
        option_parts = []
        
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, etc.
            option_mapping[letter] = option
            option_parts.append(f"{letter}. {option}")
        
        mcq_instruction = (
            "Please choose the best answer from the following options:\n\n" +
            "\n".join(option_parts) +
            "\n\nThink step by step about which answer best fits the user's query and conversation context. "
            "Provide your reasoning first, then give your final answer as 'Final Answer: [Letter]'"
        )
        
        return mcq_instruction, option_mapping
    
    
    def evaluate_row(self, row: Dict[str, Any], eval_mode: str = "mcq", 
                    use_multimodal: bool = False, size: str = '32k') -> Dict[str, Any]:
        """Evaluate a single row from the benchmark."""
        # Parse user query from JSON/Python dict string and append to chat history
        try:
            # First try JSON parsing (in case it's proper JSON)
            user_query_dict = json.loads(row['user_query'])
        except json.JSONDecodeError:
            try:
                # The CSV contains Python dict literals with single quotes, not JSON
                # Use ast.literal_eval to safely parse Python dictionary literals
                user_query_dict = ast.literal_eval(row['user_query'])
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing user_query for persona {row['persona_id']}: {e}")
                print(f"Raw user_query content: {row['user_query'][:100]}...")
                # Create a fallback user query dict
                user_query_dict = {
                    "role": "user", 
                    "content": str(row['user_query']).strip('"').strip("'")
                }
        
        # Load appropriate chat history based on size parameter
        try:
            # Construct column name based on size (e.g., 'chat_history_32k_link' or 'chat_history_128k_link')
            size_column = f'chat_history_{size}_link'
            
            # Try to get the chat history path
            if size_column in row:
                chat_history_path = row[size_column]
            # Fallback to generic 'chat_history_link' if size-specific column not found
            elif 'chat_history_link' in row:
                chat_history_path = row['chat_history_link']
                print(f"  Warning: {size_column} not found, using generic chat_history_link")
            else:
                raise KeyError(f"chat_history_{size}_link or chat_history_link")
        except KeyError as e:
            # Handle missing column error
            available_columns = list(row.keys())
            raise KeyError(f"Missing required column '{e}'. Available columns: {available_columns}")
        
        # Store the current chat history path for cache updates during context reduction
        self._current_chat_history_path = chat_history_path
        chat_history = self.load_chat_history(chat_history_path, size)
        
        # Append user query to chat history
        full_chat_history = chat_history + [user_query_dict]
        
        # Create consistent seed for this row to ensure reproducible shuffling
        row_seed = hash(f"{row['persona_id']}_{user_query_dict['content']}") % 2**32
        
        # Initialize result dictionary with all columns
        result = {
            'model_response_mcq': '',
            'predicted_answer_mcq': '',
            'is_correct_mcq': '',
            'model_response_openended': '',
            'is_correct_openended': ''
        }
        
        # Handle evaluation mode
        if eval_mode in ["mcq", "both"]:
            # Parse incorrect answers
            try:
                incorrect_answers = json.loads(row['incorrect_answers']) if row['incorrect_answers'] else []
            except json.JSONDecodeError:
                incorrect_answers = []
            
            # Create MCQ instruction
            mcq_instruction, option_mapping = self.create_mcq_options(
                row['correct_answer'], 
                incorrect_answers,
                seed=row_seed
            )
            
            # Find the correct MCQ option letter
            correct_mcq_option = "N/A"
            for letter, answer in option_mapping.items():
                if answer == row['correct_answer']:
                    correct_mcq_option = letter
                    break
            
            # Add MCQ instruction as system message and send full conversation
            messages_to_send = full_chat_history + [{"role": "system", "content": mcq_instruction}]
            
            response_mcq = self._query_with_retry(messages_to_send)
            
            # Extract final answer and check correctness
            final_answer = self.extract_final_answer(response_mcq)
            is_correct = self.check_mcq_correctness(final_answer, row['correct_answer'], option_mapping)
            
            result['model_response_mcq'] = response_mcq
            result['predicted_answer_mcq'] = final_answer
            result['is_correct_mcq'] = str(is_correct)
            result['correct_mcq_option'] = correct_mcq_option
        
        if eval_mode in ["generative", "both"]:
            # For generative, just send the chat history as is
            response_openended = self._query_with_retry(full_chat_history)
            
            result['model_response_openended'] = response_openended
            # is_correct_openended left blank as requested
        
        return result
    
    def _query_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """Query LLM with automatic retry on context length error."""
        try:
            return self.query_llm.query_llm(messages, use_history=True)
        except Exception as e:
            error_str = str(e)
            if "'code': 'context_length_exceeded'" not in error_str:
                raise
            
            print(f"  Context length exceeded, reducing and retrying...")
            
            tokens_to_remove = 2000
            match = re.search(r'resulted in (\d+) tokens', error_str)
            if match:
                tokens_to_remove = int(match.group(1)) - 128000 + 1000  # Add buffer
            
            # Reduce the messages directly
            reduced_messages = self._reduce_context_length(messages, tokens_to_remove)
            
            return self.query_llm.query_llm(reduced_messages, use_history=True)
    

    def extract_final_answer(self, response: str) -> str:
        """Extract final answer letter from MCQ response."""
        if not response:
            return ""
        
        # Look for various answer patterns
        import re
        patterns = [
            # Gemini LaTeX format: $\boxed{B}$ or \boxed{B}
            r'\$\\boxed\{([A-Z])\}\$',
            r'\\boxed\{([A-Z])\}',
            # Standard formats
            r'Final Answer:\s*([A-Z])',
            r'final answer:\s*([A-Z])',
            r'Answer:\s*([A-Z])',
            r'answer:\s*([A-Z])',
            # The final answer is [Letter]
            r'final answer is\s*\$?\\boxed\{([A-Z])\}\$?',
            r'final answer is\s*([A-Z])',
            r'the answer is\s*\$?\\boxed\{([A-Z])\}\$?',
            r'the answer is\s*([A-Z])',
            # Single letter at end
            r'\b([A-Z])\.\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        return ""
    

    def check_mcq_correctness(self, predicted_answer: str, correct_answer: str, 
                             option_mapping: Dict[str, str]) -> bool:
        """Check if MCQ answer is correct."""
        if not predicted_answer or not option_mapping:
            return False
        
        # Check if the predicted letter maps to the correct answer
        predicted_text = option_mapping.get(predicted_answer.upper(), "")
        return predicted_text == correct_answer
    

    def _create_summary_file(self, results_csv_path: str, total_processed: int, total_correct: int, overall_accuracy: float, sizes_evaluated: list):
        """Create a summary file with accuracy statistics by category."""
        # Read the results CSV to calculate category-wise accuracies
        results_df_data = []
        with open(results_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results_df_data.append(row)
        
        # Create summary file path
        summary_file = Path(results_csv_path).parent / f"{Path(results_csv_path).stem}_summary.txt"
        
        # Categories to analyze
        categories = [
            'persona_id', 'topic_query', 'topic_preference', 'conversation_scenario', 
            'pref_type', 'who', 'updated', 'sensitive_info', 
            'distance_from_related_snippet_to_query_32k', 'distance_from_related_snippet_to_query_128k',
            'num_persona_relevant_tokens_128k'
        ]
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            # Write overall statistics
            f.write("="*60 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Total processed: {total_processed}\n")
            f.write(f"Sizes evaluated: {', '.join(sizes_evaluated)}\n")
            f.write(f"Overall MCQ Accuracy (using {sizes_evaluated[0]}): {overall_accuracy:.3f} ({total_correct}/{total_processed})\n\n")
            
            # For each size, calculate overall accuracy
            for size in sizes_evaluated:
                size_correct = 0
                size_total = 0
                
                for row in results_df_data:
                    if row.get(f'is_correct_mcq_{size}') in ['True', 'False']:
                        size_total += 1
                        if row[f'is_correct_mcq_{size}'] == 'True':
                            size_correct += 1
                
                if size_total > 0:
                    size_accuracy = size_correct / size_total
                    f.write(f"Overall MCQ Accuracy ({size}): {size_accuracy:.3f} ({size_correct}/{size_total})\n")
            
            f.write("\n")
            
            # Calculate and write category-wise accuracies for each size
            for size in sizes_evaluated:
                f.write(f"\n{'='*60}\n")
                f.write(f"CATEGORY BREAKDOWN FOR {size.upper()}\n")
                f.write(f"{'='*60}\n")
                
                for category in categories:
                    f.write(f"\nACCURACY BY {category.upper()} ({size}):\n")
                    f.write("-" * 40 + "\n")
                    
                    # Check if this is a distance category that needs binning
                    is_distance_category = 'distance_from_related_snippet_to_query' in category
                    
                    if is_distance_category:
                        # Bin distance values into 1024-token intervals
                        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                        
                        for row in results_df_data:
                            if category in row and row.get(f'is_correct_mcq_{size}') in ['True', 'False']:
                                try:
                                    distance_value = int(row[category])
                                    # Create bin: e.g., "0-1023", "1024-2047", etc.
                                    bin_start = (distance_value // 1024) * 1024
                                    bin_end = bin_start + 1023
                                    bin_label = f"{bin_start}-{bin_end}"
                                    
                                    category_stats[bin_label]['total'] += 1
                                    if row[f'is_correct_mcq_{size}'] == 'True':
                                        category_stats[bin_label]['correct'] += 1
                                except (ValueError, TypeError):
                                    # Handle non-numeric values
                                    category_value = row[category]
                                    category_stats[category_value]['total'] += 1
                                    if row[f'is_correct_mcq_{size}'] == 'True':
                                        category_stats[category_value]['correct'] += 1
                        
                        # Sort bins by their starting value
                        def sort_key(item):
                            try:
                                # Extract starting value from bin label like "0-1023"
                                return int(item[0].split('-')[0])
                            except:
                                return float('inf')  # Put non-numeric values at the end
                        
                        sorted_categories = sorted(category_stats.items(), key=sort_key)
                    else:
                        # Regular category handling
                        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                        
                        for row in results_df_data:
                            if category in row and row.get(f'is_correct_mcq_{size}') in ['True', 'False']:
                                category_value = row[category]
                                category_stats[category_value]['total'] += 1
                                if row[f'is_correct_mcq_{size}'] == 'True':
                                    category_stats[category_value]['correct'] += 1
                        
                        # Sort by category value for consistent output
                        sorted_categories = sorted(category_stats.items())
                    
                    for cat_value, stats in sorted_categories:
                        if stats['total'] > 0:
                            cat_accuracy = stats['correct'] / stats['total']
                            f.write(f"{cat_value}: {cat_accuracy:.3f} ({stats['correct']}/{stats['total']})\n")
                        else:
                            f.write(f"{cat_value}: N/A (0/0)\n")
                    
                    # Write category summary
                    total_in_category = sum(stats['total'] for stats in category_stats.values())
                    total_correct_in_category = sum(stats['correct'] for stats in category_stats.values())
                    if total_in_category > 0:
                        category_overall_accuracy = total_correct_in_category / total_in_category
                        f.write(f"\nCategory Overall ({size}): {category_overall_accuracy:.3f} ({total_correct_in_category}/{total_in_category})\n")
        
        print(f"Summary file created: {summary_file}")

    def run_evaluation(self, benchmark_file: str = None, eval_mode: str = "mcq", 
                      use_multimodal: bool = False, max_items: int = None, size: str = '32k') -> str:
        """Run evaluation on the benchmark dataset."""
        # Auto-select benchmark file if not specified
        if benchmark_file is None:
            if use_multimodal:
                benchmark_file = "benchmark/multimodal/benchmark.csv"
            else:
                benchmark_file = "benchmark/text/benchmark.csv"
        
        print(f"Starting evaluation...")
        print(f"Benchmark file: {benchmark_file}")
        print(f"Evaluation mode: {eval_mode}")
        print(f"Use multimodal: {use_multimodal}")
        print(f"Size: {size}")
        
        # Load benchmark data
        rows = []
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
                if max_items and len(rows) >= max_items:
                    break
        
        print(f"Loaded {len(rows)} rows from benchmark")
        
        # Determine which sizes to evaluate
        if size == 'both':
            sizes_to_evaluate = ['32k', '128k']
            # Create output CSV file without size in filename
            run_timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
            output_file = self.results_dir / f"evaluation_results_{eval_mode}{'_multimodal' if use_multimodal else ''}_{run_timestamp}.csv"
        else:
            sizes_to_evaluate = [size]
            # Create output CSV file with size in filename
            run_timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
            output_file = self.results_dir / f"evaluation_results_{eval_mode}{'_multimodal' if use_multimodal else ''}_{size}_{run_timestamp}.csv"
        
        # Add new columns to fieldnames based on sizes to evaluate
        output_fieldnames = list(fieldnames)
        for eval_size in sizes_to_evaluate:
            output_fieldnames.extend([
                f'model_response_mcq_{eval_size}', 
                f'predicted_answer_mcq_{eval_size}', 
                f'is_correct_mcq_{eval_size}',
                f'model_response_openended_{eval_size}', 
                f'is_correct_openended_{eval_size}'
            ])
        
        # Process each row and write to CSV incrementally
        processed_count = 0
        correct_count = 0
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(tqdm(rows, desc="Processing rows")):

                # if i > 5:
                #     continue  # For testing, limit to first 5 rows
                
                try:
                    # Create output row with all original columns
                    output_row = row.copy()
                    
                    # Initialize all possible output columns
                    for eval_size in sizes_to_evaluate:
                        output_row[f'model_response_mcq_{eval_size}'] = ''
                        output_row[f'predicted_answer_mcq_{eval_size}'] = ''
                        output_row[f'is_correct_mcq_{eval_size}'] = ''
                        output_row[f'model_response_openended_{eval_size}'] = ''
                        output_row[f'is_correct_openended_{eval_size}'] = ''
                    
                    # Evaluate for each size
                    all_results = {}
                    for eval_size in sizes_to_evaluate:
                        result = self.evaluate_row(row, eval_mode, use_multimodal, eval_size)
                        all_results[eval_size] = result
                        
                        # Store results with size suffix
                        output_row[f'model_response_mcq_{eval_size}'] = result.get('model_response_mcq', '')
                        output_row[f'predicted_answer_mcq_{eval_size}'] = result.get('predicted_answer_mcq', '')
                        output_row[f'is_correct_mcq_{eval_size}'] = result.get('is_correct_mcq', '')
                        output_row[f'model_response_openended_{eval_size}'] = result.get('model_response_openended', '')
                        output_row[f'is_correct_openended_{eval_size}'] = result.get('is_correct_openended', '')
                        
                        # Count correct answers for MCQ (use first size for overall count)
                        if eval_size == sizes_to_evaluate[0] and result.get('is_correct_mcq', '') == 'True':
                            correct_count += 1
                    
                    processed_count += 1
                    
                    # Print verbose output if enabled
                    if self.verbose:
                        # ANSI color codes
                        BLUE = '\033[94m'
                        RESET = '\033[0m'
                        
                        print(f"  Verbose output for row {i+1}:")
                        print(f"    {BLUE}user_query{RESET}: {row.get('user_query', 'N/A')}")
                        print(f"    {BLUE}correct_answer{RESET}: {row.get('correct_answer', 'N/A')}")
                        print(f"    {BLUE}preference{RESET}: {row.get('preference', 'N/A')}")
                        
                        for eval_size in sizes_to_evaluate:
                            result = all_results[eval_size]
                            correct_mcq_option = result.get('correct_mcq_option', 'N/A')
                            print(f"    --- {eval_size.upper()} Results ---")
                            print(f"    {BLUE}correct_mcq_option_{eval_size}{RESET}: {correct_mcq_option}")
                            print(f"    {BLUE}model_response_mcq_{eval_size}{RESET}: {output_row[f'model_response_mcq_{eval_size}']}")
                            print(f"    {BLUE}predicted_answer_mcq_{eval_size}{RESET}: {output_row[f'predicted_answer_mcq_{eval_size}']}")
                            print(f"    {BLUE}is_correct_mcq_{eval_size}{RESET}: {output_row[f'is_correct_mcq_{eval_size}']}")
                            print(f"    {BLUE}model_response_openended_{eval_size}{RESET}: {output_row[f'model_response_openended_{eval_size}']}")
                            print(f"    {BLUE}is_correct_openended_{eval_size}{RESET}: {output_row[f'is_correct_openended_{eval_size}']}")
                        print('-' * 50)
                    
                    writer.writerow(output_row)
                    f.flush()  # Ensure data is written immediately
                    
                except Exception as e:
                    print(f"Error processing row {i+1}: {e}")
                    # Write error to CSV
                    output_row = row.copy()
                    for eval_size in sizes_to_evaluate:
                        output_row[f'model_response_mcq_{eval_size}'] = f"ERROR: {str(e)}"
                        output_row[f'predicted_answer_mcq_{eval_size}'] = ''
                        output_row[f'is_correct_mcq_{eval_size}'] = ''
                        output_row[f'model_response_openended_{eval_size}'] = ''
                        output_row[f'is_correct_openended_{eval_size}'] = ''
                    writer.writerow(output_row)
                    f.flush()
        
        print(f"\nResults saved to {output_file}")
        
        # Print evaluation statistics and create summary file
        if eval_mode in ["mcq", "both"] and processed_count > 0:
            accuracy = correct_count / processed_count
            print(f"\n{'='*50}")
            print(f"EVALUATION STATISTICS")
            print(f"{'='*50}")
            print(f"Total processed: {processed_count}")
            print(f"Overall MCQ Accuracy: {accuracy:.3f} ({correct_count}/{processed_count})")
            
            # Create summary file
            self._create_summary_file(output_file, processed_count, correct_count, accuracy, sizes_to_evaluate)
        
        return str(output_file)
    

    def run_judge_evaluation(self, results_csv_path: str) -> str:
        """Run judge evaluation on existing results CSV and update it in place."""
        print(f"Starting judge evaluation on {results_csv_path}...")
        
        # Read existing results
        rows = []
        with open(results_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames)
            rows = list(reader)
        
        # Add new judge columns if not already present
        new_columns = [
            'is_correct_openended_narrow',
            'judge_responses_narrow',
            'is_correct_openended_broad',
            'judge_responses_broad'
        ]
        
        for col in new_columns:
            if col not in fieldnames:
                fieldnames.append(col)
        
        # Create temporary output file
        temp_output = results_csv_path + '.tmp'
        
        # Process each row with judges
        with open(temp_output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(tqdm(rows, desc="Processing judge evaluation")):
                # if i > 5:
                #     break  # For testing, limit to first 5 rows
                
                # Check if we have an openended response to evaluate
                model_response_openended = row.get('model_response_openended', '').strip()
                
                if model_response_openended and not model_response_openended.startswith('ERROR:'):
                    try:
                        # Evaluate with narrow judge
                        narrow_decision, narrow_responses = evaluate_narrow_judge(
                            row, model_response_openended, 
                            self.query_llm.query_llm, self.load_chat_history
                        )
                        row['is_correct_openended_narrow'] = str(narrow_decision)
                        row['judge_responses_narrow'] = narrow_responses
                        
                        # Evaluate with broad judge
                        broad_decision, broad_responses = evaluate_broad_judge(
                            row, model_response_openended,
                            self.query_llm.query_llm
                        )
                        row['is_correct_openended_broad'] = str(broad_decision)
                        row['judge_responses_broad'] = broad_responses
                        
                        print(f"  Narrow: {narrow_decision}, Broad: {broad_decision}")
                        
                    except Exception as e:
                        print(f"  Error evaluating row {i+1}: {e}")
                        row['is_correct_openended_narrow'] = ''
                        row['judge_responses_narrow'] = f"ERROR: {str(e)}"
                        row['is_correct_openended_broad'] = ''
                        row['judge_responses_broad'] = f"ERROR: {str(e)}"
                else:
                    # No openended response to evaluate
                    row['is_correct_openended_narrow'] = ''
                    row['judge_responses_narrow'] = ''
                    row['is_correct_openended_broad'] = ''
                    row['judge_responses_broad'] = ''
                
                writer.writerow(row)
                f.flush()
        
        # Replace original file with updated one
        os.replace(temp_output, results_csv_path)
        
        print(f"\nJudge evaluation completed. Updated results saved to {results_csv_path}")
        return results_csv_path
    



if __name__ == "__main__":
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Run evaluation on ImplicitPersona benchmark')
    parser.add_argument('--benchmark_file', type=str, default=None,
                       help='Path to benchmark CSV file (auto-selects based on --use_multimodal if not specified)')
    parser.add_argument('--eval_mode', type=str, choices=['mcq', 'generative', 'both'], default='mcq',
                       help='Evaluation mode: mcq, generative, or both')
    parser.add_argument('--use_multimodal', action='store_true',
                       help='Use multimodal chat history instead of regular chat history')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to process (for testing)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    # Supported models: gpt-4.1, gpt-4.1-mini, gpt-4o,  gpt-4o-mini, 
    # gpt-5-chat, gpt-5-mini, gpt-5-nano, o1, o1-mini, o3-mini, o4-mini
    # gemini-2.5-pro, gemini-2.5-flash, gemini-pro, gemini-flash
    # claude-3-5-sonnet, claude-3-5-haiku, claude-sonnet, claude-haiku
    parser.add_argument('--model_name', type=str, default='gpt-5-chat',
                       help='Model name to use for evaluation (overrides config file)')
    parser.add_argument('--result_path', type=str, default='results/',
                       help='Directory to save evaluation results (default: results/)')
    parser.add_argument('--size', type=str, default='32k',
                       help='Chat history size to use (one of 32k, 128k, both). Uses chat_history_{size}_link column from benchmark CSV')
    parser.add_argument('--run_judges', action='store_true',
                       help='Run judge evaluation on existing results CSV. Requires --results_csv_path')
    parser.add_argument('--results_csv_path', type=str, default=None,
                       help='Path to existing results CSV file for judge evaluation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output to print detailed response information')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PersonaBenchmarkEvaluator(args.config, args.model_name, args.result_path, args.verbose)
    
    # Run judge evaluation if requested
    if args.run_judges:
        if not args.results_csv_path:
            print("Error: --results_csv_path is required when using --run_judges")
            exit(1)
        
        output_file = evaluator.run_judge_evaluation(args.results_csv_path)
        print(f"\nJudge evaluation completed. Results updated in: {output_file}")
    else:
        # Run normal inference evaluation
        output_file = evaluator.run_evaluation(
            args.benchmark_file,
            args.eval_mode,
            args.use_multimodal,
            args.max_items,
            args.size
        )
        
        print(f"\nEvaluation completed. Results saved to: {output_file}")
