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

from query_llm import QueryLLM
from inference_utils import (
    evaluate_narrow_judge,
    evaluate_broad_judge
)


class PersonaBenchmarkEvaluator:
    def __init__(self, config_path: str = "config.yaml", model_name: str = None, result_path: str = "results/"):
        """Initialize the evaluator with configuration."""
        self.config = self._load_config(config_path)
        
        # Override model name if specified
        if model_name and model_name in self._map_model_name(model_name):
            self.config['models']['llm_model'] = self._map_model_name(model_name)
        
        self.query_llm = QueryLLM(self.config)
        self.results_dir = Path(result_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        

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
    

    def load_chat_history(self, chat_history_path: str) -> List[Dict[str, str]]:
        """Load chat history from JSON file."""
        if not chat_history_path or not os.path.exists(chat_history_path):
            return []
        
        try:
            with open(chat_history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract conversation history - handle different formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'conversations' in data:
                    return data['conversations']
                elif isinstance(data, dict):
                    # Try to find conversation data in nested structure
                    for key, value in data.items():
                        if isinstance(value, dict) and 'conversations' in value:
                            return value['conversations']
                        elif isinstance(value, list):
                            return value
                return []
        except Exception as e:
            print(f"Error loading chat history from {chat_history_path}: {e}")
            return []
    

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
        
        chat_history = self.load_chat_history(chat_history_path)
        
        # Append user query to chat history
        full_chat_history = chat_history + [user_query_dict]
        
        # Create consistent seed for this row to ensure reproducible shuffling
        row_seed = hash(f"{row['persona_id']}_{user_query_dict['content']}") % 2**32
        
        # Initialize result dictionary with all columns
        result = {
            'model_response_mcq': '',
            'predicted_answer_mcq': '',
            'is_correct_mcq': '',
            'response_time_mcq': '',
            'model_response_openended': '',
            'is_correct_openended': '',
            'response_time_openended': ''
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
            
            # Add MCQ instruction as system message and send full conversation
            messages_to_send = full_chat_history + [{"role": "system", "content": mcq_instruction}]
            
            start_time_mcq = time.time()
            response_mcq = self.query_llm.query_llm(messages_to_send, use_history=True)
            end_time_mcq = time.time()
            
            # Extract final answer and check correctness
            final_answer = self.extract_final_answer(response_mcq)
            is_correct = self.check_mcq_correctness(final_answer, row['correct_answer'], option_mapping)
            
            result['model_response_mcq'] = response_mcq
            result['predicted_answer_mcq'] = final_answer
            result['is_correct_mcq'] = str(is_correct)
            result['response_time_mcq'] = str(end_time_mcq - start_time_mcq)
        
        if eval_mode in ["generative", "both"]:
            # For generative, just send the chat history as is
            start_time_openended = time.time()
            response_openended = self.query_llm.query_llm(full_chat_history, use_history=True)
            end_time_openended = time.time()
            
            result['model_response_openended'] = response_openended
            result['response_time_openended'] = str(end_time_openended - start_time_openended)
            # is_correct_openended left blank as requested
        
        return result
    

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
        
        # Create output CSV file
        run_timestamp = int(time.time())
        output_file = self.results_dir / f"evaluation_results_{eval_mode}{'_multimodal' if use_multimodal else ''}_{size}_{run_timestamp}.csv"
        
        # Add new columns to fieldnames
        output_fieldnames = list(fieldnames) + [
            'model_response_mcq', 
            'predicted_answer_mcq', 
            'is_correct_mcq', 
            'response_time_mcq',
            'model_response_openended', 
            'is_correct_openended', 
            'response_time_openended'
        ]
        
        # Process each row and write to CSV incrementally
        processed_count = 0
        correct_count = 0
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(rows):
                print(f"Processing row {i+1}/{len(rows)} (Persona {row['persona_id']})")
                
                try:
                    result = self.evaluate_row(row, eval_mode, use_multimodal, size)
                    processed_count += 1
                    
                    # Create output row with all original columns plus new ones
                    output_row = row.copy()
                    output_row['model_response_mcq'] = result.get('model_response_mcq', '')
                    output_row['predicted_answer_mcq'] = result.get('predicted_answer_mcq', '')
                    output_row['is_correct_mcq'] = result.get('is_correct_mcq', '')
                    output_row['response_time_mcq'] = result.get('response_time_mcq', '')
                    output_row['model_response_openended'] = result.get('model_response_openended', '')
                    output_row['is_correct_openended'] = result.get('is_correct_openended', '')
                    output_row['response_time_openended'] = result.get('response_time_openended', '')
                    
                    # Count correct answers for MCQ
                    if result.get('is_correct_mcq', '') == 'True':
                        correct_count += 1
                    
                    writer.writerow(output_row)
                    f.flush()  # Ensure data is written immediately
                    
                    print(f"  Row {i+1} completed and saved")
                    
                except Exception as e:
                    print(f"Error processing row {i+1}: {e}")
                    # Write error to CSV
                    output_row = row.copy()
                    output_row['model_response_mcq'] = f"ERROR: {str(e)}"
                    output_row['predicted_answer_mcq'] = ''
                    output_row['is_correct_mcq'] = ''
                    output_row['response_time_mcq'] = ''
                    output_row['model_response_openended'] = ''
                    output_row['is_correct_openended'] = ''
                    output_row['response_time_openended'] = ''
                    writer.writerow(output_row)
                    f.flush()
        
        print(f"\nResults saved to {output_file}")
        
        # Print evaluation statistics
        if eval_mode in ["mcq", "both"] and processed_count > 0:
            accuracy = correct_count / processed_count
            print(f"\n{'='*50}")
            print(f"EVALUATION STATISTICS")
            print(f"{'='*50}")
            print(f"Total processed: {processed_count}")
            print(f"Overall MCQ Accuracy: {accuracy:.3f} ({correct_count}/{processed_count})")
        
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
            
            for i, row in enumerate(rows):
                print(f"Processing row {i+1}/{len(rows)} (Persona {row.get('persona_id', 'unknown')})")
                
                # Check if we have an openended response to evaluate
                model_response_openended = row.get('model_response_openended', '').strip()
                
                if model_response_openended and not model_response_openended.startswith('ERROR:'):
                    try:
                        # Evaluate with narrow judge
                        print(f"  Evaluating with narrow judge...")
                        narrow_decision, narrow_responses = evaluate_narrow_judge(
                            row, model_response_openended, 
                            self.query_llm.query_llm, self.load_chat_history
                        )
                        row['is_correct_openended_narrow'] = str(narrow_decision)
                        row['judge_responses_narrow'] = narrow_responses
                        
                        # Evaluate with broad judge
                        print(f"  Evaluating with broad judge...")
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
                       help='Chat history size to use (one of 32k, 128k). Uses chat_history_{size}_link column from benchmark CSV')
    parser.add_argument('--run_judges', action='store_true',
                       help='Run judge evaluation on existing results CSV. Requires --results_csv_path')
    parser.add_argument('--results_csv_path', type=str, default=None,
                       help='Path to existing results CSV file for judge evaluation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PersonaBenchmarkEvaluator(args.config, args.model_name, args.result_path)
    
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
