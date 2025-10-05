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
        model_mapping = {
            'gpt-4o': 'gpt-4o-0806',  # Route gpt-4o to specific version
            'gpt-4.1': 'gpt-4.1',
            'gpt-4.1-mini': 'gpt-4.1-mini',
            'gpt-4o-0806': 'gpt-4o-0806',
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-5-chat': 'gpt-5-chat',
            'gpt-5-mini': 'gpt-5-mini',
            'gpt-5-nano': 'gpt-5-nano',
            'o1': 'o1',
            'o1-mini': 'o1-mini',
            'o3-mini': 'o3-mini',
            'o4-mini': 'o4-mini',
            'gemini-2.5-pro': 'gemini-2.5-pro',
            'gemini-2.5-flash': 'gemini-2.5-flash',
            'gemini-pro': 'gemini-2.5-pro',
            'gemini-flash': 'gemini-2.5-flash',
            'claude-3-5-sonnet': 'claude-3-5-sonnet-20241022',
            'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022',
            'claude-3-5-haiku': 'claude-3-5-haiku-20241022',
            'claude-3-5-haiku-20241022': 'claude-3-5-haiku-20241022',
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
                    use_multimodal: bool = False) -> Dict[str, Any]:
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
        
        # Load appropriate chat history
        try:
            if use_multimodal:
                # Try the correct column name first, then fallback to old format
                if 'chat_history_32k_link' in row:
                    chat_history_path = row['chat_history_32k_link']
                elif 'chat_history_link' in row:
                    chat_history_path = row['chat_history_link']
                else:
                    raise KeyError("chat_history_32k_link")
            else:
                # Use 32k chat history by default (could also use 128k)
                if 'chat_history_32k_link' in row:
                    chat_history_path = row['chat_history_32k_link']
                elif 'chat_history_link' in row:
                    chat_history_path = row['chat_history_link']
                else:
                    raise KeyError("chat_history_32k_link")
        except KeyError as e:
            # Handle missing column error
            available_columns = list(row.keys())
            raise KeyError(f"Missing required column '{e}'. Available columns: {available_columns}")
        
        chat_history = self.load_chat_history(chat_history_path)
        
        # Append user query to chat history
        full_chat_history = chat_history + [user_query_dict]
        
        # Create consistent seed for this row to ensure reproducible shuffling
        row_seed = hash(f"{row['persona_id']}_{user_query_dict['content']}") % 2**32
        
        # Handle evaluation mode
        option_mapping = None
        start_time = time.time()
        
        if eval_mode == "mcq":
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
            response = self.query_llm.query_llm(messages_to_send, use_history=True)
            
        else:  # generative
            # For generative, just send the chat history as is
            response = self.query_llm.query_llm(full_chat_history, use_history=True)
        
        end_time = time.time()
        
        # Create result record
        result = {
            'persona_id': row['persona_id'],
            'conversation_scenario': row['conversation_scenario'],
            'pref_type': row['pref_type'],
            'updated': row['updated'],
            'who': row['who'],
            'user_query': row['user_query'],
            'correct_answer': row['correct_answer'],
            'incorrect_answers': row['incorrect_answers'],
            'topic_preference': row['topic_preference'],
            'preference': row['preference'],
            'chat_history_used': chat_history_path,
            'eval_mode': eval_mode,
            'use_multimodal': use_multimodal,
            'messages_sent': messages_to_send if eval_mode == "mcq" else full_chat_history,
            'model_response': response,
            'response_time': end_time - start_time,
            'timestamp': time.time()
        }
        
        # Add evaluation-specific fields
        if eval_mode == "mcq":
            # Extract final answer from response
            final_answer = self.extract_final_answer(response)
            result['predicted_answer'] = final_answer
            result['option_mapping'] = option_mapping
            result['is_correct'] = self.check_mcq_correctness(final_answer, row['correct_answer'], option_mapping)
        
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
            for row in reader:
                rows.append(row)
                if max_items and len(rows) >= max_items:
                    break
        
        print(f"Loaded {len(rows)} rows from benchmark")
        
        # Create individual results directory for this run
        run_timestamp = int(time.time())
        individual_results_dir = self.results_dir / f"individual_results_{eval_mode}{'_multimodal' if use_multimodal else ''}_{run_timestamp}"
        individual_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each row and save individually
        results = []
        processed_count = 0
        
        for i, row in enumerate(rows):
            print(f"Processing row {i+1}/{len(rows)} (Persona {row['persona_id']})")
            
            # Create individual result file path
            individual_file = individual_results_dir / f"result_persona_{row['persona_id']}_row_{i+1}.json"
            
            # Skip if already processed (for resuming interrupted runs)
            if individual_file.exists():
                print(f"  Skipping row {i+1} - already processed")
                try:
                    with open(individual_file, 'r', encoding='utf-8') as f:
                        existing_result = json.load(f)
                        results.append(existing_result)
                        processed_count += 1
                except Exception as e:
                    print(f"  Error loading existing result: {e}")
                continue
            
            try:
                result = self.evaluate_row(row, eval_mode, use_multimodal)
                results.append(result)
                processed_count += 1
                
                # Save individual result immediately
                with open(individual_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"  Result saved to {individual_file}")
                
            except Exception as e:
                print(f"Error processing row {i+1}: {e}")
                # Add error result
                error_result = {
                    'persona_id': row['persona_id'],
                    'row_number': i+1,
                    'error': str(e),
                    'timestamp': time.time()
                }
                results.append(error_result)
                
                # Save error result individually
                with open(individual_file, 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)
                
                print(f"  Error result saved to {individual_file}")
        
        # Save aggregated results
        output_file = self.results_dir / f"evaluation_results_{eval_mode}{'_multimodal' if use_multimodal else ''}_{run_timestamp}.json"
        
        evaluation_data = {
            'metadata': {
                'benchmark_file': benchmark_file,
                'model_name': self.config['models']['llm_model'],
                'eval_mode': eval_mode,
                'use_multimodal': use_multimodal,
                'total_rows': len(rows),
                'processed_rows': processed_count,
                'max_items': max_items,
                'timestamp': run_timestamp,
                'individual_results_dir': str(individual_results_dir)
            },
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        print(f"Aggregated results saved to {output_file}")
        print(f"Individual results saved in {individual_results_dir}")
        
        # Print evaluation statistics
        self.print_evaluation_stats(results, eval_mode)
        
        return str(output_file)
    

    def print_evaluation_stats(self, results: List[Dict[str, Any]], eval_mode: str):
        """Print evaluation statistics."""
        print(f"\n{'='*50}")
        print(f"EVALUATION STATISTICS")
        print(f"{'='*50}")
        
        total_results = len([r for r in results if 'error' not in r])
        error_count = len([r for r in results if 'error' in r])
        
        print(f"Total processed: {total_results}")
        print(f"Errors: {error_count}")
        
        if eval_mode == "mcq" and total_results > 0:
            # MCQ-specific stats
            correct_count = len([r for r in results if r.get('is_correct', False)])
            accuracy = correct_count / total_results if total_results > 0 else 0
            
            print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/{total_results})")
            
            # Stats by metadata
            stats_by_field = {}
            fields = ['conversation_scenario', 'pref_type', 'updated', 'who']
            
            for field in fields:
                field_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
                for result in results:
                    if 'error' in result:
                        continue
                    field_value = result.get(field, 'unknown')
                    field_stats[field_value]['total'] += 1
                    if result.get('is_correct', False):
                        field_stats[field_value]['correct'] += 1
                
                print(f"\nAccuracy by {field}:")
                for value, stats in sorted(field_stats.items()):
                    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                    print(f"  {value}: {acc:.3f} ({stats['correct']}/{stats['total']})")
        
        # Response time stats
        response_times = [r.get('response_time', 0) for r in results if 'error' not in r and 'response_time' in r]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print(f"\nAverage response time: {avg_time:.2f} seconds")


if __name__ == "__main__":
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Run evaluation on ImplicitPersona benchmark')
    parser.add_argument('--benchmark_file', type=str, default=None,
                       help='Path to benchmark CSV file (auto-selects based on --use_multimodal if not specified)')
    parser.add_argument('--eval_mode', type=str, choices=['mcq', 'generative'], default='mcq',
                       help='Evaluation mode: mcq or generative')
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
                       help='Size of evaluation benchmark to be used')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PersonaBenchmarkEvaluator(args.config, args.model_name, args.result_path)
    
    # Run evaluation
    output_file = evaluator.run_evaluation(
        args.benchmark_file,
        args.eval_mode,
        args.use_multimodal,
        args.max_items,
        args.size
    )
    
    print(f"\nEvaluation completed. Results saved to: {output_file}")
