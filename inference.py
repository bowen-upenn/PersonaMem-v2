#!/usr/bin/env python3
"""
Simple evaluation script for the ImplicitPersona benchmark.
Runs evaluation on benchmark.csv using chat history and evaluates responses.
"""

import csv
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml
from collections import defaultdict
import time

from query_llm import QueryLLM


class PersonaBenchmarkEvaluator:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.query_llm = QueryLLM(self.config)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
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
    
    def create_mcq_prompt(self, user_query: str, correct_answer: str, 
                         incorrect_answers: List[str], chat_history: List[Dict[str, str]] = None, 
                         seed: int = None) -> Tuple[str, Dict[str, str]]:
        """Create multiple choice question prompt."""
        # Combine all options and shuffle with consistent seed
        import random
        if seed is not None:
            random.seed(seed)
        
        options = [correct_answer] + incorrect_answers
        random.shuffle(options)
        
        # Create mapping of letters to answers
        option_mapping = {}
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, etc.
            option_mapping[letter] = option
        
        # Create prompt
        prompt_parts = []
        
        if chat_history:
            prompt_parts.append("Previous conversation history:")
            for msg in chat_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                prompt_parts.append(f"{role.title()}: {content}")
            prompt_parts.append("")
        
        prompt_parts.append(f"User Query: {user_query}")
        prompt_parts.append("\nPlease choose the best answer from the following options:")
        
        for i, option in enumerate(options):
            prompt_parts.append(f"{chr(65+i)}. {option}")
        
        prompt_parts.append("\nThink step by step about which answer best fits the user's query and conversation context.")
        prompt_parts.append("Provide your reasoning first, then give your final answer as 'Final Answer: [Letter]'")
        
        return "\n".join(prompt_parts), option_mapping
    
    def create_generative_prompt(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Create generative evaluation prompt."""
        prompt_parts = []
        
        if chat_history:
            prompt_parts.append("Previous conversation history:")
            for msg in chat_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                prompt_parts.append(f"{role.title()}: {content}")
            prompt_parts.append("")
        
        prompt_parts.append(f"User Query: {user_query}")
        prompt_parts.append("\nPlease provide a helpful and detailed response based on the conversation context.")
        
        return "\n".join(prompt_parts)
    
    def parse_user_query(self, user_query_str: str) -> str:
        """Parse user query from JSON format if needed."""
        try:
            # Try to parse as JSON first
            user_query_data = json.loads(user_query_str)
            if isinstance(user_query_data, dict) and 'content' in user_query_data:
                return user_query_data['content']
            elif isinstance(user_query_data, str):
                return user_query_data
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Return as string if not valid JSON
        return str(user_query_str)
    
    def evaluate_row(self, row: Dict[str, Any], eval_mode: str = "mcq", 
                    use_multimodal: bool = False) -> Dict[str, Any]:
        """Evaluate a single row from the benchmark."""
        # Parse user query
        user_query = self.parse_user_query(row['user_query'])
        
        # Load appropriate chat history
        chat_history_path = row['chat_history_multimodal_link'] if use_multimodal else row['chat_history_link']
        chat_history = self.load_chat_history(chat_history_path)
        
        # Create consistent seed for this row to ensure reproducible shuffling
        row_seed = hash(f"{row['persona_id']}_{user_query}") % 2**32
        
        # Create prompt based on evaluation mode
        option_mapping = None
        if eval_mode == "mcq":
            # Parse incorrect answers
            try:
                incorrect_answers = json.loads(row['incorrect_answers']) if row['incorrect_answers'] else []
            except json.JSONDecodeError:
                incorrect_answers = []
            
            prompt, option_mapping = self.create_mcq_prompt(
                user_query, 
                row['correct_answer'], 
                incorrect_answers,
                chat_history,
                seed=row_seed
            )
        else:  # generative
            prompt = self.create_generative_prompt(user_query, chat_history)
        
        # Query the LLM
        start_time = time.time()
        response = self.query_llm.query_llm(prompt, use_history=False)
        end_time = time.time()
        
        # Create result record
        result = {
            'persona_id': row['persona_id'],
            'conversation_scenario': row['conversation_scenario'],
            'pref_type': row['pref_type'],
            'updated': row['updated'],
            'who': row['who'],
            'user_query': user_query,
            'correct_answer': row['correct_answer'],
            'incorrect_answers': row['incorrect_answers'],
            'topic_preference': row['topic_preference'],
            'preference': row['preference'],
            'chat_history_used': chat_history_path,
            'eval_mode': eval_mode,
            'use_multimodal': use_multimodal,
            'prompt': prompt,
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
        
        # Look for "Final Answer: [Letter]" pattern
        import re
        patterns = [
            r'Final Answer:\s*([A-Z])',
            r'final answer:\s*([A-Z])',
            r'Answer:\s*([A-Z])',
            r'answer:\s*([A-Z])',
            r'\b([A-Z])\.\s*$'  # Single letter at end
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
    
    def run_evaluation(self, benchmark_file: str, eval_mode: str = "mcq", 
                      use_multimodal: bool = False, max_items: int = None) -> str:
        """Run evaluation on the benchmark dataset."""
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
        
        # Process each row
        results = []
        for i, row in enumerate(rows):
            print(f"Processing row {i+1}/{len(rows)} (Persona {row['persona_id']})")
            
            try:
                result = self.evaluate_row(row, eval_mode, use_multimodal)
                results.append(result)
            except Exception as e:
                print(f"Error processing row {i+1}: {e}")
                # Add error result
                results.append({
                    'persona_id': row['persona_id'],
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        # Save results
        output_file = self.results_dir / f"evaluation_results_{eval_mode}{'_multimodal' if use_multimodal else ''}_{int(time.time())}.json"
        
        evaluation_data = {
            'metadata': {
                'benchmark_file': benchmark_file,
                'eval_mode': eval_mode,
                'use_multimodal': use_multimodal,
                'total_rows': len(rows),
                'processed_rows': len(results),
                'max_items': max_items,
                'timestamp': time.time()
            },
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
        
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


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Run evaluation on ImplicitPersona benchmark')
    parser.add_argument('--benchmark_file', type=str, default='data/benchmark.csv',
                       help='Path to benchmark CSV file')
    parser.add_argument('--eval_mode', type=str, choices=['mcq', 'generative'], default='mcq',
                       help='Evaluation mode: mcq or generative')
    parser.add_argument('--use_multimodal', action='store_true',
                       help='Use multimodal chat history instead of regular chat history')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to process (for testing)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PersonaBenchmarkEvaluator(args.config)
    
    # Run evaluation
    output_file = evaluator.run_evaluation(
        args.benchmark_file,
        args.eval_mode,
        args.use_multimodal,
        args.max_items
    )
    
    print(f"\nEvaluation completed. Results saved to: {output_file}")


if __name__ == "__main__":
    main()