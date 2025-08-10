#!/usr/bin/env python3

import json
import csv
import os
import glob
import re
import tiktoken
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional

# Initialize the tokenizer
ENCODER = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    return len(ENCODER.encode(str(text)))

def extract_persona_number_from_filename(filename: str) -> Optional[int]:
    """Extract persona number from filename pattern."""
    match = re.search(r'persona(\d+)\.json$', filename)
    return int(match.group(1)) if match else None

def load_persona_data(raw_data_dir: str) -> Dict[str, Any]:
    """Load all persona data from raw_data directory."""
    persona_data = {}
    
    pattern = os.path.join(raw_data_dir, "interactions_*_persona*.json")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} persona files")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract persona number for reference
            persona_num = extract_persona_number_from_filename(file_path)
            
            for uuid, persona_info in data.items():
                persona_data[uuid] = {
                    'data': persona_info,
                    'file_path': file_path,
                    'persona_number': persona_num
                }
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(persona_data)} personas")
    return persona_data

def load_context_data(contexts_dir: str) -> Dict[str, Any]:
    """Load all context data from contexts directory."""
    context_data = {}
    
    pattern = os.path.join(contexts_dir, "context_*.json")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} context files")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract persona UUID from metadata if available
            metadata = data.get('metadata', {})
            persona_uuid = metadata.get('persona_uuid')
            
            if persona_uuid:
                context_data[persona_uuid] = {
                    'metadata': metadata,
                    'messages': data.get('messages', []),
                    'file_path': file_path
                }
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(context_data)} context files")
    return context_data

def find_conversation_turns_and_tokens_to_end(target_conversation: List[Dict], all_messages: List[Dict]) -> tuple[int, int]:
    """
    Find the number of turns and tokens from the target conversation to the end of context.
    
    Returns:
        tuple: (number_of_turns_to_end, tokens_to_end)
    """
    if not target_conversation or not all_messages:
        return -1, -1
    
    # Look for the first message of the target conversation
    target_first_msg = target_conversation[0].get('content', '')
    
    # Find the starting position of the target conversation
    target_start_idx = -1
    
    for i in range(len(all_messages)):
        current_msg = all_messages[i]
        
        # Check if this message matches the start of our target conversation
        if current_msg.get('content', '') == target_first_msg:
            # Verify it's actually our target conversation by checking a few more messages
            match = True
            for j, target_msg in enumerate(target_conversation[:3]):  # Check first 3 messages
                if i + j >= len(all_messages):
                    match = False
                    break
                if all_messages[i + j].get('content', '') != target_msg.get('content', ''):
                    match = False
                    break
            
            if match:
                target_start_idx = i
                break
    
    if target_start_idx == -1:
        return -1, -1
    
    # Count turns from target conversation to end
    turns_to_end = 0
    tokens_to_end = 0
    
    # Start counting from the target conversation to the end
    for i in range(target_start_idx, len(all_messages)):
        msg = all_messages[i]
        
        # Count turns (each user message starts a new turn)
        if msg.get('role') == 'user':
            turns_to_end += 1
        
        # Count tokens in this message
        content = msg.get('content', '')
        if isinstance(content, str):
            tokens_to_end += count_tokens(content)
        elif isinstance(content, list):
            # Handle multimodal content - only count text parts
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    tokens_to_end += count_tokens(item['text'])
                elif isinstance(item, str):
                    tokens_to_end += count_tokens(item)
    
    return turns_to_end, tokens_to_end

def categorize_topic(preference: str, conversation_scenario: str) -> str:
    """Categorize the topic based on preference content and scenario."""
    preference_lower = preference.lower()
    
    # Define topic keywords
    topic_keywords = {
        'finance': ['financial', 'money', 'investment', 'stock', 'market', 'savings', 'retirement', 'pension'],
        'food': ['food', 'meal', 'cooking', 'restaurant', 'dining', 'cuisine', 'recipe'],
        'travel': ['travel', 'vacation', 'trip', 'destination', 'flight', 'hotel'],
        'entertainment': ['movie', 'music', 'book', 'game', 'show', 'entertainment'],
        'technology': ['tech', 'digital', 'computer', 'software', 'app', 'internet'],
        'health': ['health', 'medical', 'fitness', 'exercise', 'wellness', 'therapy'],
        'fashion': ['fashion', 'clothing', 'style', 'dress', 'outfit'],
        'sports': ['sport', 'football', 'baseball', 'golf', 'game', 'team'],
        'work': ['work', 'job', 'career', 'professional', 'business', 'office'],
        'social': ['social', 'friend', 'family', 'relationship', 'gathering'],
        'pets': ['pet', 'dog', 'cat', 'animal', 'puppy'],
        'home': ['home', 'house', 'garden', 'diy', 'improvement']
    }
    
    # Check for topic keywords in preference
    for topic, keywords in topic_keywords.items():
        if any(keyword in preference_lower for keyword in keywords):
            return topic
    
    # Fallback based on conversation scenario
    scenario_topics = {
        'multimodal': 'visual',
        'personal_email': 'communication',
        'professional_email': 'work',
        'social_media_post': 'social'
    }
    
    return scenario_topics.get(conversation_scenario, 'general')

def extract_qa_pairs(persona_data: Dict[str, Any], context_data: Dict[str, Any], token_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Extract QA pairs from persona and context data."""
    qa_pairs = []
    
    for uuid, persona_info in persona_data.items():
        persona = persona_info['data']
        context_info = context_data.get(uuid)
        
        if not context_info:
            print(f"No context found for persona {uuid}")
            continue
        
        # Check if this context fits within token limit
        if token_limit and context_info['metadata'].get('final_token_count', 0) > token_limit:
            continue
        
        # Extract persona metadata (everything before stereotypical_preferences)
        persona_metadata = {}
        for key, value in persona.items():
            if key == 'stereotypical_preferences':
                break
            persona_metadata[key] = value
        
        conversations = persona.get('conversations', {})
        context_messages = context_info['messages']
        
        for conv_type, conv_list in conversations.items():
            for conv_elem in conv_list:
                # Skip if no QA data
                if 'user_query' not in conv_elem or 'correct_answer' not in conv_elem:
                    continue
                
                # Extract basic information
                user_query = conv_elem.get('user_query', '')
                correct_answer = conv_elem.get('correct_answer', '')
                incorrect_answers = conv_elem.get('incorrect_answers', [])
                
                # Create all_answers list
                all_answers = [correct_answer] + list(incorrect_answers)
                
                # Extract metadata
                preference = conv_elem.get('preference', '')
                pref_type = conv_elem.get('pref_type', '')
                who = conv_elem.get('who', '')
                updated = conv_elem.get('updated', False)
                prev_pref = conv_elem.get('prev_pref', '')
                sensitive_info = conv_elem.get('sensitive_info', '')
                
                # Handle sensitive info case
                if sensitive_info:
                    groundtruth_preference = sensitive_info
                    preference_type = 'sensitive_info'
                else:
                    groundtruth_preference = preference
                    preference_type = pref_type
                
                # Calculate distance to target conversation
                target_conversation = conv_elem.get('conversations', [])
                turns_to_end, tokens_to_end = find_conversation_turns_and_tokens_to_end(target_conversation, context_messages)
                
                # Categorize topics
                preference_topic = categorize_topic(groundtruth_preference, conv_type)
                query_topic = categorize_topic(user_query, conv_type)
                
                # Count tokens in query
                query_tokens = count_tokens(user_query)
                
                qa_pair = {
                    'persona_id': f"persona{persona_info['persona_number']}",
                    'question': user_query,
                    'correct_answer': correct_answer,
                    'all_answers': all_answers,
                    'groundtruth_preference': groundtruth_preference,
                    'preference_type': preference_type,
                    'conversation_scenario': conv_type,
                    'preference_topic': preference_topic,
                    'query_topic': query_topic,
                    'who': who,
                    'updated': updated,
                    'previous_preference': prev_pref,
                    'persona': json.dumps(persona_metadata),
                    'context_file_path': context_info['file_path'],
                    'num_tokens': context_info['metadata'].get('final_token_count', 0),
                    'number_turns_from_query_to_the_target_conversation': turns_to_end,
                    'tokens_from_query_to_the_target_conversation': tokens_to_end,
                    'persona_uuid': uuid,
                    'query_tokens': query_tokens
                }
                
                qa_pairs.append(qa_pair)
    
    return qa_pairs

def save_benchmark_data(qa_pairs: List[Dict[str, Any]], output_dir: str, token_limit: Optional[int] = None):
    """Save QA pairs to CSV and JSON formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    if token_limit:
        suffix = f"_{token_limit//1000}k"
    else:
        suffix = "_unlimited"
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"benchmark{suffix}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        if qa_pairs:
            fieldnames = qa_pairs[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for qa_pair in qa_pairs:
                # Convert lists to JSON strings for CSV
                row = qa_pair.copy()
                row['all_answers'] = json.dumps(row['all_answers'])
                writer.writerow(row)
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"benchmark{suffix}.json")
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(qa_pairs, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(qa_pairs)} QA pairs to {csv_path} and {json_path}")

def generate_summary_stats(qa_pairs: List[Dict[str, Any]], token_limit: Optional[int] = None):
    """Generate and print summary statistics."""
    if not qa_pairs:
        print("No QA pairs to analyze")
        return
    
    df = pd.DataFrame(qa_pairs)
    
    print(f"\n=== Benchmark Summary ({'Unlimited' if token_limit is None else f'{token_limit//1000}k'} tokens) ===")
    print(f"Total QA pairs: {len(qa_pairs)}")
    print(f"Unique personas: {df['persona_id'].nunique()}")
    
    print(f"\nBy preference type:")
    print(df['preference_type'].value_counts())
    
    print(f"\nBy conversation scenario:")
    print(df['conversation_scenario'].value_counts())
    
    print(f"\nBy preference topic:")
    print(df['preference_topic'].value_counts())
    
    print(f"\nToken distribution:")
    print(f"  Mean: {df['num_tokens'].mean():.0f}")
    print(f"  Median: {df['num_tokens'].median():.0f}")
    print(f"  Min: {df['num_tokens'].min()}")
    print(f"  Max: {df['num_tokens'].max()}")

def main():
    """Main function to prepare benchmark data."""
    # Define paths
    raw_data_dir = "data/raw_data"
    contexts_dir = "data/contexts"
    output_dir = "data/benchmark"
    
    print("Loading persona data...")
    persona_data = load_persona_data(raw_data_dir)
    
    print("Loading context data...")
    context_data = load_context_data(contexts_dir)
    
    # Generate different token limit versions
    token_limits = [32000, 128000, None]  # 32k, 128k, unlimited
    
    for token_limit in token_limits:
        print(f"\nProcessing {'unlimited' if token_limit is None else f'{token_limit//1000}k'} token version...")
        
        qa_pairs = extract_qa_pairs(persona_data, context_data, token_limit)
        
        if qa_pairs:
            save_benchmark_data(qa_pairs, output_dir, token_limit)
            generate_summary_stats(qa_pairs, token_limit)
        else:
            print(f"No QA pairs found for {'unlimited' if token_limit is None else f'{token_limit//1000}k'} token limit")
    
    print(f"\nBenchmark preparation complete! Files saved to {output_dir}")

if __name__ == "__main__":
    main() 