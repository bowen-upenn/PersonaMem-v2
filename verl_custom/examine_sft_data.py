#!/usr/bin/env python3
"""
Examine SFT training data and show examples.
"""

import pandas as pd
import json
import argparse
from typing import Dict, Any, List

def examine_sft_data(parquet_path: str, num_examples: int = 5):
    """
    Load and examine SFT data, showing structure and examples.
    """
    print(f"Loading SFT data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if len(df) == 0:
        print("Empty dataset!")
        return
    
    # Show data types and basic stats
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Count MCQ vs regular samples if the column exists
    if 'is_mcq' in df.columns:
        mcq_counts = df['is_mcq'].value_counts()
        total = len(df)
        print(f"\nSample types:")
        for mcq_type, count in mcq_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            sample_type = "MCQ format" if mcq_type else "Regular format"
            print(f"  {sample_type}: {count} samples ({percentage:.1f}%)")
    
    # Analyze conversation lengths
    conversation_lengths = []
    message_counts = []
    
    print(f"\n=== Analyzing Conversations ===")
    for idx, row in df.head(100).iterrows():  # Sample first 100 for stats
        try:
            messages = json.loads(row['messages'])
            if isinstance(messages, list):
                message_counts.append(len(messages))
                # Calculate total character length
                total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
                conversation_lengths.append(total_chars)
        except:
            pass
    
    if conversation_lengths:
        import numpy as np
        print(f"Conversation statistics (first 100 samples):")
        print(f"  Average messages per conversation: {np.mean(message_counts):.1f}")
        print(f"  Message count range: {min(message_counts)} - {max(message_counts)}")
        print(f"  Average character length: {np.mean(conversation_lengths):.0f}")
        print(f"  Character length range: {min(conversation_lengths)} - {max(conversation_lengths)}")
    
    # Show detailed examples
    print(f"\n=== Example Conversations ===")
    examples_shown = 0
    
    # Special logic: if num_examples == 2, show one MCQ and one non-MCQ example
    if num_examples == 2 and 'is_mcq' in df.columns:
        mcq_shown = False
        non_mcq_shown = False
        
        for idx, row in df.iterrows():
            if examples_shown >= num_examples:
                break
                
            try:
                messages = json.loads(row['messages'])
                is_mcq = row.get('is_mcq', False)
                
                if not isinstance(messages, list):
                    continue
                
                # Skip if we already have this type
                if is_mcq and mcq_shown:
                    continue
                if not is_mcq and non_mcq_shown:
                    continue
                    
                print(f"\n--- Example {examples_shown + 1} ({'MCQ' if is_mcq else 'Regular'}) ---")
                print(f"Original index: {row.get('original_idx', 'N/A')}")
                print(f"MCQ format: {is_mcq}")
                print(f"Number of messages: {len(messages)}")
                
                # Show only the last 3 rounds (6 messages max: 3 user + 3 assistant)
                # But ensure we show complete rounds
                messages_to_show = messages[-6:] if len(messages) > 6 else messages
                
                if len(messages) > 6:
                    print(f"  ... [showing last 3 rounds out of {len(messages)} total messages] ...")
                
                # Show the selected messages
                for i, msg in enumerate(messages_to_show):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    
                    # Truncate very long individual messages for readability
                    if len(content) > 1000:
                        display_content = content[:500] + f"...[{len(content)-1000} chars truncated]..." + content[-500:]
                    else:
                        display_content = content
                    
                    # Calculate the actual message number in the full conversation
                    actual_msg_num = len(messages) - len(messages_to_show) + i + 1
                    print(f"  Message {actual_msg_num} [{role}]: {display_content}")
                
                # Show if this appears to be MCQ format
                last_user_msg = None
                for msg in reversed(messages):
                    if msg.get('role') == 'user':
                        last_user_msg = msg.get('content', '')
                        break
                
                if last_user_msg and ('\\boxed{' in last_user_msg or 'multiple-choice' in last_user_msg.lower()):
                    print(f"  --> This appears to be MCQ format (detected from user message)")
                
                # Track what we've shown
                if is_mcq:
                    mcq_shown = True
                else:
                    non_mcq_shown = True
                    
                examples_shown += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
    else:
        # Default behavior: show examples sequentially
        for idx, row in df.iterrows():
            if examples_shown >= num_examples:
                break
                
            try:
                messages = json.loads(row['messages'])
                
                if not isinstance(messages, list):
                    continue
                    
                print(f"\n--- Example {examples_shown + 1} ---")
                print(f"Original index: {row.get('original_idx', 'N/A')}")
                print(f"MCQ format: {row.get('is_mcq', 'N/A')}")
                print(f"Number of messages: {len(messages)}")
                
                # Show only the last 3 rounds (6 messages max: 3 user + 3 assistant)
                # But ensure we show complete rounds
                messages_to_show = messages[-6:] if len(messages) > 6 else messages
                
                if len(messages) > 6:
                    print(f"  ... [showing last 3 rounds out of {len(messages)} total messages] ...")
                
                # Show the selected messages
                for i, msg in enumerate(messages_to_show):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    
                    # Truncate very long individual messages for readability
                    if len(content) > 1000:
                        display_content = content[:500] + f"...[{len(content)-1000} chars truncated]..." + content[-500:]
                    else:
                        display_content = content
                    
                    # Calculate the actual message number in the full conversation
                    actual_msg_num = len(messages) - len(messages_to_show) + i + 1
                    print(f"  Message {actual_msg_num} [{role}]: {display_content}")
                
                # Show if this appears to be MCQ format
                last_user_msg = None
                for msg in reversed(messages):
                    if msg.get('role') == 'user':
                        last_user_msg = msg.get('content', '')
                        break
                
                if last_user_msg and ('\\boxed{' in last_user_msg or 'multiple-choice' in last_user_msg.lower()):
                    print(f"  --> This appears to be MCQ format (detected from user message)")
                
                examples_shown += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
    
    print(f"\nShown {examples_shown} example conversations.")

def main():
    parser = argparse.ArgumentParser(description="Examine SFT training data")
    parser.add_argument("--data_dir", default="verl_custom/data/implicit_persona_sft",
                       help="Directory containing SFT parquet files")
    parser.add_argument("--file", default="train.parquet",
                       choices=["train.parquet", "val.parquet"], 
                       help="Which file to examine")
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of conversation examples to show")
    
    args = parser.parse_args()
    
    parquet_path = f"{args.data_dir}/{args.file}"
    
    print("=" * 80)
    print(f"EXAMINING SFT DATA: {args.file}")
    print("=" * 80)
    
    examine_sft_data(parquet_path, args.num_examples)

if __name__ == "__main__":
    main()