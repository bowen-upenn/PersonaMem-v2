#!/usr/bin/env python3

import json
import os
from contexts_builder import build_context

# Test with the persona file
test_file = "data/raw_data/interactions_250727_201452_persona63.json"

if os.path.exists(test_file):
    print(f"Testing context builder with {test_file}")
    
    # Load the test data
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Create contexts directory if it doesn't exist
    os.makedirs("data/contexts", exist_ok=True)
    
    # Test the build_context function with filename
    build_context(test_data, context_len=15000, input_filename=test_file)
    
    # Check the output - look for the new naming convention
    expected_filename = "data/contexts/context_250727_201452_persona63.json"
    if os.path.exists(expected_filename):
        with open(expected_filename, 'r') as f:
            output_data = json.load(f)
        print(f"\nOutput file structure:")
        print(f"  Metadata: {output_data.get('metadata', {})}")
        print(f"  Messages count: {len(output_data.get('messages', []))}")
    else:
        print(f"Expected file {expected_filename} not found")
        # Check for fallback naming
        for uuid in test_data.keys():
            fallback_file = f"data/contexts/context_{uuid}.json"
            if os.path.exists(fallback_file):
                print(f"Found fallback file: {fallback_file}")
            break
else:
    print(f"Test file {test_file} not found.") 