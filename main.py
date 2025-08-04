import yaml
import argparse
import sys
import os
import json

from contexts_builder import build_context
from qa_generator import generate_qa
from categorize_topic import categorize_topics
from utils import save_json, save_csv
from conv_generator import generate_interactions_from_persona
from query_llm import QueryLLM
from image_matcher import ImageMatcher
import utils


def main():
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4.1", help='Set LLM model. Only applicable for OpenAI. For Microsoft Azure, set the model in .env file.')
    parser.add_argument('--step', type=str, default='generate_data', help='Choose generate_data, generate_qa, categorize_topics, build_context, or run_eval.')
    parser.add_argument('--conv_output_dir', type=str, default='data/raw_data/', help='Set the directory for conversation data output')
    parser.add_argument('--qa_output_dir', type=str, default='data/raw_data/', help='Set the directory for QA data output')
    parser.add_argument('--result_path', type=str, default='results/', help='Set the path to the output directory')
    parser.add_argument('--num_persona', type=int, default=1, help='Number of personas to generate')
    parser.add_argument('--persona_start_idx', type=int, default=-1, help='Starting persona index for QA generation (-1 for all)')
    parser.add_argument('--persona_end_idx', type=int, default=-1, help='Ending persona index for QA generation (-1 for all)')
    parser.add_argument('--data_types', type=str, default="email", nargs="+", help='Conversation types for the user to implicitly express their preferences')
    parser.add_argument('--context_length', type=int, default=32000, help='Length of the total context to be generated, including irrelevant tokens')
    parser.add_argument('--self_verify', dest='self_verify', action='store_true', help='Set self_verify to True')
    parser.add_argument('--rate_limit_per_min', type=int, default=10, help='Rate limit for API calls per minute')
    parser.add_argument('--parallel', dest='parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing data and start from scratch')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    parser.add_argument('--validate_qa', dest='validate_qa', action='store_true', help='Enable QA validation to filter out problematic pairs')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['inference']['step'] = cmd_args.step if cmd_args.step is not None else args['inference']['step']
    args['data']['conv_output_dir'] = cmd_args.conv_output_dir if cmd_args.conv_output_dir is not None else args['data'].get('conv_output_dir', 'data/raw_data/')
    args['data']['qa_output_dir'] = cmd_args.qa_output_dir if cmd_args.qa_output_dir is not None else args['data'].get('qa_output_dir', 'data/raw_data/')
    args['data']['result_path'] = cmd_args.result_path if cmd_args.result_path is not None else args['data']['result_path']
    args['data']['num_persona'] = cmd_args.num_persona if cmd_args.num_persona is not None else args['inference']['num_persona']
    args['data']['data_types'] = cmd_args.data_types if cmd_args.data_types is not None else args['data']['data_types']
    args['data']['context_length'] = cmd_args.context_length if cmd_args.context_length is not None else args['data']['context_length']
    args['inference']['rate_limit_per_min'] = cmd_args.rate_limit_per_min if cmd_args.rate_limit_per_min is not None else args['inference']['rate_limit_per_min']
    args['inference']['parallel'] = cmd_args.parallel if cmd_args.parallel is not None else args['inference'].get('parallel', False)
    args['data']['clean'] = cmd_args.clean if cmd_args.clean is not None else args['data']['clean']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    args['data']['self_verify'] = cmd_args.self_verify if cmd_args.self_verify is not None else args['data']['self_verify']
    args['inference']['validate_qa'] = cmd_args.validate_qa if cmd_args.validate_qa is not None else args['inference'].get('validate_qa', False)
    args['data']['persona_start_idx'] = cmd_args.persona_start_idx if cmd_args.persona_start_idx is not None else -1
    args['data']['persona_end_idx'] = cmd_args.persona_end_idx if cmd_args.persona_end_idx is not None else -1
    print(args)

    # Build persona and preferences
    with open(args['data']['persona_path'], 'r') as file:
        all_personas = file.readlines()

    # Ensure data_types is always a list
    if isinstance(args['data']['data_types'], str):
        args['data']['data_types'] = [args['data']['data_types']]

    llm = QueryLLM(args, args['inference']['rate_limit_per_min'])
    image_matcher = ImageMatcher(llm)
    image_matcher.load_or_create_embeddings_cache()

    if args['inference']['step'] == 'generate_data':
        # Create timestamped output path
        timestamped_path = utils.create_timestamped_filename(
            args['data']['conv_output_dir'], 
            'interactions', 
            '.json'
        )
        
        print(f"Saving interactions to: {timestamped_path}")
        
        output_dict = generate_interactions_from_persona(llm, all_personas, image_matcher, output_path=timestamped_path, implicit_types=args['data']['data_types'],
                                                         num_persona=args['data']['num_persona'], self_verify=args['data']['self_verify'], clean=args['data']['clean'], 
                                                         parallel=args['inference']['parallel'], verbose=args['inference']['verbose'])
    elif args['inference']['step'] == 'generate_qa':
        # Get persona files within the specified range
        persona_files = utils.get_persona_files_in_range(
            args['data']['conv_output_dir'],
            'interactions',
            args['data']['persona_start_idx'],
            args['data']['persona_end_idx']
        )
        
        if not persona_files:
            print("No persona files found in the specified range.")
            return
        
        print(f"Found {len(persona_files)} persona files to process")
        
        # QA pairs will be added directly to the original persona files in conv_output_dir
        qa_output_path = args['data']['conv_output_dir']
        
        # Generate Q&A using the updated generate_qa function that supports parallel processing
        generate_qa(llm, persona_files, qa_output_path, parallel=args['inference']['parallel'], 
                   verbose=args['inference']['verbose'], validate_qa=args['inference']['validate_qa'])

    elif args['inference']['step'] == 'categorize_topics':
        # Get persona files within the specified range
        persona_files = utils.get_persona_files_in_range(
            args['data']['conv_output_dir'],
            'interactions',
            args['data']['persona_start_idx'],
            args['data']['persona_end_idx']
        )
        
        if not persona_files:
            print("No persona files found in the specified range.")
            return
        
        print(f"Found {len(persona_files)} persona files to categorize topics")
        
        # Categorize topics using the categorize_topics function
        categorize_topics(llm, persona_files, output_dir=None, parallel=args['inference']['parallel'], 
                         verbose=args['inference']['verbose'])

    # Build long context
    elif args['inference']['step'] == 'build_context':
        # Get persona files within the specified range
        persona_files = utils.get_persona_files_in_range(
            args['data']['conv_output_dir'],
            'interactions',
            args['data']['persona_start_idx'],
            args['data']['persona_end_idx']
        )
        
        if not persona_files:
            print("No persona files found in the specified range.")
            return
        
        print(f"Found {len(persona_files)} persona files to build context from")
        
        # Load and merge all persona data
        merged_data = {}
        for file_path in persona_files:
            with open(file_path, 'r') as file:
                file_data = json.load(file)
                merged_data.update(file_data)
        
        build_context(merged_data, args['data']['context_length'])

    elif args['inference']['step'] == 'run_eval':
        pass
    else:
        raise KeyError

if __name__ == '__main__':
    main()
