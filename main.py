import yaml
import argparse
import sys
import os
import json

from contexts_builder import build_context
from qa_generator import generate_qas
from utils import save_json, save_csv
from conv_generator import generate_interactions_from_persona
from query_llm import QueryLLM


def main():
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--model', type=str, default="gpt-4.1", help='Set LLM model.')
    parser.add_argument('--conv_output_path', type=str, default='data/interactions.jsonl', help='Set the path to the output directory')
    parser.add_argument('--result_path', type=str, default='results/', help='Set the path to the output directory')
    parser.add_argument('--num_persona', type=int, default=1, help='Number of personas to generate')
    parser.add_argument('--data_types', type=str, default="email", nargs="+", help='Conversation types for the user to implicitly express their preferences')
    parser.add_argument('--context_length', type=int, default=32000, help='Length of the total context to be generated, including irrelevant tokens')
    parser.add_argument('--self_verify', dest='self_verify', action='store_true', help='Set self_verify to True')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Start the data from cratch')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['data']['conv_output_path'] = cmd_args.conv_output_path if cmd_args.conv_output_path is not None else args['data']['conv_output_path']
    args['data']['result_path'] = cmd_args.result_path if cmd_args.result_path is not None else args['data']['result_path']
    args['data']['num_persona'] = cmd_args.num_persona if cmd_args.num_persona is not None else args['inference']['num_persona']
    args['data']['data_types'] = cmd_args.data_types if cmd_args.data_types is not None else args['data']['data_types']
    args['data']['context_length'] = cmd_args.context_length if cmd_args.context_length is not None else args['data']['context_length']
    args['data']['clean'] = cmd_args.clean if cmd_args.clean is not None else args['data']['clean']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    args['data']['self_verify'] = cmd_args.self_verify if cmd_args.self_verify is not None else args['data']['self_verify']
    print(args)

    # Build persona and preferences
    with open(args['data']['persona_path'], 'r') as file:
        all_personas = file.readlines()

    # Ensure data_types is always a list
    if isinstance(args['data']['data_types'], str):
        args['data']['data_types'] = [args['data']['data_types']]

    llm = QueryLLM(args)

    # find if the conv_output_path already exists
    # if not os.path.exists(args['data']['conv_output_path']):
    output_dict = generate_interactions_from_persona(llm, all_personas, output_path=args['data']['conv_output_path'], implicit_types=args['data']['data_types'],
                                                     num_persona=args['data']['num_persona'], self_verify=args['data']['self_verify'], clean=args['data']['clean'], verbose=args['inference']['verbose'])
    # else:
    #     print(f"File {args['data']['conv_output_path']} already exists. Loading existing interactions.")
    #     with open(args['data']['conv_output_path'], 'r') as file:
    #         output_dict = json.load(file)

    # # Build single long context
    # build_context(output_dict, args['data']['irrelevant_context_path'], args['data']['context_length'])

    # # Generate all Q&A rows
    # qas = generate_qas('interactions.json', context_id)

if __name__ == '__main__':
    main()
