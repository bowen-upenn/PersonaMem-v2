import yaml
import argparse
import sys
import os

from contexts_builder import build_context
from qa_generator import generate_qas
from data_manager import save_json, save_csv
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
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm_model'] = cmd_args.model if cmd_args.model is not None else args['models']['llm_model']
    args['data']['conv_output_path'] = cmd_args.conv_output_path if cmd_args.conv_output_path is not None else args['data']['conv_output_path']
    args['data']['result_path'] = cmd_args.result_path if cmd_args.result_path is not None else args['data']['result_path']
    args['inference']['num_persona'] = cmd_args.num_persona if cmd_args.num_persona is not None else args['inference']['num_persona']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    print(args)

    # Build persona and preferences
    with open(args['data']['persona_path'], 'r') as file:
        all_personas = file.readlines()

    llm = QueryLLM(args)
    llm.create_a_thread()

    generate_interactions_from_persona(llm, all_personas, args['data']['conv_output_path'], args['inference']['num_persona'], args['inference']['verbose'])

    # Build single long context
    contexts = build_context(args['data']['conv_output_path'], args['data']['irrelevant_context_path'])
    context_id = list(contexts.keys())[0]

    # # Generate all Q&A rows
    # qas = generate_qas('interactions.json', context_id)

    # Save outputs
    save_json(contexts, os.path.join(args['data']['result_path'], 'contexts.json'))
    # save_csv(qas, os.path.join(args['data']['result_path'], 'qas.csv'))
    print("Outputs saved to contexts.json and qas.csv")

if __name__ == '__main__':
    main()
