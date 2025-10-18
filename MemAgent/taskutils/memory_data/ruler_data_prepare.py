# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare jsonl with field `input` and `outputs`.
{
    "index" int,
    "input": str,
    "context": str,
    "outputs": [str],
}

Example usage:
python prepare.py \
    --benchmark synthetic \
    --task niah_single_1 \
    --tokenizer_path tokenizer.model \
    --tokenizer_type nemo \
    --max_seq_length 4096 \
    --model_template_type base \
    --num_samples 10
"""

import os
import argparse
import importlib
import subprocess
import time
import yaml
from pathlib import Path
from template import Templates
from tqdm import tqdm
import nltk
from synthetic.nemo import read_manifest, write_manifest

# Ensure nltk punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--data_dir", type=Path, default='data', help='Directory of the raw data')
parser.add_argument("--task", type=str, required=True, help='Task name in benchmark')
parser.add_argument("--tokenizer_path", type=str, required=True, help='Path to the tokenizer model')
parser.add_argument("--tokenizer_type", type=str, default='nemo', help='Options: nemo, hf, openai')
parser.add_argument("--max_seq_length", type=int, required=True, help='Max sequence length including all input tokens and generated tokens.')
parser.add_argument("--num_samples", type=int, default=500, help='Number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--model_template_type", type=str, default='base', help='Template type defined in template.py')
parser.add_argument("--remove_newline_tab", action='store_true', help='Remove \\n and \\t in all strings.')
parser.add_argument("--chunk_amount", type=int, default=1, help='Total number of chunks:parallel_size')
args = parser.parse_args()
save_dir = args.data_dir
import tempfile
temp_dir = Path(tempfile.mkdtemp())
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
save_name = f"eval_{args.task}_{args.max_seq_length}.json"

# Function to run one chunk
def run_chunk(chunk_args):
    chunk_idx, num_samples, pre_samples, args, config, curr_folder = chunk_args
    random_seed = args.random_seed + chunk_idx
    if chunk_idx == 0:
        print(temp_dir / (save_name + f".{chunk_idx}"))
    script = os.path.join(curr_folder, f"{args.benchmark}/{config['task']}.py")
    additional_args = " ".join([f"--{k} {v}" for k, v in config['args'].items()])
    command = f"""python {script} \
        --save_dir  {temp_dir} \
        --save_name {save_name + f".{chunk_idx}"} \
        --tokenizer_path {args.tokenizer_path} \
        --tokenizer_type {args.tokenizer_type} \
        --max_seq_length {args.max_seq_length} \
        --tokens_to_generate {config['tokens_to_generate']} \
        --num_samples {num_samples} \
        --random_seed {random_seed} \
        {additional_args} \
        {f"--remove_newline_tab" if args.remove_newline_tab else ""} \
        {f"--pre_samples {pre_samples}" if config['task'] == 'qa' else ""} \
        --template "{config['template']}" \
        --answer_prefix "{config.get('answer_prefix', '')}" 
    """
    return subprocess.Popen(command, shell=True, text=True, stdout=None if chunk_idx == 0 else subprocess.DEVNULL)



# Main function
def main():

    start_time = time.time()
    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
    except ImportError:
        raise ImportError(f"Module {args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"synthetic/{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in {args.benchmark}.yaml')

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    chunk_save_file = save_dir / save_name
    file_exists = False
    if os.path.exists(chunk_save_file):
        import json
        with open(chunk_save_file, "r") as f:
            data = json.load(f)
        if len(data) >= args.num_samples:
            file_exists = True
    if file_exists:
        print(f"✅ {chunk_save_file} already exists. Skipping.")
        return

    # Prepare template
    assert args.model_template_type in Templates, f'{args.model_template_type} not found in Templates'
    model_template = Templates[args.model_template_type]
    task_template = config['template']
    answer_prefix = config.get('answer_prefix', '')
    config['template'] = model_template.format(task_template=task_template) #+ answer_prefix
    config['answer_prefix'] = answer_prefix
    # Chunking logic
    chunks = [(args.num_samples // args.chunk_amount) + (1 if i < args.num_samples % args.chunk_amount else 0) for i in range(args.chunk_amount)]
    chunk_args = [(i, chunks[i], sum(chunks[:i]), args, config, curr_folder) for i in range(args.chunk_amount)]

    # Parallel execution
    processes = []
    for i in range(args.chunk_amount):
        p = run_chunk(chunk_args[i])
        if p:
            processes.append(p)
    for i, p in enumerate(processes):
        p.wait()
        if p.returncode!= 0:
            for p in processes:
                p.kill()
            raise RuntimeError(f"Chunk {i} failed with return code {p.returncode}")

    # Gather
    import glob
    files = glob.glob(str(temp_dir / f"eval_{args.task}_{args.max_seq_length}.json.*"))
    assert len(files) == args.chunk_amount, f"Expected {args.chunk_amount} files, but found {len(files)}"
    data = []
    for ifile in files:
        data.extend(read_manifest(ifile))
    import json
    with open(save_dir / save_name, "w") as f:
        json.dump(data, f, indent=4)
    for ifile in files:
        os.remove(ifile)
            

    print(f"\n✅ Prepared {args.task} with {args.num_samples} samples → {save_dir / save_name}")
    print(f"⏱️ Total time: {round((time.time() - start_time) / 60, 1)} minutes")

if __name__ == '__main__':
    print(args)
    print(save_name)
    main()
