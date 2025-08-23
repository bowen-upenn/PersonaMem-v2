# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import random
from multiprocessing import Pool
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path

# Global variables
QAS = None
DOCS = None
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# SQuAD dataset processing
def read_squad(file):
    with open(file) as f:
        data = json.load(f)
        
    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })
    return total_qas, total_docs

# HotpotQA dataset processing
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)
    
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })
    return total_qas, total_docs

def generate_input_output(index, num_docs):
    global QAS, DOCS
    curr_q = QAS[index]['query']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])
    
    if num_docs < len(DOCS):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS
    
    random.Random(4).shuffle(all_docs)
    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    
    formatted_output = {
        "data_source": "hotpotqa",
        "prompt": [{
            "role": "user",
            "content": curr_q,
        }],
        "context": context,
        "ability": "memory",
        "reward_model": {
            "style": "rule",
            "ground_truth": curr_a
        },
        "extra_info": {
            'index': index,
            "question": curr_q,
            "num_docs": num_docs,
        }
    }
    return formatted_output
def generate_dataset(num_samples: int, save_dir: str, incremental: int = 10, qas=None, docs=None):
    global QAS, DOCS
    if qas is None or docs is None:
        raise ValueError("QAS and DOCS must be provided.")
    
    QAS = qas
    DOCS = docs
    
    length = min(num_samples, len(QAS))
    print("start")
    
    from utils import TqdmExecutor
    write_jsons = TqdmExecutor(max_workers=os.cpu_count()).run(generate_input_output, range(length), num_docs=incremental)
    # tokens = [len(x) for x in tokenizer([j['context'] for j in write_jsons])['input_ids']]
    # print(max(tokens), min(tokens), sum(tokens) / len(tokens))
    # Save to Parquet file
    df = pd.DataFrame(write_jsons)
    df.to_parquet(save_dir + ".parquet")
    return write_jsons

if __name__ == "__main__":
    random.seed(42)
    
    QAS_train, DOCS_train = read_hotpotqa('hotpotqa_train.json')
    generate_dataset(80000, 'hotpotqa_train_process', 200, QAS_train, DOCS_train)
    QAS_dev, DOCS_dev = read_hotpotqa('hotpotqa_dev.json')
    generate_dataset(200, 'hotpotqa_dev_process', 200, QAS_dev, DOCS_dev)