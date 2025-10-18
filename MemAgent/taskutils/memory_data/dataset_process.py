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
import shutil
import glob
import urllib.request
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
import pyarrow as pa
import pyarrow.parquet as pq  # 显式导入 parquet
from pathlib import Path

# squad数据集每段一个问题
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

# hotpotqa数据集每篇文章一个问题
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

file = 'hotpotqa_train.json'
QAS, DOCS = read_hotpotqa(file)

def generate_input_output(index, num_docs, QAS, DOCS):
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
    template = "{context}\n\n{query}"
    input_text = context 
    query = curr_q

    return input_text, query, curr_a #input_text:拼接的文本,query:问题,curr_a:答案

def generate_dataset(num_samples: int, save_dir: str, incremental: int = 10, QAS=None, DOCS=None):
    if QAS is None or DOCS is None:
        raise ValueError("QAS and DOCS must be provided.")
    
    write_jsons = []
    for index in range(min(num_samples, len(QAS))):
        used_docs = incremental
        input_text, question, answer = generate_input_output(index, used_docs, QAS, DOCS)
        formatted_output = {
            "question": question,
            "solution": answer,
            "context": input_text
        }
        write_jsons.append(formatted_output)
    
    # 保存为 Parquet 文件
    table = pa.Table.from_pylist(write_jsons)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pq.write_table(table, f"{save_dir}.parquet")  # 使用 pq.write_table
    
    return write_jsons

if __name__ == "__main__":
    random.seed(42)

    QAS_train, DOCS_train = read_hotpotqa('hotpotqa_train.json')
    QAS_dev, DOCS_dev = read_hotpotqa('hotpotqa_dev.json')
    
    # generate_dataset(7000, 'hotpotqa_train_process', 300, QAS_train, DOCS_train)
    generate_dataset(1000, 'hotpotqa_dev_process', 300, QAS_dev, DOCS_dev)