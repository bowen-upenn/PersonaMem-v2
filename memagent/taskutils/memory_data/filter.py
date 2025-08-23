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
from openai import AsyncOpenAI, OpenAI
from rich import print
import fire


base_client = AsyncOpenAI(
    base_url=f"http://localhost:8000/v1", # PORT 5
    api_key="NOT A REAL KEY",
    timeout=36000,
)
inst_client = AsyncOpenAI(
    base_url=f"http://localhost:8001/v1", # PORT 5
    api_key="NOT A REAL KEY",
    timeout=36000,
)

TEMPLATE = """{prompt}
Your final answer in \\boxed{{}}.
"""

async def chat(model, messages):
    if "Inst" in model:
        client = inst_client
    else:
        client = base_client
    completion = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": TEMPLATE.format(prompt = messages[0]['content']),
        }],
        temperature=1.0,
        top_p=0.7,
        n=2,
        max_completion_tokens=512,
    )
    return completion

import asyncio
import time
from memory_data.utils import async_main

from hotpotqa_verifier import compute_score
def scores(args):
    responses, reward_model = args
    gts:list = reward_model['ground_truth']
    scores = [
        max(compute_score(r, gt) for gt in gts)
        for r in responses
    ]
    return scores
def contains(args):
    responses, reward_model = args
    gts:list = reward_model['ground_truth']
    scores = [
        max(gt.lower() in r.lower() for gt in gts)
        for r in responses
    ]
    return scores
def main(input, output, resume=True):
    print(locals())
    import pandas as pd
    
    from pathlib import Path
    if not output.endswith(".parquet"):
        output += ".parquet"
    Path(output).parent.mkdir(exist_ok=True, parents=True)
    if not resume or not Path(output).exists():
        df = pd.read_parquet(input)
        tasks = [chat(model, msg) for msg in df['prompt'] for model in ["Qwen2.5-7B", "Qwen2.5-7B-Instruct"]]
        ret = asyncio.run(async_main(tasks, max_workers=1024*8))
        base, inst = ([
            [ret[idx].choices[i].message.content for i in [0, 1]] # each response in the api return value
            for idx in range(model, len(tasks), 2) # each return valuse of api calling to the model
        ] for model in [0,1]) # start=0, basemodel
        df['responses_pretrain'] = base
        df['responses_instruct'] = inst
        df.to_parquet(output)
    else:
        df = pd.read_parquet(output)
    from utils import TqdmExecutor
    exe = TqdmExecutor(chunksize=10)
    for model in ["pretrain", "instruct"]:
        df[f'score_boxed_{model}'] = exe.run(scores, df[[f'responses_{model}','reward_model']].values)
        df[f'score_contains_{model}'] = exe.run(contains, df[[f'responses_{model}','reward_model']].values)
    df.to_parquet(output)
if __name__ == "__main__":
    fire.Fire(main)
