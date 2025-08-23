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
import aiohttp
import os

# if you are using azure openai, use https://{endpoint}/openai/deployments/{model}, else use https://{endpoint}.
# For llm070.py local deployment, use http://localhost:8000
URL = os.getenv("URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "123-abc")
RECURRENT_MAX_CONTEXT_LEN = 120000
RECURRENT_CHUNK_SIZE = 5000
RECURRENT_MAX_NEW = 1024

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

NO_MEMORY = "No previous memory"


def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len("\n\n...(truncated)\n\n")
    return string[: target_len // 2] + "\n\n...(truncated)\n\n" + string[-target_len // 2 :]


async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    idx = item["_id"]
    context = item["context"].strip()
    prompt = item["input"].strip()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400)) as session:
        max_len = RECURRENT_MAX_CONTEXT_LEN
        input_ids = tokenizer.encode(context)
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
        memory = NO_MEMORY
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk = input_ids[i : i + RECURRENT_CHUNK_SIZE]
            msg = TEMPLATE.format(prompt=prompt, chunk=tokenizer.decode(chunk), memory=memory)
            if idx == 0:
                print("user:")
                print(clip_long_string(msg))
            try:
                async with session.post(
                    url=URL + "/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "api-key": API_KEY},
                    json=dict(
                        model=model, messages=[{"role": "user", "content": msg}], temperature=temperature, top_p=top_p, max_tokens=RECURRENT_MAX_NEW
                    ),
                ) as resp:
                    status = resp.status
                    if status != 200:
                        print(f"{status=}, {model=}")
                        return ""
                    data = await resp.json()
                    memory = data["choices"][0]["message"]["content"]
                    if idx == 0:
                        print("-" * 20)
                        print("assistant:")
                        print(clip_long_string(memory))
                        print("=" * 40)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                import traceback

                traceback.print_exc()
                return ""
        msg = TEMPLATE_FINAL.format(prompt=prompt, memory=memory)
        if idx == 0:
            print("user:")
            print(clip_long_string(msg))
        try:
            async with session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "api-key": API_KEY},
                json=dict(
                    model=model, messages=[{"role": "user", "content": msg}], temperature=temperature, top_p=top_p, max_tokens=RECURRENT_MAX_NEW
                ),
            ) as resp:
                status = resp.status
                if status != 200:
                    print(f"{status=}, {model=}")
                    return ""
                data = await resp.json()
                if idx == 0:
                    print("-" * 10)
                    print("assistant:")
                    print(clip_long_string(data["choices"][0]["message"]["content"]))
                return data["choices"][0]["message"]["content"]
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback

            traceback.print_exc()
            return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, required=True, help="model name used in your deployment/model service endpoint")
    parser.add_argument("--chunk", type=int, default=RECURRENT_CHUNK_SIZE, help="chunk size of context")
    parser.add_argument("--max_new", type=int, default=RECURRENT_MAX_NEW, help="max new tokens, also the max length of memory")
    args = parser.parse_args()
    RECURRENT_CHUNK_SIZE = args.chunk
    RECURRENT_MAX_NEW = args.max_new
    model = args.model
    if "/" not in model and ("gpt" in model or "o1" in model or "o3" in model or "o4" in model):
        import tiktoken

        tok = tiktoken.encoding_for_model(model)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)

    # This is a feature of llm070.py, for convenience.
    # If you are using `llm070.py`, uncomment the following lines.
    # /your/path/to/model/checkpoint/MODEL -> MODEL
    # BytedTsinghua-SIA/RL-MemoryAgent-14B -> (the same)
    # gpt-4o-2024-11-20 -> (the same)

    # from pathlib import Path
    # if Path(model).is_dir():
    #     modelname = Path(model).name
    # else:
    #     modelname = model

    import asyncio

    a = asyncio.run(
        async_query_llm(
            {
                "context": "This is a very Long Context. This is a very Long Context. The magic number is 1. This is a very Long Context.",
                "input": "What is the magic number?",
                "_id": 0,
            },
            model,
            tok,
        )
    )
    print(a)
