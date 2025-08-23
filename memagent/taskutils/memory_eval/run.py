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
import os
import time
from dataclasses import dataclass
import sys

sys.stdout.reconfigure(line_buffering=True)
DASH_PORT = os.getenv("DASH_PORT", "8265")
SERVE_PORT = os.getenv("SERVE_PORT", "8000")
MODELROOT = os.getenv("MODELROOT", "/mnt/hdfs/hongli/model")


@dataclass
class ENV:
    # config for direct generation
    MAX_INPUT_LEN: int = 120000
    MAX_OUTPUT_LEN: int = 10000
    # Config for memory agent
    RECURRENT_MAX_CONTEXT_LEN: int = None
    RECURRENT_CHUNK_SIZE: int = None
    RECURRENT_MAX_NEW: int = None

    def setenv(self):
        if not hasattr(self, "_environ"):
            self._environ = {}
        for k, v in self.__dict__.items():
            if v is not None and k != "_environ":
                os.environ[k] = str(v)
                self._environ[k] = str(v)
                print(f"set {k}={v}")

    def unsetenv(self):
        for k in self._environ:
            os.environ[k] = self._environ[k]
        self._environ = {}


# for ruler hqa, we just control the number of distractive wiki items instead the context length
# 50~7K tokens, 100~14K tokens and so on.
RULER_HQA_TESTS = [50, 100, 200, 400, 800, 1600, 3200, 6400]
RULER_HQA_TESTS_OVER_1M = [12800, 25600]
# for other ruler task, we use the standard synthetic scripts for convenient and control the context length.
RULER_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
    "qa_1",
]
RULER_PROMPT_LENGTH = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
RULER_GENERRAL_TESTS = [(task, length) for task in RULER_TASKS for length in RULER_PROMPT_LENGTH]
import subprocess


class Config:
    SERVE_TAG = "__serve"

    def __init__(self, name, ckpt, tp, method, env, concur=1024):
        self.name = name
        self.ckpt = ckpt
        from pathlib import Path

        if Path(self.ckpt).is_dir():
            self.model = Path(self.ckpt).name
        else:
            self.model = self.ckpt
        self.method = method
        self.tp = tp
        self.env = env
        self.concur = concur
        self.test_process = {}

    def serve(self, wait=True):
        serve_script = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "serve/llm070.py"))
        cmd = f"python {serve_script} --model {self.ckpt} --tp {self.tp}"
        print("serving command:")
        print(cmd)
        if wait:
            os.system(f"yes | serve shutdown -a http://localhost:{DASH_PORT}")
            # setsid so that it can be interrupted
            serve_p = subprocess.Popen(cmd.split(), preexec_fn=os.setsid)
            self.test_process[self.SERVE_TAG] = serve_p
            while True:
                print("try to conntect...")
                p = subprocess.run(["curl", "-m", "100000000", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
                if p.returncode != 0:
                    print("waiting...")
                    time.sleep(5)
                elif rf'"id":"{self.model}"' not in p.stdout.decode():
                    print("model not found, maybe shutting down previous server...")
                    time.sleep(5)
                else:
                    print("connected")
                    break
        else:
            p = subprocess.run(["curl", "-m", "10", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
            if p.returncode != 0:
                print("server not started")
                exit(1)
        print(p.stdout)

    def run(self, tests, serve=True, force=False):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.env.setenv()
        self.serve(serve)
        concur = self.concur
        for test in tests:
            if test in RULER_HQA_TESTS:
                cmd = f"""python ruler_hqa.py --model {self.model}\
                    --length {test} \
                    --save_dir results/ruler_hqa_{test} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
            elif test in RULER_GENERRAL_TESTS:
                cmd = f"""python ruler_general.py --model {self.model}\
                    --split {test[0]} \
                    --length {test[1]} \
                    --save_dir results/ruler_{test[0]}_{test[1]} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
            elif test in RULER_HQA_TESTS_OVER_1M:
                cmd = f"""python ruler_hqa_over1m.py --model {self.model}\
                    --length {test} \
                    --save_dir results/ruler_hqa_{test} \
                    --save_file {self.name} \
                    --tokenizer {self.ckpt} \
                    --api {self.method} \
                    --n_proc {concur}"""
            else:
                print("=" * 20 + f"Not Implemented Task {test}, please check" + "=" * 20)
                continue
            if force:
                cmd += " --force"
            p = subprocess.Popen(cmd, shell=True)
            self.test_process[test] = p
            p.wait()
            self.test_process[test].wait()
        self.env.unsetenv()
        if serve:
            os.killpg(os.getpgid(self.test_process[self.SERVE_TAG].pid), 2)
            try:
                self.test_process[self.SERVE_TAG].wait(30)
            except:
                self.test_process[self.SERVE_TAG].kill()
        print("all tests finished")

    def __del__(self):
        for k, p in self.test_process.items():
            if k == self.SERVE_TAG:
                os.killpg(os.getpgid(p.pid), 2)
            else:
                p.kill()


L1 = Config(
    name="L1-120k+10k",
    ckpt="Tongyi-Zhiwen/QwenLong-L1-32B",
    tp=4,
    method="openai",
    concur=128,
    env=ENV(),
)

R1_32B = Config(
    name="R1-32B-120k+10k",
    ckpt="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

R1_14B = Config(
    name="R1-14B-120k+10k-openai",
    ckpt="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    tp=2,
    method="openai",
    concur=256,
    env=ENV(),
)

R1_7B = Config(
    name="R1-7B-120k+10k",
    ckpt="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    tp=1,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_7B_1M = Config(
    name="Qwen-7B-990k+10k",
    ckpt="Qwen/Qwen2.5-7B-Instruct-1M",
    tp=2,
    method="openai",
    concur=256,
    env=ENV(MAX_INPUT_LEN=990000, MAX_OUTPUT_LEN=10000),
)

Qwen25_14B_1M = Config(
    name="Qwen-14B-990k+10k",
    ckpt="Qwen/Qwen2.5-14B-Instruct-1M",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(MAX_INPUT_LEN=990000, MAX_OUTPUT_LEN=10000),
)

Qwen25_32B_128k = Config(
    name="Qwen-32B-120k+10k",
    ckpt=f"{MODELROOT}/Qwen2.5-32B-Instruct-128k",
    tp=4,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_14B_128k = Config(
    name="Qwen-14B-120k+10k",
    ckpt=f"{MODELROOT}/Qwen2.5-14B-Instruct-128k",
    tp=2,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_7B_128k = Config(
    name="Qwen-7B-120k+10k",
    ckpt=f"{MODELROOT}/Qwen2.5-7B-Instruct-128k",
    tp=1,
    method="openai",
    concur=256,
    env=ENV(),
)

Qwen25_32B_5k_1k = Config(
    name="Qwen-32B-5k-1k-infty",
    ckpt="Qwen/Qwen2.5-32B-Instruct",
    tp=4,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

Qwen25_14B_5k_1k = Config(
    name="Qwen-14B-5k-1k-infty",
    ckpt="Qwen/Qwen2.5-14B-Instruct",
    tp=2,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

Qwen25_7B_5k_1k = Config(
    name="Qwen-7B-5k-1k-infty",
    ckpt="Qwen/Qwen2.5-7B-Instruct",
    tp=1,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

MemoryAgent_7B_5k_1k = Config(
    name="MemoryAgent-7B-5k-1k-infty",
    ckpt="BytedTsinghua-SIA/RL-MemoryAgent-7B",
    tp=1,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)


MemoryAgent_14B_5k_1k = Config(
    name="MemoryAgent-14B-5k-1k-infty",
    ckpt="BytedTsinghua-SIA/RL-MemoryAgent-14B",
    tp=2,
    method="recurrent",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=1024),
)

CONFIGS = [
    # OURS
    MemoryAgent_7B_5k_1k,
    MemoryAgent_14B_5k_1k,
    # Main Result Baselines
    L1,
    Qwen25_14B_1M,
    Qwen25_7B_1M,
    R1_32B,
    R1_14B,
    R1_7B,
    # Ablation Baselines
    Qwen25_32B_128k,
    Qwen25_14B_128k,
    Qwen25_7B_128k,
    Qwen25_32B_5k_1k,
    Qwen25_14B_5k_1k,
    Qwen25_7B_5k_1k,
]

def run_ruler_hqa():
    for c in CONFIGS:
        task = RULER_HQA_TESTS
        if c.name.startswith("MemoryAgent"):
            task += RULER_HQA_TESTS_OVER_1M
        c.run(task, serve=True, force=False)


def run_ood_tasks():
    for c in CONFIGS:
        subset = [
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multivalue",
            "niah_multiquery",
            "vt",
            "fwe",
            "qa_1",
        ]
        lengths = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
        task = [(s, l) for s in subset for l in lengths if not (s == "qa_1" and l > 262144)]
        c.run(task, serve=True, force=False)


if __name__ == "__main__":
    print(f"{SERVE_PORT=}, {DASH_PORT=}, {MODELROOT=}")
    run_ruler_hqa()
    run_ood_tasks()
