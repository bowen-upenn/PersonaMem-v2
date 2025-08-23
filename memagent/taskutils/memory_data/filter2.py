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
import pandas as pd
def process(filename, num):
    df = pd.read_parquet(f'{filename}_result.parquet')
    def sprint(sample):
        print("======RSP PRE0")
        print(sample.responses_pretrain[0])
        print("======RSP PRE1")
        print(sample.responses_pretrain[1])
        print("======RSP INS0")
        print(sample.responses_instruct[0])
        print("======RSP INS1")
        print(sample.responses_instruct[1])
        print("======scores")
        print(sample.reward_model['ground_truth'])
        print("pre",sample.score_contains_pretrain)
        print("inst",sample.score_contains_instruct)
        print("preboxed",sample.score_boxed_pretrain)
        print("instboxed",sample.score_boxed_instruct)
        print("======END")
    sprint(df.iloc[11])
    pre_contains_corr = df['score_contains_pretrain'].map(lambda x: max(x) > 0)
    pre_boxed_corr = df['score_boxed_pretrain'].map(lambda x: max(x) > 0)
    print(sum(pre_contains_corr))
    print(sum(pre_boxed_corr))
    contains_corr = df['score_boxed_instruct'].map(lambda x: max(x) > 0)
    boxed_corr = df['score_contains_instruct'].map(lambda x: max(x) > 0)
    print(sum(contains_corr))
    print(sum(boxed_corr))
    any_corr = pre_contains_corr | pre_boxed_corr | contains_corr | boxed_corr
    print("available:", len(df)-sum(any_corr))
    df[~any_corr].head(num).to_parquet(f"{filename}.parquet")
if __name__ == "__main__":
    process("hotpotqa_train", 8192)
    process("hotpotqa_dev", 128)