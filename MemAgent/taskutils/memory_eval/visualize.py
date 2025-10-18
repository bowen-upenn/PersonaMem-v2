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
import glob
import json
import pandas as pd
from collections import defaultdict


METRICS_KEY = ['judge_sub_em'] 
def parse_jsonl_file(file_path):
    judge_values_raw = defaultdict(list)
    try:
        l = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                l += 1
                try:
                    entry = json.loads(line.strip())
                    current_sample_metrics = []
                    for key, value in entry.items():
                        if isinstance(value, (int, float)) and key in METRICS_KEY:
                            # 收集原始的judge指标值
                            judge_values_raw[key].append(value)
                except json.JSONDecodeError:
                    # 跳过不是有效JSON的行
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    if l != 128:
        print(file_path, l)
    calculated_metrics = {}
    for judge_key, values in judge_values_raw.items():
        if values:
            avg_value = round(sum(values) / len(values) * 100, 2)
            if judge_key == "judge":
                display_key = "judge"
            else:
                display_key = judge_key.replace("judge_", "")
            
            calculated_metrics[display_key] = avg_value
    return calculated_metrics


def collect_and_transform_data(base_dir, relpath):
    assert len(relpath) == 2, "relative path should looks like [f'{dataset_name}', '*.json']"
    data_for_df = []

    for file_path in glob.glob(os.path.join(base_dir, *relpath), recursive=True):
        relative_path = os.path.relpath(file_path, base_dir)
        parts = relative_path.split(os.sep)
        dataset_name = parts[0]
        method_name = os.path.basename(file_path).replace(".jsonl", "")

        metrics = parse_jsonl_file(file_path)
        
        for metric_name, value in metrics.items():
            data_for_df.append({
                "Dataset": dataset_name,
                "Metric": metric_name,
                "Method": method_name,
                "Value": value
            })
    df = pd.DataFrame(data_for_df)

    # Pivot the DataFrame to have 'Dataset', 'Metric' as index and 'Method' as columns
    pivot_df = df.pivot_table(index=['Dataset', 'Metric'], columns='Method', values='Value')

    # Prepare for custom sorting (max first, then other metrics alphabetically)
    all_rows = []
    max_only_rows_data = [] # To store data for the 'max only' DataFrame
    dataset_names = pivot_df.index.get_level_values('Dataset').unique()
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    dataset_names = sorted(dataset_names, key=natural_sort_key)
    for dataset in dataset_names:
        dataset_metrics_df = pivot_df.loc[dataset]
        other_metrics = sorted([m for m in dataset_metrics_df.index.unique()])
        for metric in other_metrics:
            row_dict = {'Dataset': dataset, 'Metric': metric}
            row_dict.update(dataset_metrics_df.loc[metric].to_dict())
            all_rows.append(row_dict)
    if all_rows:
        final_df = pd.DataFrame(all_rows).set_index(['Dataset', 'Metric'])
        final_df = final_df.reindex(columns=pivot_df.columns, fill_value=pd.NA)
    else:
        final_df = pd.DataFrame(columns=pivot_df.columns).set_index(['Dataset', 'Metric'])
    return final_df

# --- Main Logic ---
if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150) 
    pd.set_option('display.colheader_justify', 'left')

    basedir = "results"
    relpath = ['ruler_hqa*', '*.jsonl']

    full_results_df = collect_and_transform_data(basedir, relpath)

    print("--- Result ---")
    print(full_results_df)
    print("\n" + "="*80 + "\n") # 分隔符