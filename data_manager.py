import json
import csv
import os


def load_json(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows: list, filename: str):
    if not rows:
        return
    with open(filename, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
