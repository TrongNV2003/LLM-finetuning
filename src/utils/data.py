"""Reading the JSON/JSONL datasets the configs point at. No torch here either.

The place to add a new on-disk format: every task reads its data through
`load_dataset`, so a new extension only has to be handled once.
"""

import json
from typing import Dict, List

from src.env_setup import resolve_path

__all__ = ["load_dataset"]


def load_dataset(dataset_file_or_path: str) -> List[Dict]:
    if not dataset_file_or_path:
        raise ValueError("No dataset path given — the corresponding *_file is null in the config")
    dataset_file_or_path = resolve_path(dataset_file_or_path)
    if dataset_file_or_path.endswith(".json"):
        with open(dataset_file_or_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif dataset_file_or_path.endswith(".jsonl"):
        with open(dataset_file_or_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file format: {dataset_file_or_path}")
    return data
