import json
import torch
from enum import Enum
from typing import Dict, List

import bitsandbytes as bnb
from torch.nn import Module


class DataSource(str, Enum):
    HUGGINGFACE = "huggingface"
    DISK = "disk"

def load_dataset(dataset_file_or_path: str) -> List[Dict]:
    if dataset_file_or_path.endswith('.json'):
        with open(dataset_file_or_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif dataset_file_or_path.endswith('.jsonl'):
        with open(dataset_file_or_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file format: {dataset_file_or_path}")
    return data

def print_trainable_parameters(model: Module) -> Dict[str, float]:
    all_parameters = 0
    trainable_parameters = 0

    for name, param in model.named_parameters():
        all_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()

    trainable_info = {
        "Total Parameters": all_parameters,
        "Trainable Parameters": trainable_parameters,
        "Trainable (%)": round(100 * trainable_parameters / all_parameters, 2),
    }

    return trainable_info

def find_all_linear_names(model: Module) -> List[str]:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
