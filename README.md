# LLM Fine-tuning Project

This project provides a comprehensive pipeline for fine-tuning Large Language Models (LLMs) using TRL (Transformers Reinforcement Learning) and LoRA (Low-Rank Adaptation). It supports both **Supervised Fine-Tuning (SFT)** and **Continued Pre-Training (CPT)**.

## Overview
The repository is split into two main fine-tuning tasks:
1. **Continued Pre-Training (CPT)**: Used for training models on long, unstructured, unlabeled text documents (e.g., domain-specific reports) to adapt the model to a new domain.
2. **Supervised Fine-Tuning (SFT)**: Used for training models on structured, labeled instruction-response pairs (e.g., Vietnamese address normalization).

## Features
- **TRL Integration**: Uses TRL's `SFTTrainer` for both CPT and SFT tasks.
- **LoRA & QLoRA Fine-tuning**: Memory-efficient fine-tuning using Low-Rank Adaptation.
- **4-bit Quantization**: Memory optimization using `BitsAndBytesConfig`.
- **Hydra Configuration**: Manages complex training arguments easily via YAML files.

## Installation

1. **Clone or create the project directory**:
   ```bash
   git clone https://github.com/TrongNV2003/LLM-finetuning.git
   ```

2. **Create env & Install dependancies**:
   ```bash
   conda create -n tuning python==3.12
   conda activate tuning
   pip install -r requirements.txt
   ```

## Usage

### 1. Continued Pre-Training (CPT)
Used for domain adaptation on long, unlabeled text data.

#### Preparing Data:
Convert plain markdown files into chunked JSON format.
```bash
python src/finetune/cpt/prepare_cpt_data.py --source_dir /path/to/md_files --output_dir /path/to/save_json --model_name model_name_or_path

e.g. python src/finetune/cpt/prepare_cpt_data.py --source_dir data/report_dataset --output_dir data/prepared_dataset --model_name Qwen/Qwen3.5-0.8B
```

**Dataset Format (JSON)**:
```json
[
  {
    "text": "Extracted chunk of text fitting within the context length."
  }
]
```

#### Start CPT Training:
```bash
python src/finetune/cpt/cpt_train.py
```

#### Merge LoRA Weights to Base Model:
```bash
python merge_model.py
```

### 2. Supervised Fine-Tuning (SFT)
Used for task-specific training (e.g., Address Normalization).

#### Preparing Dataset Format
Your dataset should follow this structure:
```json
[
  {
    "text": "incomplete_address",
    "label": "complete_address"
  }
]
```

#### Start SFT Training:
```bash
python src/finetune/sft/train.py
```

## Configuration
### Hyperparameter Tuning

Configurations are located in `src/conf/`. Key hyperparameters to experiment with:

- **LoRA rank (r)**: Higher values = more parameters, better performance, more memory.
- **Learning rate**: Start with 2e-4, adjust based on convergence.
- **Batch size**: Adjust based on GPU memory.
- **Number of epochs**: Monitor training loss to avoid overfitting.

## Memory Optimization

The project includes several memory optimization techniques:

1. **4-bit Quantization**: Reduces model memory usage by ~75%.
2. **LoRA**: Only trains a small fraction of parameters.
3. **Gradient Accumulation**: Simulates larger batch sizes with limited memory.
