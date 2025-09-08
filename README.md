# LLM Fine-tuning for Address Normalization with TRL and LoRA
This project demonstrates how to fine-tune a Large Language Model (LLM) using TRL (Transformers Reinforcement Learning) and LoRA (Low-Rank Adaptation) for the task of text normalization, specifically for Vietnamese address completion.

## Overview
The goal is to train a model that can complete partial/noisy Vietnamese addresses. For example:
- **Input**: "68, Xthuy, Cgiay, HN"
- **Output**: "68, Xuân Thủy, Cầu Giấy, Hà Nội"

## Features
- **TRL Integration**: Uses TRL's SFTTrainer for supervised fine-tuning
- **LoRA Fine-tuning**: Memory-efficient fine-tuning using Low-Rank Adaptation
- **4-bit Quantization**: Memory optimization using BitsAndBytesConfig

## Project Structure
```
LLM-finetuning/
├── data/                           # Dataset example format
└── src/                            # Directory contain code training model
    ├── callbacks/                  # Directory for callback functions
    ├── conf/                       # Directory contain arguments for training model
        ├── dataset/                # Directory contain arguments for dataset
        ├── model/                  # Directory contain arguments for model
        └── finetuning.yaml         # Main config for training model
    ├── metrics.py                  # Metric for evaluate model
    ├── prompts.py                  # Prompt template LLM
    ├── train.py                    # Main training script
    └── utils.py                    # Scripts contain utilize code
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

1. **Clone or create the project directory**:
   ```bash
   cd /home/trongnv130/Desktop/self-project/LLM-finetuning
   ```

2. **Create env & Install dependancies**:
   ```bash
   conda create -n tuning python==3.11.11
   pip install -r requirements.txt
   ```

## Usage
### 1. Dataset Format
Your dataset should follow this structure:

```json
[
  {
    "text": "incomplete_address",
    "label": "complete_address"
  }
]
```

### 2. Training
**Start training**:
```bash
python train.py
```

The training script will:
- Load the dataset
- Initialize the model with LoRA adapters
- Fine-tune the model using TRL's SFTTrainer

### 3. Inference
After training, test the model:
```bash
python inference.py
```

## Configuration
### Hyperparameter Tuning

Key hyperparameters to experiment with:

- **LoRA rank (r)**: Higher values = more parameters, better performance, more memory
- **Learning rate**: Start with 2e-4, adjust based on convergence
- **Batch size**: Adjust based on GPU memory
- **Number of epochs**: Monitor training loss to avoid overfitting

## Memory Optimization

The project includes several memory optimization techniques:

1. **4-bit Quantization**: Reduces model memory usage by ~75%
2. **LoRA**: Only trains a small fraction of parameters
3. **Gradient Accumulation**: Simulates larger batch sizes with limited memory



## License
This project is for educational and research purposes. Please check the licenses of the underlying models and libraries used.
