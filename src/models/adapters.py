"""Quantization and PEFT"""

import torch
import logging
from typing import List, cast
from torch.nn import Module
from omegaconf import DictConfig
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

from src.utils.config import as_container

logger = logging.getLogger(__name__)

__all__ = [
    "apply_peft",
    "build_quantization_config",
    "find_all_linear_names",
    "resolve_target_modules",
]


def build_quantization_config(model_args: DictConfig, compute_dtype: torch.dtype):
    if not model_args.get("qlora", False):
        logger.info("Quantization: none (16-bit base weights)")
        return None
    logger.info("Quantization: QLoRA 4-bit (nf4, double quant)")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def resolve_target_modules(lora_args: DictConfig):
    """Explicit list from the model YAML, or PEFT's own `all-linear` sweep.

    `all-linear` replaces the old hand-rolled `find_all_linear_names`: PEFT
    already skips the output layer and handles quantized layers.
    """
    target_modules = as_container(lora_args.get("target_modules"))
    if not target_modules:
        logger.info("LoRA target_modules: all-linear (resolved by PEFT)")
        return "all-linear"
    logger.info(f"LoRA target_modules: {target_modules}")
    return target_modules


def apply_peft(model, model_args: DictConfig, use_gradient_checkpointing: bool = True):
    """Wrap with LoRA/QLoRA, or return the model untouched for full fine-tuning."""
    lora_args = model_args.get("lora")
    if lora_args is None:
        logger.info("PEFT: disabled -> full fine-tuning")
        return model

    if model_args.get("qlora", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing
        )

    if lora_args.get("use_dora", False) and model_args.get("qlora", False):
        logger.warning(
            "use_dora=true with 4-bit QLoRA is slow and harder to merge cleanly; "
            "prefer a larger r with plain LoRA unless you specifically need DoRA."
        )

    lora_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        target_modules=resolve_target_modules(lora_args),
        modules_to_save=as_container(lora_args.get("modules_to_save")) or None,
        use_dora=lora_args.get("use_dora", False),
        use_rslora=lora_args.get("use_rslora", False),
        task_type=TaskType.CAUSAL_LM,
    )
    # `get_peft_model` is annotated `PeftModel | PeftMixedModel`; only `mixed=True`
    # yields the latter, and TRL's trainers only accept a plain `PeftModel`.
    model = cast(PeftModel, get_peft_model(model, lora_config))

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters:      {trainable / 1e6:.2f}M")
    logger.info(f"Total parameters:          {total / 1e6:.2f}M")
    logger.info(f"% of trainable parameters: {100 * trainable / total:.4f}%")
    return model


def find_all_linear_names(model: Module) -> List[str]:
    """List every Linear leaf name.

    Passing this straight to LoRA also targets vision towers, MoE routers and other layers rarely want. 
    Set `lora.target_modules` explicitly in the model YAML, or leave it null to get PEFT's own `"all-linear"` handling (see `resolve_target_modules`).
    """
    try:
        import bitsandbytes as bnb

        quantized_linear = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        quantized_linear = ()

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, *quantized_linear)):
            parts = name.split(".")
            module_names.add(parts[0] if len(parts) == 1 else parts[-1])

    module_names.discard("lm_head")
    return sorted(module_names)
