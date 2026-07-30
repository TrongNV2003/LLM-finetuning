"""Merge a LoRA/DoRA adapter into its base model.

The base model is loaded in 16-bit on CPU even when training used 4-bit QLoRA:
- merging into dequantized weights is the standard path, but it is not bit-exact
  with what the adapter saw, so the merged model deserves its own eval run;
- staying on CPU means the merge never competes with the training job for VRAM.
"""

import os
import gc
import json
import torch
import logging
from typing import Optional
from omegaconf import DictConfig
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer

from src.env_setup import resolve_path
from src.models.registry import resolve_model_class

logger = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def merge_adapter(
    adapter_dir: str,
    output_dir: str,
    base_model: Optional[str] = None,
    model_class: str = "causal_lm",
    dtype: str = "bfloat16",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
) -> str:
    """Write `base_model + adapter_dir` to `output_dir`. Returns `output_dir`."""
    adapter_dir = resolve_path(adapter_dir)
    output_dir = resolve_path(output_dir)
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"{adapter_config_path} not found — is {adapter_dir} really an adapter checkpoint?"
        )

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model = base_model or peft_config.base_model_name_or_path
    if not base_model:
        raise ValueError("Base model not recorded in the adapter; pass --base explicitly")

    with open(adapter_config_path) as f:
        raw_adapter_config = json.load(f)
    logger.info(
        f"Adapter: r={raw_adapter_config.get('r')}, "
        f"alpha={raw_adapter_config.get('lora_alpha')}, "
        f"use_dora={raw_adapter_config.get('use_dora')}, "
        f"targets={raw_adapter_config.get('target_modules')}"
    )

    loader = resolve_model_class(model_class)

    logger.info(f"Loading base model {base_model} in {dtype} (CPU)")
    model = loader.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        token=token,
        cache_dir=cache_dir,
        revision=revision,
        dtype=DTYPES[dtype],
    )

    logger.info(f"Applying adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir, token=token)

    logger.info("Merging adapter weights into the base model")
    model = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    if os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json")):
        tokenizer_source = adapter_dir
    else:
        logger.info(f"No tokenizer in {adapter_dir}, taking it from the base model")
        tokenizer_source = base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=trust_remote_code, token=token, cache_dir=cache_dir
    )
    tokenizer.save_pretrained(output_dir)

    del model
    gc.collect()

    logger.info(f"Merged model saved to {output_dir}")
    return output_dir


def resolve_merge_output_dir(cfg: DictConfig, adapter_dir: str) -> str:
    configured = cfg.get("merge", {}).get("output_dir")
    if configured:
        return resolve_path(str(configured))
    return f"{adapter_dir.rstrip('/')}-merged"


def merge_from_config(cfg: DictConfig, adapter_dir: str) -> Optional[str]:
    """Run the post-training merge if `merge.enabled`, else return None.

    Skipped with a note when the run had no adapter to merge (full fine-tuning),
    because `output_dir` is then already a complete model.
    """
    merge_cfg = cfg.get("merge")
    if not merge_cfg or not merge_cfg.get("enabled", False):
        return None

    if cfg.model.get("lora") is None:
        logger.info(
            "merge.enabled=true but LoRA is disabled — "
            f"{adapter_dir} is already a complete model, nothing to merge"
        )
        return None

    output_dir = resolve_merge_output_dir(cfg, adapter_dir)
    logger.info(f"Merging {adapter_dir} -> {output_dir}")
    return merge_adapter(
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        base_model=cfg.model.model_name_or_path,
        model_class=cfg.model.get("model_class", "causal_lm"),
        dtype=merge_cfg.get("dtype", "bfloat16"),
        token=cfg.token,
        cache_dir=cfg.model.cache_dir,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
    )
