"""Tokenizer/model loading shared by SFT, CPT and evaluation."""

import os
import logging
from typing import Optional, cast
from omegaconf import DictConfig
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.models.adapters import apply_peft, build_quantization_config
from src.models.registry import resolve_compute_dtype, resolve_model_class

logger = logging.getLogger(__name__)

__all__ = [
    "load_base_model",
    "load_model_for_inference",
    "load_model_for_training",
    "load_tokenizer",
]


def load_tokenizer(cfg: DictConfig) -> PreTrainedTokenizerBase:
    model_args = cfg.model
    # `AutoTokenizer.from_pretrained` is annotated as the backend union
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
            token=cfg.token,
            revision=model_args.revision,
            padding_side=model_args.padding_side,
        ),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    cfg: DictConfig,
    tokenizer,
    model_name_or_path: Optional[str] = None,
    quantize: bool = True,
):
    """Load the base model straight from `from_pretrained`.

    No hand-built `AutoConfig` is passed: for composite (VLM) checkpoints the
    top-level config is not the text config, so `use_cache`/`pad_token_id` set
    there would silently not reach the language model. They are set afterwards
    through `get_text_config()` / `generation_config` instead.
    """
    model_args = cfg.model
    compute_dtype = resolve_compute_dtype()
    quantization_config = build_quantization_config(model_args, compute_dtype) if quantize else None
    model_class = resolve_model_class(model_args.get("model_class", "causal_lm"))

    model = model_class.from_pretrained(
        model_name_or_path or model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        token=cfg.token,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
        dtype=compute_dtype,
    )

    text_config = model.config.get_text_config()
    text_config.use_cache = bool(model_args.use_cache)
    if model.generation_config is not None:
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

    logger.info(
        f"Loaded {model_name_or_path or model_args.model_name_or_path} "
        f"as {type(model).__name__} (dtype={compute_dtype}, "
        f"attn={model.config._attn_implementation})"
    )
    return model


def load_model_for_training(cfg: DictConfig, tokenizer):
    use_gradient_checkpointing = bool(
        cfg.training_arguments.get("gradient_checkpointing", True)
    )
    return apply_peft(
        load_base_model(cfg, tokenizer),
        cfg.model,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def load_model_for_inference(cfg: DictConfig, model_path: str, tokenizer):
    """Load an adapter checkpoint on top of its base, or a merged/full model."""

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        peft_config = PeftConfig.from_pretrained(model_path)
        base_name = peft_config.base_model_name_or_path or cfg.model.model_name_or_path
        logger.info(f"Loading PEFT adapter {model_path} on base {base_name}")
        base_model = load_base_model(cfg, tokenizer, model_name_or_path=base_name)
        return PeftModel.from_pretrained(base_model, model_path)

    logger.info(f"Loading full (no-adapter) model from {model_path}")
    return load_base_model(cfg, tokenizer, model_name_or_path=model_path, quantize=False)
