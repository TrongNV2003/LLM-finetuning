"""Building the TRL/Transformers training config from `cfg.training_arguments`."""

import logging

from src.env_setup import resolve_path

import torch
from omegaconf import DictConfig

from src.utils.config import as_container

logger = logging.getLogger(__name__)

__all__ = ["build_training_args", "resolve_packing"]


def build_training_args(cfg: DictConfig, config_cls):
    """Build a TRL/Transformers config from `cfg.training_arguments`.

    Mixed precision and TF32 are resolved *before* the config is constructed, not
    patched onto it afterwards, for two reasons:

    - `TrainingArguments.__post_init__` raises outright for `bf16=true` /
      `tf32=true` on a pre-Ampere GPU, so a YAML that hardcodes them never reaches
      any later fixup.
    - `__post_init__` also freezes the resolved mode: it derives
      `args.mixed_precision` from `bf16`/`fp16` there and hands *that* to the
      Accelerator, so a later `training_args.bf16 = True` never reaches the
      Accelerator and training silently runs in fp32 while the log claims bf16.
    """
    overrides = as_container(cfg.training_arguments)

    # bf16 and TF32 both need Ampere (sm_80) or newer.
    ampere_or_newer = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    overrides["bf16"] = ampere_or_newer
    overrides["fp16"] = torch.cuda.is_available() and not ampere_or_newer
    overrides["tf32"] = bool(overrides.get("tf32", False)) and ampere_or_newer

    if overrides.get("output_dir"):
        overrides["output_dir"] = resolve_path(overrides["output_dir"])
    overrides.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    overrides.setdefault("dataloader_pin_memory", False)
    overrides.setdefault("dataloader_num_workers", 0)

    training_args = config_cls(**overrides)

    resolve_packing(training_args, cfg.model)

    logger.info(
        f"Precision: BF16={training_args.bf16}, FP16={training_args.fp16}, "
        f"TF32={training_args.tf32}, packing={getattr(training_args, 'packing', False)}"
    )
    return training_args


def resolve_packing(training_args, model_args: DictConfig) -> None:
    """Disable packing on architectures where it is not sound.

    TRL's bfd packing flattens a batch into one sequence and relies on
    `cu_seq_lens` to keep samples apart. Hybrid models (e.g. Qwen3.5's Gated
    DeltaNet layers) only honour that inside the `flash-linear-attention`
    kernels; the pure-torch fallback ignores it, so the recurrent state bleeds
    from one packed sample into the next.
    """
    if getattr(training_args, "packing", False) and not model_args.get("supports_packing", True):
        logger.warning(
            "packing=true but this model declares supports_packing=false "
            "(hybrid/linear-attention layers leak state across packed samples) -> forcing packing off"
        )
        training_args.packing = False
        training_args.eval_packing = False
