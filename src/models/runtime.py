"""Device placement, KV cache and VRAM housekeeping around a loaded model."""

import gc
import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

__all__ = ["ensure_on_device", "free_gpu_memory", "inference_cache"]


def ensure_on_device(model) -> torch.device:
    """Return the model's device, moving it to CUDA first when it is movable."""
    if getattr(model, "hf_quantizer", None) is not None or getattr(model, "hf_device_map", None):
        return next(model.parameters()).device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return device


@contextmanager
def inference_cache(model):
    """Temporarily re-enable the KV cache.

    Training configs set `use_cache=false` (required with gradient
    checkpointing); leaving it off during `generate()` disables the KV cache and
    makes generation several times slower.
    """
    text_config = model.config.get_text_config()
    previous = text_config.use_cache
    text_config.use_cache = True
    try:
        yield model
    finally:
        text_config.use_cache = previous


def free_gpu_memory() -> None:
    """Release cached VRAM between the training model and the merge/eval load."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
