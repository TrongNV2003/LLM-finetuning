"""Which architectures this repo supports, and in what dtype.

Everything model-specific lives in `src/conf/model/*.yaml`, so supporting a new
base model means adding a YAML file, not editing code. The knobs that actually
differ per model:

- `model_class`      : `causal_lm` keeps only the language model of a VLM
                       checkpoint; `image_text_to_text` keeps the vision tower.
                       Both are decoder-only — this repo's objectives
                       (prompt-completion SFT, causal-LM CPT, perplexity) do not
                       cover encoder-decoder models.
- `attn_implementation`: `sdpa` everywhere by default; `flash_attention_2` only
                       if `flash-attn` is installed.
- `lora.target_modules`: explicit per architecture (`null` -> PEFT
                       `"all-linear"`, which also grabs exotic projections).
- `supports_packing` : `false` for hybrid/linear-attention models, where the
                       recurrent state leaks across packed samples.
- `chat_template_kwargs`: e.g. `enable_thinking: false` for Qwen3/Qwen3.5.
"""

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

__all__ = ["MODEL_CLASSES", "resolve_compute_dtype", "resolve_model_class"]

MODEL_CLASSES = {
    "causal_lm": AutoModelForCausalLM,
    "image_text_to_text": AutoModelForImageTextToText,
}


def resolve_model_class(name: str):
    try:
        model_class = MODEL_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"Unknown model.model_class={name!r}, expected one of {sorted(MODEL_CLASSES)}"
        ) from None
    return model_class


def resolve_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        # fp16 math is unimplemented for many CPU kernels; keep CPU runs in fp32.
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
