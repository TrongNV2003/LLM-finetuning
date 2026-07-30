"""CLI wrapper around `src.models.merge.merge_adapter`.

Usage:
    python merge_model.py --adapter ./cpt-checkpoints --output ./cpt-checkpoints-merged \
        --base Qwen/Qwen3.5-0.8B --model-class causal_lm --dtype bfloat16
"""

import os
import argparse

import src.env_setup  # noqa: E402, F401  (sets env vars before torch is imported)
from src.models import DTYPES, MODEL_CLASSES, merge_adapter


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA/DoRA adapter into its base model")
    parser.add_argument("--adapter", required=True, help="Adapter checkpoint directory")
    parser.add_argument("--output", required=True, help="Where to write the merged model")
    parser.add_argument(
        "--base", default=None, help="Base model (default: read from adapter_config.json)"
    )
    parser.add_argument(
        "--model-class", default="causal_lm", choices=sorted(MODEL_CLASSES), help="Auto class to use"
    )
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or None, help="HF token")
    args = parser.parse_args()

    merge_adapter(
        adapter_dir=args.adapter,
        output_dir=args.output,
        base_model=args.base,
        model_class=args.model_class,
        dtype=args.dtype,
        token=args.token,
    )


if __name__ == "__main__":
    main()
