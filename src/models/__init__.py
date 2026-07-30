"""Loading, adapting, merging and running the base models.

Re-exports the public API so call sites stay short (`from src.models import
load_tokenizer`). Unlike `src.utils`, everything in here needs torch anyway, so
there is nothing to gain from importing submodules directly.

`src.env_setup` is imported first, before any submodule pulls torch in: importing
a submodule runs this file first, so `CUDA_VISIBLE_DEVICES` and the tokenizer /
cuBLAS env vars are always set before torch initialises — even from a test or a
notebook that never goes through one of the entry points.
"""

import src.env_setup  # noqa: F401  (side effects: env vars, logging, .env)

from src.models.adapters import (  # noqa: E402
    apply_peft,
    build_quantization_config,
    find_all_linear_names,
    resolve_target_modules,
)
from src.models.loading import (  # noqa: E402
    load_base_model,
    load_model_for_inference,
    load_model_for_training,
    load_tokenizer,
)
from src.models.merge import (  # noqa: E402
    DTYPES,
    merge_adapter,
    merge_from_config,
    resolve_merge_output_dir,
)
from src.models.registry import (  # noqa: E402
    MODEL_CLASSES,
    resolve_compute_dtype,
    resolve_model_class,
)
from src.models.runtime import ensure_on_device, free_gpu_memory, inference_cache  # noqa: E402

__all__ = [
    "DTYPES",
    "MODEL_CLASSES",
    "apply_peft",
    "build_quantization_config",
    "ensure_on_device",
    "find_all_linear_names",
    "free_gpu_memory",
    "inference_cache",
    "load_base_model",
    "load_model_for_inference",
    "load_model_for_training",
    "load_tokenizer",
    "merge_adapter",
    "merge_from_config",
    "resolve_compute_dtype",
    "resolve_merge_output_dir",
    "resolve_model_class",
    "resolve_target_modules",
]
