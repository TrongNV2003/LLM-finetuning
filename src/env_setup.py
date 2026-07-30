import os
import sys
import logging
from typing import Optional, overload

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

os.environ.setdefault("HYDRA_FULL_ERROR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

_gpus = os.environ.get("GPU_DEVICES")
if _gpus and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
for _name in ("src", "__main__"):
    logging.getLogger(_name).setLevel(_log_level)


@overload
def resolve_path(path: str) -> str: ...


@overload
def resolve_path(path: None) -> None: ...


def resolve_path(path: Optional[str]) -> Optional[str]:
    """Anchor a relative path to the project root instead of the shell cwd.

    Datasets, checkpoints and result files all go through this, so a run started
    from any directory reads and writes the same locations.
    """
    if not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


__all__ = ["PROJECT_ROOT", "resolve_path"]
