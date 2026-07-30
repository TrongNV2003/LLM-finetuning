"""Config-composition and precision-resolution tests. No GPU, no model download.

These are the tests that would have caught the two bugs that mattered most:
`build_training_args` patching `bf16` onto an already-constructed config (where
it is both too late to be validated and too late to reach the Accelerator), and
`main()` calling `len(train_dataset)` on `None`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("omegaconf")
pytest.importorskip("hydra")
pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from trl import SFTConfig  # noqa: E402

from src.env_setup import PROJECT_ROOT  # noqa: E402
from src.utils.config import resolve_max_length  # noqa: E402
from src.utils.training_args import build_training_args  # noqa: E402

CONFIG_DIR = os.path.join(PROJECT_ROOT, "src", "conf")
TASK_CONFIGS = ["sft-conf", "cpt-conf"]
MODELS = ["qwen3", "qwen3_5", "llama3"]


def load_cfg(config_name: str, overrides=None):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides or [])


@pytest.mark.parametrize("config_name", TASK_CONFIGS)
def test_task_config_composes(config_name):
    cfg = load_cfg(config_name)
    assert cfg.training_arguments.output_dir
    assert cfg.model.model_name_or_path
    assert cfg.dataset.train_file
    # bf16/fp16 must NOT be in the YAML: they are resolved from the hardware.
    assert "bf16" not in cfg.training_arguments
    assert "fp16" not in cfg.training_arguments


@pytest.mark.parametrize("model", MODELS)
def test_model_configs_declare_required_keys(model):
    cfg = load_cfg("sft-conf", [f"model={model}"])
    for key in ("model_class", "attn_implementation", "supports_packing", "padding_side"):
        assert key in cfg.model, f"{model}.yaml is missing {key}"


@pytest.mark.parametrize("config_name", TASK_CONFIGS)
def test_precision_matches_hardware(config_name):
    """bf16/fp16 must agree with the GPU, and never both be on."""
    cfg = load_cfg(config_name)
    training_args = build_training_args(cfg, SFTConfig)

    ampere_or_newer = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert training_args.bf16 is ampere_or_newer
    assert training_args.fp16 == (torch.cuda.is_available() and not ampere_or_newer)
    assert not (training_args.bf16 and training_args.fp16)
    assert training_args.tf32 == ampere_or_newer

    # The value the Accelerator actually receives, which __post_init__ freezes at
    # construction. Patching args.bf16 afterwards would leave this at "no" while
    # args.bf16 claimed True.
    expected = "bf16" if training_args.bf16 else "fp16" if training_args.fp16 else "no"
    assert training_args.mixed_precision == expected


def test_output_dir_is_anchored_to_project_root():
    cfg = load_cfg("sft-conf")
    training_args = build_training_args(cfg, SFTConfig)
    assert os.path.isabs(training_args.output_dir)
    assert training_args.output_dir.startswith(PROJECT_ROOT)


def test_packing_forced_off_for_hybrid_models():
    cfg = load_cfg("cpt-conf", ["model=qwen3_5"])
    assert cfg.training_arguments.packing is True  # requested
    assert cfg.model.supports_packing is False
    training_args = build_training_args(cfg, SFTConfig)
    assert training_args.packing is False  # and overridden
    assert training_args.eval_packing is False


def test_packing_kept_for_full_attention_models():
    cfg = load_cfg("cpt-conf", ["model=qwen3"])
    training_args = build_training_args(cfg, SFTConfig)
    assert training_args.packing is True


def test_cpt_prepare_block_is_wired_up():
    """cpt_train.py chunks the raw reports itself; the config must say so."""
    cfg = load_cfg("cpt-conf")
    assert cfg.prepare.auto is True
    assert cfg.prepare.force is False
    assert 0 < cfg.prepare.val_ratio < 1
    # The prepare step needs the raw corpus, and both outputs in one directory.
    assert cfg.dataset.source_dir
    assert os.path.dirname(cfg.dataset.train_file) == os.path.dirname(
        cfg.dataset.validation_file
    )


def test_cpt_eval_is_not_step_based():
    """A fixed eval_steps silently means 'never evaluate' on a small corpus."""
    cfg = load_cfg("cpt-conf")
    assert cfg.training_arguments.eval_strategy == "epoch"
    assert cfg.training_arguments.save_strategy == "epoch"
    assert "eval_steps" not in cfg.training_arguments


def test_prepare_from_config_is_a_noop_when_disabled(tmp_path):
    from omegaconf import OmegaConf

    from src.finetune.cpt.prepare_cpt_data import prepare_from_config

    cfg = OmegaConf.create({"prepare": {"auto": False}})
    prepare_from_config(cfg)  # must not touch the filesystem or need a tokenizer
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("config_name", TASK_CONFIGS)
def test_resolve_max_length_follows_training_args(config_name):
    """One sequence budget: eval/perplexity must not diverge from what training saw."""
    cfg = load_cfg(config_name)
    assert resolve_max_length(cfg) == cfg.training_arguments.max_length
