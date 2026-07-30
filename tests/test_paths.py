"""Path anchoring. Stdlib + python-dotenv only, so this runs without the ML stack."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env_setup import PROJECT_ROOT, resolve_path  # noqa: E402


def test_project_root_is_the_repo_root():
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "src", "conf"))
    assert os.path.isfile(os.path.join(PROJECT_ROOT, "pyproject.toml"))


def test_relative_paths_anchor_to_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("data/x.json") == os.path.join(PROJECT_ROOT, "data", "x.json")
    assert resolve_path("./finetuning-checkpoints") == os.path.join(
        PROJECT_ROOT, "finetuning-checkpoints"
    )


def test_absolute_paths_pass_through():
    assert resolve_path("/abs/x.json") == "/abs/x.json"


def test_empty_values_pass_through():
    assert resolve_path(None) is None
    assert resolve_path("") == ""


def test_process_env_is_set_before_torch_would_load():
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["HYDRA_FULL_ERROR"] == "1"
