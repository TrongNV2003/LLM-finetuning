"""Metric edge cases — the ones that used to raise or silently return 0.0."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("pyvi")
pytest.importorskip("nltk")

from src.finetune.sft.metrics import EvaluateMetrics  # noqa: E402


@pytest.fixture
def metrics():
    return EvaluateMetrics()


def test_identical_predictions_score_perfectly(metrics):
    texts = ["68, Xuân Thủy, Cầu Giấy, Hà Nội", "123, Nguyễn Du, Quận 1, Hồ Chí Minh"]
    result = metrics.metrics_evaluate(texts, list(texts))
    assert result.exact_match == 1.0
    assert result.lexical_similarity == 1.0
    assert result.rouge_l == pytest.approx(1.0)
    assert result.num_samples == 2


def test_partial_match_is_between_zero_and_one(metrics):
    result = metrics.metrics_evaluate(
        ["45, Lê Thánh, Hoàn Kiếm, Hà Nội"], ["45, Lê Thánh Tông, Hoàn Kiếm, Hà Nội"]
    )
    assert result.exact_match == 0.0
    assert 0.0 < result.rouge_l < 1.0
    assert 0.0 < result.lexical_similarity < 1.0


def test_empty_input_returns_nan_not_a_crash(metrics):
    """Used to be ZeroDivisionError for exact_match and a warning-laden nan elsewhere."""
    result = metrics.metrics_evaluate([], [])
    assert math.isnan(result.exact_match)
    assert math.isnan(result.bleu_score)
    assert math.isnan(result.rouge_l)
    assert math.isnan(result.lexical_similarity)
    assert result.num_samples == 0


def test_empty_strings_score_zero_not_nan(metrics):
    result = metrics.metrics_evaluate(["", ""], ["Hà Nội", "Hồ Chí Minh"])
    assert result.exact_match == 0.0
    assert result.bleu_score == 0.0
    assert result.rouge_l == 0.0


def test_mismatched_lengths_raise(metrics):
    with pytest.raises(ValueError, match="Mismatched sizes"):
        metrics.metrics_evaluate(["a", "b"], ["a"])


def test_exact_match_ignores_case_and_surrounding_space(metrics):
    assert metrics.exact_match(["  Hà Nội  "], ["hà nội"]) == 1.0


def test_lcs_length(metrics):
    assert metrics._lcs_length(["a", "b", "c"], ["a", "c"]) == 2
    assert metrics._lcs_length([], ["a"]) == 0
