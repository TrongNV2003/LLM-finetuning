import logging
import numpy as np
from pyvi import ViTokenizer
from typing import Dict, List
from difflib import SequenceMatcher
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Returned instead of 0.0 when a metric cannot be computed. A zero is
# indistinguishable from a genuinely terrible model; NaN is not.
UNAVAILABLE = float("nan")

METRIC_NAMES = ("exact_match", "bleu_score", "rouge_l", "lexical_similarity")


@dataclass
class EvaluationResult:
    exact_match: float
    bleu_score: float
    rouge_l: float
    lexical_similarity: float
    num_samples: int


def _mean(scores: List[float]) -> float:
    """`np.mean([])` is a NaN plus a RuntimeWarning; be explicit instead."""
    return float(np.mean(scores)) if scores else UNAVAILABLE


class EvaluateMetrics:
    def _word_tokenize(self, text: str) -> List[str]:
        return ViTokenizer.tokenize(text.strip()).split()

    def _check_pairs(self, predictions: List[str], references: List[str]) -> None:
        if len(predictions) != len(references):
            raise ValueError(
                f"Mismatched sizes: {len(predictions)} predictions vs {len(references)} references"
            )

    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        self._check_pairs(predictions, references)
        if not predictions:
            return UNAVAILABLE

        matches = sum(
            1
            for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        return matches / len(predictions)

    def bleu_score(self, predictions: List[str], references: List[str]) -> float:
        self._check_pairs(predictions, references)
        smoothing = SmoothingFunction().method1
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = self._word_tokenize(pred.lower())
            ref_tokens = [self._word_tokenize(ref.lower())]

            if not pred_tokens or not ref_tokens[0]:
                scores.append(0.0)
                continue

            scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing))

        return _mean(scores)

    def rouge_l_score(self, predictions: List[str], references: List[str]) -> float:
        self._check_pairs(predictions, references)
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = self._word_tokenize(pred.lower())
            ref_tokens = self._word_tokenize(ref.lower())

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            lcs_length = self._lcs_length(pred_tokens, ref_tokens)

            precision = lcs_length / len(pred_tokens)
            recall = lcs_length / len(ref_tokens)

            if precision + recall > 0:
                f_score = 2 * precision * recall / (precision + recall)
            else:
                f_score = 0.0

            scores.append(f_score)

        return _mean(scores)

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def lexical_similarity_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate lexical similarity using string matching"""
        self._check_pairs(predictions, references)
        scores = [
            SequenceMatcher(None, pred.lower(), ref.lower()).ratio()
            for pred, ref in zip(predictions, references)
        ]
        return _mean(scores)

    def metrics_evaluate(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        self._check_pairs(predictions, references)

        return EvaluationResult(
            exact_match=self.exact_match(predictions, references),
            bleu_score=self.bleu_score(predictions, references),
            rouge_l=self.rouge_l_score(predictions, references),
            lexical_similarity=self.lexical_similarity_score(predictions, references),
            num_samples=len(predictions),
        )

    def print_evaluation_report(
        self, result: EvaluationResult, title: str = "Evaluation Results"
    ) -> None:
        print(f"\n{'=' * 60}")
        print(f"{title:^60}")
        print(f"{'=' * 60}")
        print(f"Number of samples: {result.num_samples}")
        print(f"{'=' * 60}")
        print(f"Exact Match Accuracy:  {result.exact_match:.4f}")
        print(f"Lexical Similarity:    {result.lexical_similarity:.4f}")
        print(f"BLEU Score:            {result.bleu_score:.4f}")
        print(f"ROUGE-L Score:         {result.rouge_l:.4f}")
        print(f"{'=' * 60}")


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to token ids on the accelerator, before they are gathered.

    The Trainer otherwise accumulates the full `[batch, seq, vocab]` tensor for
    the entire eval set (with a 248k-token vocab that is tens of GB of host RAM,
    which `eval_accumulation_steps` only moves to CPU, not shrinks).
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics_fn(tokenizer):
    """Create a compute_metrics function for use with Transformers Trainer.

    Uses the label mask (-100) to score only completion tokens, which avoids
    unreliable string splitting on chat-template special tokens.

    These numbers are teacher-forced (each step is predicted from the *gold*
    prefix), so they run ahead of what free-running generation achieves. Treat
    them as a training signal, not as the reported quality — that comes from
    `Predictor` / `evaluation.py`.
    """
    metrics_calculator = EvaluateMetrics()

    def unavailable(reason: str) -> Dict[str, float]:
        logger.warning(f"Eval metrics unavailable: {reason}")
        return dict.fromkeys(METRIC_NAMES, UNAVAILABLE)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if predictions.ndim > 2:  # no preprocess_logits_for_metrics was used
            predictions = np.argmax(predictions, axis=-1)

        # Labels come back unshifted: logits at position t predict token t+1,
        # so predictions must be aligned one step to the left before masking,
        # otherwise every decoded string is off by one token.
        shifted_preds = predictions[:, :-1].astype(np.int32)
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != -100

        processed_preds = []
        processed_labels = []

        for pred_row, label_row, mask_row in zip(shifted_preds, shifted_labels, mask):
            if not mask_row.any():
                continue
            processed_preds.append(
                tokenizer.decode(pred_row[mask_row], skip_special_tokens=True).strip()
            )
            processed_labels.append(
                tokenizer.decode(
                    label_row[mask_row].astype(np.int32), skip_special_tokens=True
                ).strip()
            )

        if not processed_preds:
            return unavailable("no completion tokens in the eval batch (all labels masked)")

        result = metrics_calculator.metrics_evaluate(processed_preds, processed_labels)
        return {
            "exact_match": result.exact_match,
            "bleu_score": result.bleu_score,
            "rouge_l": result.rouge_l,
            "lexical_similarity": result.lexical_similarity,
        }

    return compute_metrics
