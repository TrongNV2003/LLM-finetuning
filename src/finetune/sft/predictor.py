"""Batched generation + scoring for the SFT task."""

import json
import torch
import logging
from omegaconf import DictConfig
from typing import Dict, List, Optional, Tuple

from src.finetune.sft.prompts import PROMPT_TEMPLATE
from src.finetune.sft.metrics import EvaluateMetrics, EvaluationResult
from src.utils.config import as_container, resolve_max_length
from src.models import ensure_on_device, inference_cache
from src.env_setup import resolve_path

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, tokenizer, cfg: DictConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model_args = cfg.model
        self.eval_args = cfg.evaluation_arguments
        self.metrics = EvaluateMetrics()
        self.chat_template_kwargs = as_container(self.model_args.get("chat_template_kwargs")) or {}
        self.max_length = resolve_max_length(cfg)

    def format_prompt(self, text: str) -> str:
        """Render exactly what TRL feeds the model as the prompt half of a
        prompt-completion sample (it also uses add_generation_prompt=True)."""
        prompt_str = PROMPT_TEMPLATE.format(text=text)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_str}],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def _generation_kwargs(self) -> Dict:
        kwargs = {
            "max_new_tokens": self.eval_args.max_new_tokens,
            "do_sample": self.eval_args.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.eval_args.do_sample:
            kwargs.update(
                temperature=self.eval_args.temperature,
                top_p=self.eval_args.top_p,
                top_k=self.eval_args.top_k,
            )
        if float(self.eval_args.get("repetition_penalty", 1.0)) != 1.0:
            kwargs["repetition_penalty"] = self.eval_args.repetition_penalty
        return kwargs

    @torch.no_grad()
    def generate(self, texts: List[str]) -> List[str]:
        device = ensure_on_device(self.model)
        self.model.eval()
        batch_size = int(self.eval_args.get("batch_size", 1))
        gen_kwargs = self._generation_kwargs()

        # Left padding is required for correct batched decoder-only generation.
        # truncation_side must be left. the tail of a rendered prompt is the
        # generation prompt itself (`<|im_start|>assistant`), so right-truncating an
        # over-long input would hand the model a prompt that never asks for an answer.
        original_padding_side = self.tokenizer.padding_side
        original_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        predictions: List[str] = []
        try:
            with inference_cache(self.model):
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    prompts = [self.format_prompt(text) for text in batch]
                    encoded = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        add_special_tokens=False,
                    ).to(device)

                    outputs = self.model.generate(**encoded, **gen_kwargs)
                    generated = outputs[:, encoded["input_ids"].shape[1] :]
                    predictions.extend(
                        self.tokenizer.decode(row, skip_special_tokens=True).strip()
                        for row in generated
                    )
                    logger.debug(f"Generated {len(predictions)}/{len(texts)}")
        finally:
            self.tokenizer.padding_side = original_padding_side
            self.tokenizer.truncation_side = original_truncation_side
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predictions

    def predict_one(self, text: str) -> str:
        return self.generate([text])[0]

    def evaluate_samples(
        self, samples: List[Dict], text_col: str = "text", label_col: str = "label"
    ) -> Tuple[EvaluationResult, List[Dict]]:
        if not samples:
            raise ValueError("No samples to evaluate — the test file is empty")

        inputs = [sample[text_col] for sample in samples]
        references = [sample[label_col] for sample in samples]

        logger.info(f"Generating predictions for {len(inputs)} samples")
        predictions = self.generate(inputs)

        result = self.metrics.metrics_evaluate(predictions, references)
        detailed = [
            {
                "id": idx,
                "input": inp,
                "prediction": pred,
                "reference": ref,
                "exact_match": pred.strip().lower() == ref.strip().lower(),
            }
            for idx, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
        ]
        return result, detailed

    @staticmethod
    def save_results(
        results_file: str,
        result: EvaluationResult,
        detailed: List[Dict],
        model_path: Optional[str] = None,
    ) -> None:
        payload = {
            "model_path": model_path,
            "metrics": {
                "exact_match": result.exact_match,
                "bleu_score": result.bleu_score,
                "rouge_l": result.rouge_l,
                "lexical_similarity": result.lexical_similarity,
                "num_samples": result.num_samples,
            },
            "detailed_results": detailed,
        }
        results_file = resolve_path(results_file)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to {results_file}")
