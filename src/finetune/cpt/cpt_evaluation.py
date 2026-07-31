import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import src.env_setup  # noqa: E402, F401  (sets env vars before torch is imported)

import json
import math
import logging

import hydra
import torch
from typing import List, Optional, cast
from omegaconf import DictConfig
from transformers import set_seed

from src.env_setup import resolve_path
from src.models import (
    ensure_on_device,
    inference_cache,
    load_model_for_inference,
    load_tokenizer,
)
from src.utils.config import resolve_max_length
from src.utils.data import load_dataset

logger = logging.getLogger(__name__)


class CPTEvaluator:
    """Evaluate a CPT model: perplexity on the validation chunks, plus optional
    generation samples for a qualitative look at the acquired domain style."""

    def __init__(
        self, cfg: DictConfig, model=None, tokenizer=None, model_path: Optional[str] = None
    ) -> None:
        self.cfg = cfg
        self.data_args = cfg.dataset
        self.model_args = cfg.model
        self.max_length = resolve_max_length(cfg)
        self.model_path = resolve_path(model_path or cfg.training_arguments.output_dir)

        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(cfg)
        self.model = (
            model
            if model is not None
            else load_model_for_inference(cfg, self.model_path, self.tokenizer)
        )
        logger.info(f"Evaluating model: {self.model_path}")

    @torch.no_grad()
    def evaluate_perplexity(self):
        """Token-weighted perplexity = exp(total_loss / predicted_tokens)."""
        device = ensure_on_device(self.model)
        self.model.eval()

        val_data = load_dataset(self.data_args.validation_file)
        if not val_data:
            raise ValueError(
                f"{self.data_args.validation_file} is empty — nothing to compute perplexity on"
            )

        logger.info(f"Evaluating perplexity on {len(val_data)} samples")

        total_loss = 0.0
        total_tokens = 0

        for idx, sample in enumerate(val_data):
            tokenized = self.tokenizer(
                sample["text"],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            # The loss averages over predicted positions
            num_predicted = max(1, int(attention_mask.sum().item()) - 1)
            total_loss += outputs.loss.item() * num_predicted
            total_tokens += num_predicted

            if (idx + 1) % 100 == 0:
                logger.info(
                    f"  [{idx + 1}/{len(val_data)}] Running perplexity: "
                    f"{math.exp(total_loss / total_tokens):.4f}"
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Perplexity: {perplexity:.4f}")
        logger.info(f"Total tokens evaluated: {total_tokens:,}")

        return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        device = ensure_on_device(self.model)
        self.model.eval()

        tokenized = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(device)

        # eos_token_id is left to the checkpoint's generation_config
        with inference_cache(self.model):
            output = self.model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = output[0][tokenized["input_ids"].shape[1] :]
        # `decode` is annotated `str | list[str]` for the batched case; a single
        # sequence goes in here, so a single string comes out.
        text = cast(str, self.tokenizer.decode(generated, skip_special_tokens=True)).strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text

    def full_evaluation(self, generation_prompts: Optional[List[str]] = None):
        ppl_result = self.evaluate_perplexity()

        generation_results = []
        if generation_prompts:
            logger.info(f"Running generation tests with {len(generation_prompts)} prompts")
            for i, prompt in enumerate(generation_prompts):
                generated = self.generate_text(prompt)
                generation_results.append({"prompt": prompt, "generated": generated})
                logger.info(f"  Prompt {i + 1}: {prompt[:80]}...")
                logger.info(f"  Generated: {generated[:200]}...")

        results = {
            "model_path": self.model_path,
            "perplexity": ppl_result,
            "generation_results": generation_results,
        }

        results_file = resolve_path("cpt_evaluation_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to {results_file}")

        print(f"\n{'=' * 60}")
        print(f"{'CPT Evaluation Results':^60}")
        print(f"{'=' * 60}")
        print(f"Model: {self.model_path}")
        print(f"Perplexity:     {ppl_result['perplexity']:.4f}")
        print(f"Average Loss:   {ppl_result['avg_loss']:.4f}")
        print(f"Total Tokens:   {ppl_result['total_tokens']:,}")
        if generation_results:
            print(f"\nGeneration Tests: {len(generation_results)}")
            for i, gen in enumerate(generation_results):
                print(f"\n  [{i + 1}] Prompt: {gen['prompt'][:80]}...")
                print(f"      Output: {gen['generated'][:200]}...")
        print(f"{'=' * 60}")

        return results


@hydra.main(version_base=None, config_path="../../conf", config_name="cpt-conf")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=bool(cfg.get("deterministic", False)))

    evaluator = CPTEvaluator(cfg=cfg)
    evaluator.full_evaluation(generation_prompts=None)


if __name__ == "__main__":
    main()
