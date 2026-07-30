import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import src.env_setup  # noqa: E402, F401  (sets env vars before torch is imported)

import hydra
import logging
from typing import Optional
from omegaconf import DictConfig
from transformers import set_seed

from src.finetune.sft.predictor import Predictor
from src.models import load_model_for_inference, load_tokenizer
from src.utils.data import load_dataset
from src.env_setup import resolve_path

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """Standalone evaluation of a checkpoint (adapter or merged model)."""

    def __init__(self, cfg: DictConfig, model_path: Optional[str] = None) -> None:
        self.cfg = cfg
        self.data_args = cfg.dataset
        self.model_path = resolve_path(model_path or cfg.training_arguments.output_dir)

        self.tokenizer = load_tokenizer(cfg)
        self.model = load_model_for_inference(cfg, self.model_path, self.tokenizer)
        self.predictor = Predictor(self.model, self.tokenizer, cfg)
        logger.info(f"Model loaded from: {self.model_path}")

    def evaluate(self, results_file: str = "evaluation_results.json"):
        if not self.data_args.test_file:
            raise ValueError(
                f"dataset={self.data_args.dataset_name} has no test_file to evaluate on"
            )
        test_data = load_dataset(self.data_args.test_file)
        result, detailed = self.predictor.evaluate_samples(
            test_data, self.data_args.text_col, self.data_args.label_col
        )
        self.predictor.metrics.print_evaluation_report(result, "Model Evaluation Results")
        self.predictor.save_results(results_file, result, detailed, self.model_path)
        return result

    def evaluate_single(self, text: str) -> str:
        return self.predictor.predict_one(text)


@hydra.main(version_base=None, config_path="../../conf", config_name="sft-conf")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=bool(cfg.get("deterministic", False)))

    evaluator = LLMEvaluator(cfg=cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
