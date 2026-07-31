import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import src.env_setup  # noqa: E402, F401  (sets env vars before torch is imported)

import hydra
import logging
from typing import cast
from omegaconf import DictConfig
from trl import SFTConfig, SFTTrainer
from datasets import Dataset, DatasetDict
from transformers import set_seed

from src.finetune.sft.predictor import Predictor
from src.finetune.sft.prompts import PROMPT_TEMPLATE
from src.finetune.sft.metrics import compute_metrics_fn, preprocess_logits_for_metrics
from src.utils.experiment import build_callbacks, setup_experiment_tracking
from src.utils.training_args import build_training_args
from src.utils.config import as_container
from src.utils.data import load_dataset
from src.models import (
    free_gpu_memory,
    load_model_for_training,
    load_tokenizer,
    merge_from_config,
)

logger = logging.getLogger(__name__)


class Dataloader:
    """Turns `{"text", "label"}` rows into a TRL prompt-completion dataset."""

    def __init__(self, cfg: DictConfig, tokenizer) -> None:
        self.data_args = cfg.dataset
        self.model_args = cfg.model
        self.tokenizer = tokenizer
        self.chat_template_kwargs = as_container(self.model_args.get("chat_template_kwargs")) or {}

        self.raw_datasets = DatasetDict()
        for split, file_key in (
            ("train", "train_file"),
            ("validation", "validation_file"),
            ("test", "test_file"),
        ):
            file_path = self.data_args.get(file_key)
            if not file_path:
                continue
            rows = load_dataset(file_path)
            if rows:
                self.raw_datasets[split] = Dataset.from_list(rows)

    def preprocess_fn(self, examples):
        prompts, completions = [], []
        for text, label in zip(
            examples[self.data_args.text_col], examples[self.data_args.label_col]
        ):
            prompts.append([{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}])
            completions.append([{"role": "assistant", "content": label}])

        batch = {"prompt": prompts, "completion": completions}
        if self.chat_template_kwargs:
            batch["chat_template_kwargs"] = [self.chat_template_kwargs] * len(prompts)
        return batch

    def get_processed_datasets(self, training_args) -> DatasetDict:
        if not self.raw_datasets:
            raise ValueError("No datasets available for processing")

        reference_split = next(iter(self.raw_datasets))
        column_names = self.raw_datasets[reference_split].column_names

        # `datasets` errors out when num_proc exceeds the number of rows in a split.
        smallest_split = min(len(split) for split in self.raw_datasets.values())
        num_proc = max(1, min(int(self.data_args.preprocessing_num_workers or 1), smallest_split))

        with training_args.main_process_first(desc="dataset mapping"):
            datasets = self.raw_datasets.map(
                self.preprocess_fn,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                num_proc=num_proc,
                desc="Building prompt-completion dataset",
            )

        if "train" not in datasets:
            raise ValueError("Training requires a train dataset (dataset.train_file)")
        train_dataset = datasets["train"]
        if self.data_args.shuffle:
            train_dataset = cast(Dataset, train_dataset.shuffle(seed=training_args.seed))
        if self.data_args.max_train_samples is not None:
            limit = min(len(train_dataset), self.data_args.max_train_samples)
            train_dataset = cast(Dataset, train_dataset.select(range(limit)))
        datasets["train"] = train_dataset

        if training_args.do_eval:
            if "validation" not in datasets:
                raise ValueError(
                    "do_eval=true requires dataset.validation_file. Set do_eval=false, or "
                    "carve a validation split out of train — the test split must not be used, "
                    "or early stopping would select on the split you report on."
                )
            eval_dataset = datasets["validation"]
            if self.data_args.max_eval_samples is not None:
                limit = min(len(eval_dataset), self.data_args.max_eval_samples)
                eval_dataset = cast(Dataset, eval_dataset.select(range(limit)))
            datasets["validation"] = eval_dataset

        if "test" in datasets and self.data_args.max_test_samples is not None:
            limit = min(len(datasets["test"]), self.data_args.max_test_samples)
            datasets["test"] = cast(Dataset, datasets["test"].select(range(limit)))

        return datasets


class LLMFinetuning:
    def __init__(self, cfg: DictConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.data_args = cfg.dataset
        self.model_args = cfg.model
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(cfg)
        self.model = load_model_for_training(cfg, self.tokenizer)

    def train(self, train_dataset, training_args, eval_dataset=None):
        compute_metrics = compute_metrics_fn(self.tokenizer) if eval_dataset is not None else None

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if compute_metrics else None,
            callbacks=build_callbacks(self.cfg, training_args),
        )
        trainer.train()
        trainer.save_model()
        logger.info(f"Model saved to {training_args.output_dir}")
        return trainer

    def release(self):
        """Drop the training model so the merge/eval load does not fight it for VRAM."""
        self.model = None
        free_gpu_memory()

    def evaluate(self, results_file: str = "evaluation_results.json"):
        """Generation-based evaluation on the test split."""
        test_data = load_dataset(self.data_args.test_file)
        predictor = Predictor(self.model, self.tokenizer, self.cfg)

        result, detailed = predictor.evaluate_samples(
            test_data, self.data_args.text_col, self.data_args.label_col
        )
        predictor.metrics.print_evaluation_report(result, "Final Model Evaluation")
        predictor.save_results(
            results_file, result, detailed, self.cfg.training_arguments.output_dir
        )
        return result


@hydra.main(version_base=None, config_path="../../conf", config_name="sft-conf")
def main(cfg: DictConfig):
    # deterministic=True also flips torch.use_deterministic_algorithms, which
    # errors on some fused/flash kernels — opt in explicitly.
    set_seed(cfg.seed, deterministic=bool(cfg.get("deterministic", False)))

    logger.info("Starting SFT training")
    training_args = build_training_args(cfg, SFTConfig)
    setup_experiment_tracking(cfg, training_args)

    if not training_args.do_train:
        raise ValueError(
            "do_train=false has no meaning here — use src/finetune/sft/evaluation.py "
            "to score an existing checkpoint without training"
        )

    tokenizer = load_tokenizer(cfg)
    dataloader = Dataloader(cfg, tokenizer)
    datasets = dataloader.get_processed_datasets(training_args)

    train_dataset = datasets["train"]
    val_dataset = datasets.get("validation")
    test_dataset = datasets.get("test")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    if test_dataset is not None:
        logger.info(f"Test dataset size: {len(test_dataset)}")

    finetuning = LLMFinetuning(cfg=cfg, tokenizer=tokenizer)
    finetuning.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args,
    )

    merge_cfg = cfg.get("merge") or {}
    merge_enabled = bool(merge_cfg.get("enabled", False)) and cfg.model.get("lora") is not None
    has_test = bool(cfg.dataset.test_file)
    eval_merged = has_test and merge_enabled and bool(merge_cfg.get("evaluate_merged", True))

    if has_test and not eval_merged:
        finetuning.evaluate()

    merged_dir = None
    if merge_enabled:
        finetuning.release()
        merged_dir = merge_from_config(cfg, training_args.output_dir)

    if eval_merged:
        if merged_dir:
            from src.finetune.sft.evaluation import LLMEvaluator

            logger.info("Evaluating the merged model")
            LLMEvaluator(cfg=cfg, model_path=merged_dir).evaluate()
        else:
            logger.warning("Nothing was merged, skipping merged-model evaluation")


if __name__ == "__main__":
    main()
