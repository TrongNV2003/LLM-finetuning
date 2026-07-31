import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import src.env_setup  # noqa: E402, F401  (sets env vars before torch is imported)

import math
import hydra
import logging
from typing import cast
from omegaconf import DictConfig
from trl import SFTConfig, SFTTrainer
from datasets import Dataset, DatasetDict
from transformers import set_seed

from src.finetune.cpt.prepare_cpt_data import prepare_from_config
from src.models import (
    free_gpu_memory,
    load_model_for_training,
    load_tokenizer,
    merge_from_config,
)
from src.utils.data import load_dataset
from src.utils.experiment import build_callbacks, setup_experiment_tracking
from src.utils.training_args import build_training_args

logger = logging.getLogger(__name__)


class CPTDataloader:
    """Dataloader for Continued Pre-training.

    Loads pre-chunked JSON produced by `prepare_cpt_data.py` in the form
    `[{"text": "chunk"}, ...]`. No prompt formatting and no completion mask:
    plain causal language modelling over the whole sequence is the objective.
    """
    def __init__(self, cfg: DictConfig, tokenizer) -> None:
        self.data_args = cfg.dataset
        self.model_args = cfg.model
        self.tokenizer = tokenizer

        self.raw_datasets = DatasetDict()
        for split, file_key in (("train", "train_file"), ("validation", "validation_file")):
            file_path = self.data_args.get(file_key)
            if not file_path:
                continue
            rows = load_dataset(file_path)
            if rows:
                self.raw_datasets[split] = Dataset.from_list(rows)

    def get_processed_datasets(self, training_args) -> DatasetDict:
        datasets = self.raw_datasets

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
                raise ValueError("do_eval=true requires a validation dataset")
            eval_dataset = datasets["validation"]
            if self.data_args.max_eval_samples is not None:
                limit = min(len(eval_dataset), self.data_args.max_eval_samples)
                eval_dataset = cast(Dataset, eval_dataset.select(range(limit)))
            datasets["validation"] = eval_dataset

        return datasets


class CPTFinetuning:
    """Continued Pre-training for domain adaptation, scored with perplexity."""

    def __init__(self, cfg: DictConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.data_args = cfg.dataset
        self.model_args = cfg.model
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(cfg)
        self.model = load_model_for_training(cfg, self.tokenizer)

    def release(self):
        """Drop the training model so the merge/eval load does not fight it for VRAM."""
        self.model = None
        free_gpu_memory()

    def train(self, train_dataset, training_args, eval_dataset=None):
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=build_callbacks(self.cfg, training_args),
        )
        trainer.train()
        trainer.save_model()
        logger.info(f"Model saved to {training_args.output_dir}")

        if eval_dataset is not None:
            eval_result = trainer.evaluate()
            eval_loss = eval_result.get("eval_loss")
            if eval_loss is not None:
                logger.info(f"Final eval_loss: {eval_loss:.4f}")
                logger.info(f"Final perplexity: {math.exp(eval_loss):.4f}")

        return trainer


@hydra.main(version_base=None, config_path="../../conf", config_name="cpt-conf")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=bool(cfg.get("deterministic", False)))

    logger.info("Starting CPT training")
    training_args = build_training_args(cfg, SFTConfig)
    setup_experiment_tracking(cfg, training_args)

    if not training_args.do_train:
        raise ValueError(
            "do_train=false has no meaning here — use src/finetune/cpt/cpt_evaluation.py "
            "to score an existing checkpoint without training"
        )

    tokenizer = load_tokenizer(cfg)

    prepare_from_config(cfg, tokenizer=tokenizer)

    dataloader = CPTDataloader(cfg, tokenizer)
    datasets = dataloader.get_processed_datasets(training_args)

    train_dataset = datasets["train"]
    val_dataset = datasets.get("validation")

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"Validation dataset size: {len(val_dataset)}")

    finetuning = CPTFinetuning(cfg=cfg, tokenizer=tokenizer)
    finetuning.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args,
    )

    merge_cfg = cfg.get("merge") or {}
    merge_enabled = bool(merge_cfg.get("enabled", False)) and cfg.model.get("lora") is not None
    eval_merged = val_dataset is not None and merge_enabled and bool(
        merge_cfg.get("evaluate_merged", True)
    )

    if val_dataset is not None and not eval_merged:
        from src.finetune.cpt.cpt_evaluation import CPTEvaluator

        # Reuse the trained model instead of reloading the checkpoint from disk.
        CPTEvaluator(
            cfg=cfg, model=finetuning.model, tokenizer=finetuning.tokenizer
        ).full_evaluation()

    merged_dir = None
    if merge_enabled:
        finetuning.release()
        merged_dir = merge_from_config(cfg, training_args.output_dir)

    if eval_merged:
        from src.finetune.cpt.cpt_evaluation import CPTEvaluator

        if merged_dir:
            logger.info("Evaluating the merged model")
            CPTEvaluator(cfg=cfg, model_path=merged_dir).full_evaluation()
        else:
            logger.warning("Nothing was merged, skipping merged-model evaluation")


if __name__ == "__main__":
    main()
