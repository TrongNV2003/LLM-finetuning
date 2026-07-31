"""Wiring the trainer up to callbacks and experiment tracking."""

import os
import logging
from typing import List
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

__all__ = ["build_callbacks", "setup_experiment_tracking"]


def build_callbacks(cfg: DictConfig, training_args) -> List:
    from transformers import EarlyStoppingCallback, TrainerCallback

    from src.callbacks.memory_callback import MemoryLoggerCallback
    from src.callbacks.time_callback import TimeLoggerCallback

    callbacks: List[TrainerCallback] = [MemoryLoggerCallback(), TimeLoggerCallback()]

    patience = int(cfg.get("early_stopping_patience", 0) or 0)
    if patience > 0:
        if training_args.load_best_model_at_end:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
            logger.info(f"Early stopping enabled (patience={patience})")
        else:
            logger.warning("early_stopping_patience is set but load_best_model_at_end=false -> ignored")
    return callbacks


def setup_experiment_tracking(cfg: DictConfig, training_args) -> None:
    """Make the `logging.mlflow` block live instead of dead config."""
    report_to = training_args.report_to or []
    if isinstance(report_to, str):
        report_to = [report_to]
    if "mlflow" in report_to:
        experiment_name = cfg.logging.mlflow.experiment_name
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
