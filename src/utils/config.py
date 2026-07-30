"""Reading values out of the Hydra config."""

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["as_container", "resolve_max_length"]


def as_container(value):
    """OmegaConf list/dict -> plain python (PEFT rejects OmegaConf types)."""
    if isinstance(value, (ListConfig, DictConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def resolve_max_length(cfg: DictConfig) -> int:
    """The one sequence budget for both training and inference.

    `training_arguments.max_length` is what the trainer truncates to, so eval and
    perplexity follow it rather than keeping a second, silently diverging
    `model.model_max_length`.
    """
    return int(cfg.training_arguments.get("max_length") or cfg.model.get("model_max_length", 2048))
