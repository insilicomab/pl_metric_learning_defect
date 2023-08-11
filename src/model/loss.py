import torch
from omegaconf import DictConfig
from torch import nn


def get_loss_fn(config: DictConfig) -> torch.nn.Module:
    if config.loss_fn.name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown optimizer: {config.loss_fn.name}")
