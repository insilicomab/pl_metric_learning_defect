import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig


def get_optimizer(config: DictConfig, net: nn.Module) -> optim.Optimizer:
    if config.optimizer.name == "Adam":
        return optim.Adam(
            net.parameters(),
            lr=config.optimizer.adam.lr,
            weight_decay=config.optimizer.adam.weight_decay,
        )
    elif config.optimizer.name == "AdamW":
        return optim.AdamW(
            net.parameters(),
            lr=config.optimizer.adamW.lr,
            weight_decay=config.optimizer.adamW.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")
