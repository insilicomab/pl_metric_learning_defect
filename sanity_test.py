import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from .src.utils import data_split


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    # data split
    x_train, x_val, y_train, y_val = data_split(config=cfg)

    # set random seed
    pl.seed_everything(seed=cfg.seed, workers=True)

    