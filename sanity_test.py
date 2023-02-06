import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from .src.utils import data_split
from .src.dataset import ImageDataset
from .src.model import Net


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    # data split
    x_train, x_val, x_test, y_train, y_val, y_test = data_split(config=cfg)

    # set random seed
    pl.seed_everything(seed=cfg.seed, workers=True)

    # datamodule
    datamodule = ImageDataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        config=cfg
    )

    # net
    net = Net(config=cfg)

    
    