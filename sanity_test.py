import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    # data split
    x_train, x_val, y_train, y_val = data_split(config=cfg)

    # set random seed
    pl.seed_everything(seed=cfg.seed, workers=True)

    