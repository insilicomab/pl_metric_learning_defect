import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from src.utils import data_split
from src.dataset import DataModule
from src.model import Net, get_callbacks


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    # data split
    x_train, x_val, x_test, y_train, y_val, y_test = data_split(config=cfg)

    # set random seed
    pl.seed_everything(seed=cfg.seed, workers=True)

    # datamodule
    datamodule = DataModule(
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

    # trainer
    trainer = pl.Trainer(
        logger=False,
        max_epochs=cfg.trainer.max_epochs,
        callbacks=get_callbacks(config=cfg),
        gpus=cfg.trainer.gpus,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        auto_lr_find=cfg.trainer.auto_lr_find,
        deterministic=cfg.trainer.deterministic,
    )
    
    # train
    trainer.fit(net, datamodule=datamodule)

    # test
    trainer.test(net, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
    