import os
import shutil
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig
import wandb

from src.utils import data_split
from src.dataset import DataModule
from src.model import Net, get_callbacks


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    # initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config={
            'data': os.path.basename(cfg.wandb.data_dir),
            'model': cfg.wandb.model_name,
            'layer': cfg.wandb.layer_name,
        }
    )

    # save crucial artifacts before training begins
    shutil.copy2('config/config.yaml', os.path.join(wandb.run.dir, 'hydra_config.yaml'))

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
        logger=WandbLogger(project=cfg.wandb.project),
        max_epochs=cfg.trainer.max_epochs,
        callbacks=get_callbacks(config=cfg),
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        auto_lr_find=cfg.trainer.auto_lr_find,
        deterministic=cfg.trainer.deterministic,
    )
    
    # train
    trainer.fit(net, datamodule=datamodule)

    # test
    trainer.test(net, datamodule=datamodule, ckpt_path="best")
    shutil.copy2(
        'output/classification_report.txt',
        os.path.join(wandb.run.dir,'classification_report.txt')
    )
    shutil.copy2(
        'output/confusion_matrix.png',
        os.path.join(wandb.run.dir,'confusion_matrix.png')
    )
    shutil.copy2(
        'output/roc_curve.png',
        os.path.join(wandb.run.dir,'roc_curve.png')
    )



if __name__ == "__main__":
    main()
    