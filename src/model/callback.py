import wandb
from omegaconf import DictConfig
from pytorch_lightning import callbacks


def get_callbacks(config: DictConfig) -> list:
    callback_list = []
    if config.callbacks.early_stopping.enable:
        earlystopping = callbacks.EarlyStopping(
            monitor=config.callbacks.early_stopping.monitor,
            patience=config.callbacks.early_stopping.patience,
            mode=config.callbacks.early_stopping.mode,
            verbose=True,
            strict=True,
        )
        callback_list.append(earlystopping)
    if config.callbacks.model_checkpoint.enable:
        model_checkpoint = callbacks.ModelCheckpoint(
            dirpath=wandb.run.dir,
            filename=wandb.run.name,
            monitor=config.callbacks.model_checkpoint.monitor,
            mode=config.callbacks.model_checkpoint.mode,
            save_top_k=config.callbacks.model_checkpoint.save_top_k,
            save_last=config.callbacks.model_checkpoint.save_last,
        )
        callback_list.append(model_checkpoint)
        
    return callback_list