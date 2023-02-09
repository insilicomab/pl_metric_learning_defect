import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig

from .dataset import ImageDataset
from .transformation import Transforms


class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            x_train: list, 
            y_train: list, 
            x_val: list, 
            y_val: list, 
            x_test: list,
            y_test: list,
            config: DictConfig
        ):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.config = config

    def setup(self, stage=None) -> None:
        self.train_dataset = ImageDataset(
            image_name_list=self.x_train,
            label_list=self.y_train,
            img_dir=self.config.img_dir,
            transform=Transforms(config=self.config),
            phase='train'
        )
        self.val_dataset = ImageDataset(
            image_name_list=self.x_val,
            label_list=self.y_val,
            img_dir=self.config.img_dir,
            transform=Transforms(config=self.config),
            phase='val'
        )
        self.test_dataset = ImageDataset(
            image_name_list=self.x_test,
            label_list=self.y_test,
            img_dir=self.config.img_dir,
            transform=Transforms(config=self.config),
            phase='test'
        )

    def train_dataloader(self) -> DataLoader:   
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_dataloader.batch_size,
            shuffle=self.config.train_dataloader.shuffle,
            num_workers=self.config.train_dataloader.num_workers,
            pin_memory=self.config.train_dataloader.pin_memory,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_dataloader.batch_size,
            shuffle=self.config.val_dataloader.shuffle,
            num_workers=self.config.val_dataloader.num_workers,
            pin_memory=self.config.val_dataloader.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test_dataloader.batch_size,
            shuffle=self.config.test_dataloader.shuffle,
            num_workers=self.config.test_dataloader.num_workers,
            pin_memory=self.config.test_dataloader.pin_memory,
        )