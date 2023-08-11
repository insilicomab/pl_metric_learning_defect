import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from .transformation import TestTransforms


class ImageDataset(Dataset):
    def __init__(
        self,
        image_name_list: np.ndarray,
        label_list: np.ndarray,
        img_dir: str,
        transform: Compose = None,
        phase: str = None,
    ) -> None:
        self.image_name_list = image_name_list
        self.label_list = label_list
        self.img_dir = img_dir
        self.phase = phase
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_name_list)

    def __getitem__(self, index) -> tuple:
        image_path = os.path.join(self.img_dir, self.image_name_list[index])
        image = Image.open(image_path)
        image = self.transform(self.phase, image)
        label = self.label_list[index]

        return image, label


def get_inference_dataloader(df_dir: str, img_dir: str, image_size: int) -> DataLoader:
    # read test data
    test_df = pd.read_csv(df_dir, header=None)
    x_test = test_df[0].values

    # test dataset
    test_dataset = ImageDataset(
        x_test,
        x_test,
        img_dir=img_dir,
        transform=TestTransforms(image_size=image_size),
        phase="test",
    )

    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_dataloader


def get_image_dataset(
    df_dir: str,
    img_dir: str,
    image_size: int,
) -> tuple:
    # read data
    image_df = pd.read_csv(df_dir)

    # train image name list & label list
    image_name_list = image_df["id"]
    label_list = image_df["target"]

    # index2target: key=index, value=target
    index2target = image_df["target"].to_dict()

    image_dataset = ImageDataset(
        image_name_list,
        label_list,
        img_dir=img_dir,
        transform=TestTransforms(image_size=image_size),
        phase="test",
    )

    return image_dataset, index2target
