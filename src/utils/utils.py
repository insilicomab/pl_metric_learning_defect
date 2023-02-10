import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


def data_split(config: DictConfig) -> tuple:
    train_df = pd.read_csv(config.train_df_dir)
    image_name_list = train_df['id'].values
    label_list = train_df['target'].values
    x_train, x_val, y_train, y_val = train_test_split(
        image_name_list, 
        label_list, 
        test_size=config.train_test_split.test_size, 
        stratify=label_list, 
        random_state=config.train_test_split.random_state
    )
    x_test, y_test = x_val, y_val
    return x_train, x_val, x_test, y_train, y_val, y_test