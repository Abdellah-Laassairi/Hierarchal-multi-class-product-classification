import glob
import os
from pathlib import Path
from typing import List, Union

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pyparsing import Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from treelib import Node, Tree

from utils.rich import *


class HDataset(Dataset):
    def __init__(
        self,
        data: pd.core.frame.DataFrame,
        target_columns: List[str],
        A=None,
        B=None,
        C=None,
        is_test: bool = False,
    ):
        self.is_test = is_test

        self.target = data[target_columns].values

        data.drop(target_columns, inplace=True, axis=1)

        self.features = data.drop(["product_id"], axis=1).values

    def __getitem__(self, idx):
        data = self.features[idx]
        if self.is_test:
            return torch.tensor(data, dtype=torch.float32)  # x
        else:
            target = self.target[idx]
            return torch.tensor(data, dtype=torch.float32), torch.tensor(
                target, dtype=torch.long
            )  # x, y

    def __len__(self):
        return len(self.features)
