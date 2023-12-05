import torch
from torch.utils.data import Dataset

import pandas as pd

from typing import *


class SentimentData(Dataset):
    # инициализация датасета
    def __init__(
        self, dataframe: pd.DataFrame, mode: str, 
        col_name: str, split_param: float=0.9
    ) -> None:        
        assert self.mode in ['val', 'train', 'test']

        self.mode = mode # train/test
        self.data = dataframe # data
        self.col_name = col_name # column for analyzing
        
        data_size = self.data.shape[0]
        if self.mode in ['val', 'train']:
            if self.mode == 'train':
                self.data = self.data.iloc[:int(data_size * split_param)]
            else:
                self.data = self.data.iloc[int(data_size * split_param):]

    # для получения размера датасета
    def __len__(self) -> int:
        return self.data.shape[0]

    # для получения элемента по индексу
    def __getitem__(
        self, index: int
    ) -> Dict[str, Union[str, torch.Tensor]]:
        text = self.data.iloc[index][self.col_name]
        target1 = self.data.iloc[index]['0class']
        target2 = self.data.iloc[index]['1class']

        return {
            'text': text,
            'target1': torch.tensor(target1, dtype=torch.long),
            'target2': torch.tensor(target2, dtype=torch.long)
        }