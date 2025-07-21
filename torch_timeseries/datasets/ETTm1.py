import os

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url

# import resource
from .dataset import Freq, TimeSeriesDataset


class ETTm1(TimeSeriesDataset):
    name: str = 'ETTm1'
    num_features: int = 7
    freq: Freq = 't'
    length: int = 69680
    windows: int = 384

    def download(self):
        download_url(
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
            self.dir,
            filename='ETTm1.csv',
            md5="82d6bd89109c63d075d99c1077b33f38"
        )

    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, 'ETTm1.csv')
        self.df = pd.read_csv(self.file_path, parse_dates=[0])
        self.dates = pd.DataFrame({'date': self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data


if __name__ == "__main__":
    a = ETTm1()
