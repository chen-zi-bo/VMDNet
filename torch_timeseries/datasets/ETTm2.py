import os

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url

# import resource
from .dataset import Freq, TimeSeriesDataset


class ETTm2(TimeSeriesDataset):
    name:str= 'ETTm2'
    num_features: int = 7
    freq : Freq = 't'
    length : int  = 69680
    windows : int = 384
    
    def download(self):
        download_url(
         "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
         self.dir,
         filename='ETTm2.csv',
         md5="7687e47825335860bf58bccb31be0c56"
        )

        
    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, 'ETTm2.csv')
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
    
if __name__ == "__main__":
    a = ETTm2()