import os

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive

# import resource
from .dataset import TimeSeriesDataset


class ILI(TimeSeriesDataset):
    name:str= 'ILI'
    num_features: int = 7
    sample_rate:int # in munites
    length : int  = 966
    freq:str = 'h'
    windows : int = 48
    
    def download(self):
        download_and_extract_archive(
         "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/illness.zip",
         self.dir,
         filename='illness.zip',
         md5="e18542fafd509ad52ffedde1b0f1018d"
        )
        
        
        
    def _load(self) -> np.ndarray:
        self.file_path =os.path.join( os.path.join(self.dir, 'illness')  , 'national_illness.csv')
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.date})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
if __name__ == "__main__":
    a = ILI()