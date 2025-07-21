import os

import numpy as np
import pandas as pd

from .dataset import Freq, TimeSeriesStaticGraphDataset


class PeMS_D7(TimeSeriesStaticGraphDataset):
    name:str= 'PeMS-D7'
    num_features: int = 228
    length : int  = 12672
    freq : Freq = 'h'
    windows : int = 288
    
    def download(self):
        """download from https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/dataset
        """
        pass
        
    def _load(self) -> np.ndarray:
        self._load_static_graph()
        self.file_path = os.path.join(self.dir, 'V_228.csv')
        self.df = pd.read_csv(self.file_path, header=None)
        self.df['date'] = pd.date_range(start='05/01/2012 00:00', periods=self.length, freq='5T')  # '5T' for 5 minutes
        self.dates =  pd.DataFrame({'date': self.df['date'] })
        self.data = self.df.drop("date", axis=1).values
        return self.data
    
    def _load_static_graph(self):
        self.adj = pd.read_csv(os.path.join(self.dir, 'W_228.csv'), header=None).values
        
    
    