import os

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url

# import resource
from .dataset import TimeSeriesDataset


# 继承了TimeSeriesDataset基类
class Electricity(TimeSeriesDataset):
    """The raw dataset is in https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. 
    It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014.
    Because the some dimensions are equal to 0. So we eliminate the records in 2011.
    Final we get data contains electircity consumption of 321 clients from 2012 to 2014. 
    And we converted the data to reflect hourly consumption.
    
    due to the missing data , we use the data processed in paper
    《H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.”》
    
    """
    # 定义数据集名称并设置指定值
    name: str = 'electricity'
    # 特定的特征数
    num_features: int = 321
    # 特定的事件序列频率
    freq: str = 't'  # in minuts
    # 总长度
    length: int = 26304

    # 实现download方法
    def download(self):
        # 下载指定url的csv文件然后下载到指定目录下，定义文件名称以及MD5校验
        download_url(
            "https://raw.githubusercontent.com/wayne155/multivariate_timeseries_datasets/main/electricity/electricity.csv",
            self.dir,
            filename="electricity.csv",
            md5="a1973ba4f4bed84136013ffa1ca27dc8",
        )
    # 实现load方法
    def _load(self) -> np.ndarray:
        # 读取csv文件，并将日期存储到dates，数据存储到data中
        self.file_name = os.path.join(self.dir, 'electricity.csv')
        self.df = pd.read_csv(self.file_name, parse_dates=['date'])
        self.dates = pd.DataFrame({'date': self.df.date})
        self.data = self.df.drop("date", axis=1).values
        return self.data
if __name__ == "__main__":
    a = Electricity()