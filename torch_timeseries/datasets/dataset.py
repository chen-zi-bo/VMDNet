import os
from abc import abstractmethod
from enum import Enum, unique
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch.utils.data


@unique
# 定义了一个类，这个类继承了str和Enum，枚举类
class Freq(str, Enum):
    seconds = "s"
    minutes = "t"
    hours = "h"
    days = "d"
    months = "m"
    years = "y"


# 数据集的基类
class Dataset(torch.utils.data.Dataset):
    # 数据集名称
    name: str
    # 特征数量
    num_features: int
    # 长度
    length: int
    # 时间频率
    freq: Freq

    def __init__(self, root: str):
        """_summary_

        Args:
            root (str): data save location
            transform (Optional[Callable], optional): _description_. Defaults to None.
            target_transform (Optional[Callable], optional): _description_. Defaults to None.
            single_step (bool, optional): True for single_step data, False for multi_steps data. Defaults to True.
        """
        super().__init__()
        # 一个num数据存储数据
        self.data: np.ndarray
        # 一个pandas类型的结构化数据
        self.df: pd.DataFrame

    # 下载数据集方法，需要子类去实现
    def download(self):
        r"""Downloads the dataset to the :obj:`self.dir` folder."""
        raise NotImplementedError

    # 实现父类方法，获得样本数量
    def __len__(self):
        return len(self.data)


# StoreTypes = Union[np.ndarray, Tensor]
StoreTypes = np.ndarray

# 时间序列数据集子类
class TimeSeriesDataset(Dataset):
    # 接收一个参数root，root代表数据存放的根目录，默认为data文件夹
    def __init__(self, root: str = './data'):
        """_summary_

        Args:
            root (str): data save location

        """
        super().__init__(root)
        # 存储根目录设置
        self.root = root
        # 根据root和name数据集名称构建出完整的存储路径
        self.dir = os.path.join(root, self.name)
        # 创建对应的目录，有则无事，无则创建
        os.makedirs(self.dir, exist_ok=True)

        # 初始化时就会调用这些函数，包括下载加载，具体实现子类去实现，而process没有什么处理
        # self.download()
        self._process()
        self._load()
        # 设置变量dates
        self.dates: Optional[pd.DataFrame]

    @abstractmethod
    def download(self):
        r"""Downloads the dataset to the :obj:`self.dir` folder."""
        raise NotImplementedError

    def _process(self):
        pass

    @abstractmethod
    def _load(self) -> StoreTypes:
        """Loads the dataset to the :attr:`self.data` .

        Raises:
            NotImplementedError: _description_

        Returns:
            T: should return a numpy.array or torch.tensor or pandas.Dataframe
        """

        raise NotImplementedError


class TimeSeriesStaticGraphDataset(TimeSeriesDataset):
    adj: np.ndarray

    def _load_static_graph(self):
        raise NotImplementedError()

# 时间序列的子集
class TimeseriesSubset(torch.utils.data.Subset):
    def __init__(self, dataset: TimeSeriesDataset, indices: Sequence[int]) -> None:
        # 父类中的两个变量
        self.dataset = dataset
        # 索引范围
        self.indices = indices

        self.data = self.dataset.data[indices]
        self.df = self.dataset.df.iloc[indices]
        self.dates = self.dataset.dates.iloc[indices]
        self.num_features = dataset.num_features
        self.name = dataset.name
        self.length = len(self.indices)
        self.freq = dataset.freq
