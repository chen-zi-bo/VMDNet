from typing import Generic, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

StoreType = TypeVar(
    "StoreType", bound=Union[pd.DataFrame, np.ndarray, torch.Tensor]
)  # Type variable for input and output data


# 缩放器的基类，可以处理三种类型数据输入
class Scaler(Generic[StoreType]):
    # fit方法主要是计算一些统计属性
    def fit(self, data: StoreType) -> None:
        """
        Fit the Scaler  to the input dataset.

        Args:
            data:
                The input dataset to fit the Scaler  to.
        Returns:
            None.
        """
        raise NotImplementedError()

    # transform方法主要是将数据集进行缩放
    def transform(self, data: StoreType) -> StoreType:
        """
        Transform the input dataset using the Scaler .

        Args:
            data:
                The input dataset to transform using the Scaler .
        Returns:
            The transformed dataset of the same type as the input data.
        """
        raise NotImplementedError()

    # 反向缩放，把缩放过的数据进行反向转换，恢复原始数据
    def inverse_transform(self, data: StoreType) -> StoreType:
        """
        Perform an inverse transform on the input dataset using the Scaler .

        Args:
            data:
                The input dataset to perform an inverse transform on.
        Returns:
            The inverse transformed dataset of the same type as the input data.
        """
        raise NotImplementedError()


class MaxAbsScaler(Scaler[StoreType]):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    def __init__(self) -> None:
        self.scale = None

    def fit(self, data: StoreType):
        if isinstance(data, np.ndarray):
            self.scale = np.max(np.abs(data), axis=0)
        elif isinstance(data, Tensor):
            self.scale = data.abs().max(axis=0).values
        else:
            raise ValueError(f"not supported type : {type(data)}")

    def transform(self, data) -> StoreType:
        # (b , n)  or (n)
        return data / self.scale

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return data * self.scale
        elif isinstance(data, Tensor):
            return data * torch.tensor(self.scale, device=data.device)
        else:
            raise ValueError(f"not supported type : {type(data)}")

    # def __call__(self, tensor:Tensor):
    #     for ch in tensor:
    #         scale = 1.0 / (ch.max(dim=0)[0] - ch.min(dim=0)[0])
    #         ch.mul_(scale).sub_(ch.min(dim=0)[0])
    #     return tensor


class MinMaxScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    pass


# 继承Scaler基类
class StandarScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    # 构造函数，存在device参数，默认cpu
    def __init__(self, device="cpu") -> None:
        # 三个变量，均值、标准差以及处理的设备
        self.mean = None
        self.std = None
        self.device = device

    # 根据输入数据计算一些统计变量
    def fit(self, data: StoreType):
        # 要处理的数据有不同的类型，根据不同的类型调用不同的api，分别计算出数据的均值和标准差
        if isinstance(data, np.ndarray):
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        elif isinstance(data, Tensor):
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
        else:
            raise ValueError(f"not supported type : {type(data)}")

        # print("Scaler:", self.mean, self.std)
    # 实现标准化，都缩放到均值为 0、标准差为 1 的范围
    def transform(self, data):
        return (data - self.mean) / self.std
    # 标准化的数据反向转换为原始数据
    def inverse_transform(self, data: StoreType) -> StoreType:
        # 也是根据数据的不同类型进行反向转换
        if isinstance(data, np.ndarray):
            return data * self.std + self.mean
        elif isinstance(data, Tensor):
            return data * torch.tensor(self.std, device=data.device) + torch.tensor(
                self.mean, device=data.device
            )
        else:
            raise ValueError(f"not supported type : {type(data)}")


class NoScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """

    def __init__(self, device="cpu") -> None:
        self.device = device

    def fit(self, data: StoreType):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data: StoreType) -> StoreType:
        return data
