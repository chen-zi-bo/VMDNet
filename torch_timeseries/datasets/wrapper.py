import torch
from torch.utils.data import Dataset

from torch_timeseries.data.scaler import Scaler
from torch_timeseries.utils.timefeatures import time_features
from .dataset import TimeSeriesDataset, TimeseriesSubset


class MultiStepTimeFeatureSet(Dataset):
    # 构造函数
    #  dataset
    # StandarScaler
    #         # 0
    #         # 96
    #         # 1
    #         # 96
    #         # "h"
    #         # False
    def __init__(self, dataset: TimeseriesSubset, scaler: Scaler, time_enc=0, window: int = 168, horizon: int = 3,
                 steps: int = 2, freq=None, scaler_fit=True, cycleindex=None):
        # 各种参数的赋值
        self.cycle_index = None
        if cycleindex is not None:
            self.cycle_index = cycleindex

        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps
        self.time_enc = time_enc
        self.scaler = scaler

        self.num_features = self.dataset.num_features
        self.length = self.dataset.length
        # 如果未输入频率，则使用传入数据集的频率
        if freq is None:
            self.freq = self.dataset.freq
        else:
            self.freq = freq
        # 判断是否还需要进行拟合
        if scaler_fit:
            # 需要则对数据进行拟合
            self.scaler.fit(self.dataset.data)

        # 拟合完之后实现缩放，并存储缩放之后的数据
        self.scaled_data = self.scaler.transform(self.dataset.data)
        # 传入dataset.dates日期,time_enc默认为0，以及频率，
        # 返回不同时间特征之下的dates数据
        self.date_enc_data = time_features(
            self.dataset.dates, self.time_enc, self.freq)
        print(len(self.dataset))
        # 输入数据长度和预测范围长度不能超过数据集的长度。
        assert len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1 > 0, "Dataset is not long enough!!!"

        # 实现缩放

    def transform(self, values):
        return self.scaler.transform(values)
        # 缩放逆操作

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)

    # 可以遍历获得X、y x——enc、y_enc
    def __getitem__(self, index):
        # x : (B, T, N)
        # y : (B, O, N)
        # x_date_enc : (B, T, D)
        # y_date_eDc : (B, O, D)
        if isinstance(index, int):
            # x缩放的数据以及时间编码
            scaled_x = self.scaled_data[index:index + self.window]

            x_date_enc = self.date_enc_data[index:index + self.window]
            # y缩放的数据以及时间编码
            scaled_y = self.scaled_data[self.window + self.horizon - 1 +
                                        index:self.window + self.horizon - 1 + index + self.steps]
            y_date_enc = self.date_enc_data[self.window + self.horizon -
                                            1 + index:self.window + self.horizon - 1 + index + self.steps]
            # y真实数据
            y = self.dataset.data[self.window + self.horizon - 1 +
                                  index:self.window + self.horizon - 1 + index + self.steps]
            if self.cycle_index is not None:
                cycle_index = torch.tensor(self.cycle_index[index + self.window])
                return scaled_x, scaled_y, y, x_date_enc, y_date_enc, cycle_index
            return scaled_x, scaled_y, y, x_date_enc, y_date_enc, -1
        else:
            raise TypeError('Not surpported index type!!!')

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1


class SingleStepWrapper(Dataset):

    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        return self.dataset.data[index:index + self.window], self.dataset.data[self.window + self.horizon - 1 + index]

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1


class MultiStepWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3, steps: int = 2):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps

    def __getitem__(self, index):
        return self.dataset.data[index:index + self.window], self.dataset.data[
                                                             self.window + self.horizon - 1 + index:self.window + self.horizon - 1 + index + self.steps]

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1


class SingStepFlattenWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.dataset.data[index:index + self.window]
        y = self.dataset.data[self.window + self.horizon - 1 + index]
        return x.flatten(), y

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1


class MultiStepFlattenWrapper(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, window: int = 168, horizon: int = 3, steps: int = 2):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps

    def __getitem__(self, index):
        x = self.dataset.data[index:index + self.window]
        y = self.dataset.data[self.window + self.horizon - 1 +
                              index:self.window + self.horizon - 1 + index + self.steps]
        return x.flatten(), y.flatten()

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1
