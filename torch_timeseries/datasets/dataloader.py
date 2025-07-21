import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.stattools import acf
from torch_timeseries.data.scaler import Scaler
from torch_timeseries.datasets.dataset import (
    TimeSeriesDataset,
    TimeseriesSubset,
)
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet


class ChunkSequenceTimefeatureDataLoader:
    # 构造函数
    def __init__(
            self,
            # 参数中有数据集,缩放器,时间编码器默认为0,输入数据窗口大小,预测多少时间步长,一次预测的时间步长
            #     是否仅在训练集缩放,是否随机打乱,时间序列频率
            dataset: TimeSeriesDataset,
            scaler: Scaler,
            time_enc=0,
            window: int = 168,
            horizon: int = 3,
            steps: int = 2,
            scale_in_train=False,
            shuffle_train=True,
            freq=None,
            # 批次大小
            batch_size: int = 32,
            # 训练集比例
            # train_ratio: float = 0.7,
            train_ratio: float = 0.7,
            # 验证集比例
            val_ratio: float = 0.2,
            # 加载数据并行处理数
            num_worker: int = 0,
            # 测试集验证集的切分方式是否和训练集一样
            uniform_eval=True,
            isCycle=False
    ) -> None:
        """
        Split the dataset sequentially, and then randomly sample from each subset.

        :param dataset: the input dataset, must be of type datasets.Dataset
        :param train_ratio: the ratio of the training set
        :param test_ratio: the ratio of the testing set
        :param val_ratio: the ratio of the validation set
        :param uniform_eval: if True, the evalution will not be affected by input window length
        :param independent_scaler: whether to set independent scaler for train , val and test dataset,
                default: False, will have a global scaler for all data
                if set to True, scaler is fitted by differenct part of data
        """
        self.isCycle = isCycle

        # 数据集划分为三个,三个的比例赋值
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - self.train_ratio - self.val_ratio

        self.uniform_eval = uniform_eval

        # 三个的比例必须需要和为1
        assert (
                self.train_ratio + self.val_ratio + self.test_ratio == 1.0
        ), "Split ratio must sum up to 1.0"
        # 变量的赋值
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataset = dataset

        self.scaler = scaler
        self.window = window
        self.freq = freq
        self.time_enc = time_enc
        self.steps = steps
        self.horizon = horizon
        self.shuffle_train = shuffle_train
        self.scale_in_train = scale_in_train
        self.train_index = None
        self.val_index = None
        self.test_index = None
        # 调用load方法，实际上即为初始化是进行加载任务
        self._load()

    def _load(self):
        # 调用两个方法

        self._load_dataset()

        self._load_dataloader()

    def calculate_acf_period(self, dataset, max_lag):
        """
            计算时间序列的主周期，通过 ACF 识别周期。

            :param time_series: 输入的时间序列
            :param max_lag: 最大滞后数，用于计算 ACF
            :return: 返回主周期
            """
        time_series = np.mean(dataset, axis=1)
        acf_values = acf(time_series, nlags=max_lag, fft=True)

        # 寻找 ACF 的显著峰值位置
        peaks = np.where(acf_values[1:] < acf_values[:-1])[0]

        if len(peaks) > 0:
            # 返回第一个周期性峰值的位置作为主周期
            return peaks[0] + 1  # 加 1，因为 ACF 从滞后 0 开始
        else:
            return None

    def _load_dataset(self):
        """
        Return the splitted training, testing and validation dataloders

        :return: a tuple of train_dataloader, test_dataloader and val_dataloader
        """
        # fixed suquence dataset
        # 通过ACF方法求出数据集的cycle。
        if self.isCycle:
            self.cycle = self.calculate_acf_period(self.dataset.data, max_lag=1000)
            self.cycle_index = (np.arange(len(self.dataset.data)) % self.cycle)

        # 获取数据的索引范围
        indices = range(0, len(self.dataset))

        # 根据前面所设计的三种数据集的占比比例求出对应的数据集大小
        train_size = int(self.train_ratio * len(self.dataset))
        val_size = int(self.val_ratio * len(self.dataset))
        test_size = len(self.dataset) - val_size - train_size
        # 通过创建一个TimeseriesSubset对象,切分出训练集子集
        train_subset = TimeseriesSubset(self.dataset, indices[0:train_size])
        if self.isCycle:
            self.train_index = self.cycle_index[0:train_size]
        # 判断是否切分方式要一样
        if self.uniform_eval:
            # 如果要一样,需要调整以使窗口一致
            val_subset = TimeseriesSubset(  # self.window + self.horizon - 1
                self.dataset, indices[train_size - self.window - self.horizon + 1: (val_size + train_size)]
            )
            if self.isCycle:
                self.val_index = self.cycle_index[train_size - self.window - self.horizon + 1: (val_size + train_size)]
        else:
            # 不一样的话,则不需要调整
            val_subset = TimeseriesSubset(
                self.dataset, indices[train_size: (val_size + train_size)]
            )
            if self.isCycle:
                self.val_index = self.cycle_index[train_size: (val_size + train_size)]

        # 对测试集也是一样的处理方法
        if self.uniform_eval:
            test_subset = TimeseriesSubset(self.dataset, indices[-test_size - self.window - self.horizon + 1:])
            if self.isCycle:
                self.test_index = self.cycle_index[-test_size - self.window - self.horizon + 1:]
        else:
            test_subset = TimeseriesSubset(self.dataset, indices[-test_size:])
            if self.isCycle:
                self.test_index = self.cycle_index[-test_size - self.window - self.horizon + 1:]
        # 如果只对训练进行拟合,只在train_subset进行fit
        if self.scale_in_train:
            self.scaler.fit(train_subset)
        else:
            # 否则的话就对所有的数据进行拟合
            self.scaler.fit(self.dataset.data)

        # 构造三个集合的MultiStepTimeFeatureSet
        self.train_dataset = MultiStepTimeFeatureSet(
            train_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
            steps=self.steps,
            freq=self.freq,
            scaler_fit=False,
            cycleindex=self.train_index
        )
        # dataset
        # StandarScaler
        # 0
        # 96
        # 1
        # 96
        # "h"
        # False
        self.val_dataset = MultiStepTimeFeatureSet(
            val_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
            steps=self.steps,
            freq=self.freq,
            scaler_fit=False,
            cycleindex=self.val_index
        )
        self.test_dataset = MultiStepTimeFeatureSet(
            test_subset,
            scaler=self.scaler,
            time_enc=self.time_enc,
            window=self.window,
            horizon=self.horizon,
            steps=self.steps,
            freq=self.freq,
            scaler_fit=False,
            cycleindex=self.test_index
        )

    def _load_dataloader(self):
        # 三个集合的大小
        self.train_size = len(self.train_dataset)
        self.val_size = len(self.val_dataset)
        self.test_size = len(self.test_dataset)
        # RandomSampler 与 Dataloader generator都需要设置，否则还是无法复现
        # 创建三个集合对应的loader，有数据集，批次大小，是否打乱以及进程数
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_worker,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )


class DDPChunkSequenceTimefeatureDataLoader(ChunkSequenceTimefeatureDataLoader):
    def _load_dataloader(self):
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_worker,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )
