import codecs
import datetime
import hashlib
import json
import os
import random
import re
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
####
from typing import Dict, List, Type

import wandb
from prettytable import PrettyTable
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, MetricCollection, MeanAbsoluteError
from tqdm import tqdm
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.VaryFreq import VaryFreq
from torch_timeseries.datasets.CustomVaryFreq import CustomVaryFreq
from torch_timeseries.datasets.dataloader import (
    ChunkSequenceTimefeatureDataLoader,
)
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.metrics.masked_mape import MaskedMAPE
from torch_timeseries.nn.metric import R2, Corr, RMSE
from torch_timeseries.norm_experiments.Model import Model
from torch_timeseries.normalizations import *
from torch_timeseries.utils.early_stopping import EarlyStopping


# class Task(Enum):
#     SingleStepForecast : str = "single_step_forecast"
#     MultiStepForecast : str = "multi_steps_forecast"
#     Imputation : str = "imputation"
#     Classification : str = "classifation"
#     AbnomalyDetection : str = "abnormaly_detection"


def kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2):
    k = mu1.size(1)

    Sigma2_inv = torch.linalg.inv(Sigma2)

    tr_term = torch.einsum('bij,bjk->bi', Sigma2_inv, Sigma1)

    mu_diff = mu2 - mu1
    mu_term = torch.einsum('bi,bij,bj->b', mu_diff, Sigma2_inv, mu_diff)

    det_term = torch.log(torch.linalg.det(Sigma2) / torch.linalg.det(Sigma1))

    kl_div = 0.5 * (tr_term + mu_term - k + det_term)

    return kl_div.sum()


@dataclass
class ResultRelatedSettings:
    # 基类，设置了一系列的参数
    # 数据集类型
    dataset_type: str
    # 优化器算法类型，更新模型参数的算法
    optm_type: str = "Adam"
    # 模型类型
    model_type: str = ""
    # 缩放器类型
    scaler_type: str = "StandarScaler"
    # 损失函数类型 mse 均方误差
    loss_func_type: str = "mse"
    # 批次大小,默认32,每次参数更新的样本数量
    batch_size: int = 32
    # 学习率
    lr: float = 0.0003
    # L2权重衰减的λ，损失函数正则化，防止过拟合，Loss=OriginalLoss+λ∥w∥（L2范数）
    l2_weight_decay: float = 0.0005
    # 训练轮数
    epochs: int = 100
    # 一次预测时长，预测未来多少时间
    horizon: int = 3
    # 输入数据的窗口大小
    windows: int = 384
    # 训练要预测的时间步数
    pred_len: int = 1
    # 早停容忍度，早停机制，多少个没有提升就停止训练
    patience: int = 5
    # 避免梯度太大使得参数更新幅度太大，计算梯度的L2范数，超过该值，梯度就要进行缩放，最大L2范数即为该值
    max_grad_norm: float = 5.0
    # 是否使用逆变换损失
    invtrans_loss: bool = False
    # 归一化类型
    norm_type: str = ''
    # 归一化配置
    norm_config: dict = field(default_factory=lambda: {})


@dataclass
# 继承ResultRelatedSettings
class Settings(ResultRelatedSettings):
    # 加了一些参数
    # 数据存储位置
    data_path: str = "./data"
    # 运行设备
    device: str = "cuda:0"
    # 数据加载时使用的工作线程数
    num_worker: int = 20
    # 训练结果保存目录
    save_dir: str = "./results"
    # 根据当前时间生成每个实验的唯一标签
    experiment_label: str = str(int(time.time()))


def count_parameters(model, print_fun=print):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print_fun(table)
    print_fun(f"Total Trainable Params: {total_params}")
    return total_params


class NormExperiment(Settings):
    # 先调用该函数，
    # 以run_fan_wandb为例，project和name传入Norm
    def config_wandb(
            self,
            project: str,
            name: str,
    ):

        # TODO: add seeds config parameters
        # 字典key的转换
        def convert_dict(dictionary):
            # 创建一个字典
            converted_dict = {}
            # 实际上是把之前的字典改成一个新的字典，这个字典的key变为config.原来的key
            for key, value in dictionary.items():
                converted_dict[f"config.{key}"] = value
            return converted_dict

        # check wether this experiment had runned and reported on wandb
        # 创建一个wandb.Api的实例，以便于之后直接通过api变量调用wandb的接口
        api = wandb.Api()
        # 字典转换
        config_filter = convert_dict(self.result_related_config)
        # 根据前面的config_filter进行过滤找到wandb实例之下满足条件的实验运行
        runs = api.runs(path=project, filters=config_filter)

        try:
            # 如果获得的第一个实验的状态为完成或者正在运行中，说明已经有了该实验，不再需要去运行
            if runs[0].state == "finished" or runs[0].state == "running":
                print(
                    f"{self.model_type} {self.dataset_type} w{self.windows} w{self.horizon}  Experiment already reported, quiting...")
                # finished为True
                self.finished = True
                # 直接返回
                return
        except:
            pass

        # 如果不是上述两状态，初始化一个wandb运行
        run = wandb.init(
            # 离线模式
            mode='offline',
            # 该实验归入的wandb项目
            project=project,
            # 实验名称
            name=self.model_type + "," + self.dataset_type + "," + self.norm_type + ":" + str(self.windows) + " " + str(
                self.pred_len) + " " + str(self.horizon),
            # 实验的一些标签
            tags=[self.model_type, self.dataset_type, f"horizon-{self.horizon}", f"window-{self.windows}",
                  f"pred-{self.pred_len}", f"{self.norm_type}"],
        )
        # 把所有属性都作为配置上传给wandb
        wandb.config.update(asdict(self))
        # 标志实验与wandb集成
        self.wandb = True
        print(f"using wandb , running in config: {asdict(self)}")
        return self

    def wandb_sweep(
            self,
            project,
            name,
    ):
        run = wandb.init(
            project='BiSTGNN'
        )
        wandb.config.update(asdict(self))
        self.wandb = True
        print(f"using wandb , running in config: {asdict(self)}")
        return self

    # 判断有没有wandb这个属性，即判断是否与wandb集成好
    def _use_wandb(self):
        return hasattr(self, "wandb")

    # 第一次使用时传入参数为f"run : {self.current_run} in seed: {seed}"
    def _run_print(self, *args, **kwargs):

        # 最后获得一个带有时间戳的字符串，例如[2024-11-05 08:00:00]
        time = '[' + str(datetime.datetime.utcnow() +
                         datetime.timedelta(hours=8))[:19] + '] -'
        # 直接将传入的 * args打印
        print(*args, **kwargs)
        # 如果运行的配置已经配置好了
        if hasattr(self, "run_setuped") and getattr(self, "run_setuped") is True:
            # 在结果存储目录下output.log追加 当前时间戳以及传入参数
            with open(os.path.join(self.run_save_dir, 'output.log'), 'a+') as f:
                print(time, *args, flush=True, file=f)

    def config_wandb_verbose(
            self,
            project: str,
            name: str,
            tags: List[str],
            notes: str,
    ):
        run = wandb.init(
            project=project,
            name=name,
            notes=notes,
            tags=tags,
        )
        wandb.config.update(asdict(self))
        print(f"using wandb , running in config: {asdict(self)}")
        self.wandb = True
        return self

    # setup方法会调用下面三个初始化方法
    # 初始化数据加载器
    def _init_data_loader(self):
        # 创建一个dataset参数，该参数为TimeSeriesDataset类型
        # 根据dataset_type调用_parse_type方法返回对应的类，然后传入root参数实例化一个对象然后赋给dataset
        # 会自动调用该类的init方法，会自动下载对应的csv文件并读取数据
        self.dataset: TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        # 默认StandarScaler，创建对应的类实例赋给scaler
        self.scaler = self._parse_type(self.scaler_type)()

        # 构建一个ChunkSequenceTimefeatureDataLoader对象
        if self.model_type == "CycleNet":
            self.dataloader = ChunkSequenceTimefeatureDataLoader(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=False,
                shuffle_train=True,
                freq="h",
                batch_size=self.batch_size,
                train_ratio=0.7,
                val_ratio=0.2,  # 0.1
                num_worker=self.num_worker,
                isCycle=True
            )
            if self._use_wandb():
                wandb.config.update({"cycle": self.dataloader.cycle})

        else:
            self.dataloader = ChunkSequenceTimefeatureDataLoader(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=False,
                shuffle_train=True,
                freq="h",
                batch_size=self.batch_size,
                train_ratio=0.7,
                val_ratio=0.2,  # 0.1
                num_worker=self.num_worker,
            )

        # 把三个loader赋给本类的变量
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        # 三个集合的长度
        self.train_steps = self.dataloader.train_size
        self.val_steps = self.dataloader.val_size
        self.test_steps = self.dataloader.test_size
        # 打印出来
        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")

    # 损失函数初始化
    def _init_loss_func(self):
        # 创建了一个损失函数的字典
        loss_func_map = {"mse": MSELoss, "l1": L1Loss}
        # 根据loss_func_type找到对应的损失函数对象，创建实例并赋给loss_func
        self.loss_func = loss_func_map[self.loss_func_type]()

    # 初始化指标
    def _init_metrics(self):
        # 如果预测长度为1
        if self.pred_len == 1:
            # 构建一个MetricCollection的对象，其中有许多指标
            self.metrics = MetricCollection(
                # 包含平均R2指标，加权R2，MSE，相关性，MAE
                metrics={
                    "r2": R2(self.dataset.num_features, multioutput="uniform_average"),
                    "r2_weighted": R2(
                        self.dataset.num_features, multioutput="variance_weighted"
                    ),
                    "mse": MeanSquaredError(),
                    "corr": Corr(),
                    "mae": MeanAbsoluteError(),
                }
            )
            # 如果预测长度大于1
        elif self.pred_len > 1:
            # 另一组指标， mse，mae，mape，rmse
            self.metrics = MetricCollection(
                metrics={
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError(),
                    "mape": MaskedMAPE(null_val=0),
                    # "mape": MeanAbsolutePercentageError(),
                    'rmse': RMSE(),
                }
            )
        # 把对应的评估指标对应到设置的device之上
        self.metrics.to(self.device)

    # 返回本对象的配置的字典，但会进行一些处理，过滤一些与结果无关的键值对
    @property
    def result_related_config(self):
        # 转成字典
        ident = asdict(self)
        # 要去掉的键值对的key值
        keys_to_remove = [
            "data_path",
            "device",
            "num_worker",
            "save_dir",
            "experiment_label",
        ]
        # 去掉上面key对应的键值对
        for key in keys_to_remove:
            if key in ident:
                del ident[key]
        # 返回过滤过之后的字典
        return ident

    # 给每一次运行生成唯一标识
    def _run_identifier(self, seed) -> str:
        ident = self.result_related_config
        ident["seed"] = seed
        # only influence the evluation result, not included here
        # ident['invtrans_loss'] = False

        ident_md5 = hashlib.md5(
            json.dumps(ident, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return str(ident_md5)

    # def _init_vmd_sep_optimizer(self):
    #     # 设置归一化模型和预测模型的优化器
    #     base_params = [p for p in self.model.nm.parameters() if p not in set(self.model.nm.vmd.parameters())]
    #
    #     self.n_model_optim = Adam(
    #         base_params, lr=self.lr, weight_decay=self.l2_weight_decay
    #     )
    #     self.n_vmd_model_optim = Adam(
    #         self.model.nm.vmd.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
    #     )
    #     self.f_model_optim = Adam(
    #         self.model.fm.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
    #     )
    #
    #     # 设置预测模型的学习率调度器,控制学习率变化
    #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         self.f_model_optim, T_max=self.epochs
    #     )

    # 初始化两个优化器和一个学习率调度器
    def _init_sep_optimizer(self):
        # 设置归一化模型和预测模型的优化器
        self.n_model_optim = Adam(
            self.model.nm.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        self.f_model_optim = Adam(
            self.model.fm.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )

        # 设置预测模型的学习率调度器,控制学习率变化
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.f_model_optim, T_max=self.epochs
        )

    # def _init_vmd_optimizer(self):
    #     # 对两个模型统一优化
    #     base_params = [p for p in self.model.parameters() if p not in set(self.model.nm.vmd.parameters())]
    #
    #     self.n_vmd_model_optim = Adam(
    #         self.model.nm.vmd.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
    #     )
    #     self.model_optim = Adam(
    #         base_params, lr=self.lr, weight_decay=self.l2_weight_decay
    #     )
    #     # 对两个模型统一学习率调度
    #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         self.model_optim, T_max=self.epochs
    #     )

    def _init_optimizer(self):
        # 对两个模型统一优化
        self.model_optim = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        # 对两个模型统一学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optim, T_max=self.epochs
        )

    def _init_n_model(self):
        # 先打印创建对象时，所要使用的归一化方式以及对应的config配置
        # 当前例子norm_type为FAN，config为{freq_topk:4}
        print(f"using {self.norm_type} as normalization model., config: {self.norm_config}")
        # 获得对应的归一化类赋给Ty
        Ty = self._parse_type(self.norm_type)
        # 判断norm的类型，根据对应的类型去创建对应的归一化方法的类实例
        #  先以FAN为例
        if self.norm_type == 'RevIN':
            self.n_model: torch.nn.Module = Ty(self.dataset.num_features, True, **self.norm_config)
        elif self.norm_type == 'SAN':
            self.n_model: torch.nn.Module = Ty(self.windows, self.pred_len, 12, self.dataset.num_features,
                                               **self.norm_config)
        elif self.norm_type == 'DishTS':
            self.n_model: torch.nn.Module = Ty(self.dataset.num_features, self.windows, **self.norm_config)
        elif self.norm_type == 'No':
            self.n_model: torch.nn.Module = No()
        elif self.norm_type == 'VMD_Change':
            self.n_model: torch.nn.Module = Ty(self.windows, self.pred_len, self.dataset.num_features)
        # FAN则会走这一部分
        else:
            # 生成一个FAN对象赋给n_model    当前例子      96     96 168 336 720各一次     数据集特征数 每个数据集自己定义好了，norm配置{freq_topk:4}
            self.n_model: torch.nn.Module = Ty(self.windows, self.pred_len, self.dataset.num_features,
                                               **self.norm_config)

        # 也是将归一化模型移到对应设备
        self.n_model = self.n_model.to(self.device)

    # 子类会重写该方法
    def _init_f_model(self) -> torch.nn.Module:
        self.f_model = None
        raise NotImplementedError()

    def _init_model(self):
        # self.model = self._parse_type(self.model_type)().to(self.device)
        # 把归一化模型和预测模型整合到一起
        #     以FEDformer为例，  FEDformer       该预测模型      归一化模型
        # 并将该整合模型安置到device上
        self.model = Model(self.model_type, self.f_model, self.n_model).to(self.device)

    def is_sep_loss(self):
        # 看norm_config的seploss是否为true
        print("seploss", "seploss" in self.norm_config and self.norm_config['seploss'] == True)
        # 如果有seploss且为true返回true，否则返回false
        return "seploss" in self.norm_config and self.norm_config['seploss'] == True

    def is_vmd_sep_loss(self):
        # 看norm_config的seploss是否为true
        # print("norm_sep_loss", "norm_sep_loss" in self.norm_config and self.norm_config['norm_sep_loss'] == True)
        # 如果有seploss且为true返回true，否则返回false
        return "norm_sep_loss" in self.norm_config and self.norm_config['norm_sep_loss'] == True

    # 初始化
    def _setup(self):

        # init data loader
        self._init_data_loader()

        # init metrics
        self._init_metrics()

        # init loss function based on given loss func type
        self._init_loss_func()

        # 初始化训练轮数
        self.current_epochs = 0
        # 初始化运行计数
        self.current_run = 0
        # 已然setuped
        self.setuped = True

    # 训练任务设置
    def _setup_run(self, seed):
        # setup experiment  only once
        # 如果没有被设置过
        if not hasattr(self, "setuped"):
            # 调用setup进行设置
            self._setup()

        # setup torch and numpy random seed
        # 确保实验可复现性
        self.reproducible(seed)
        # init model, optimizer and loss function

        # 初始化模型
        # 初始化归一化模型
        self._init_n_model()

        # 初始化预测模型
        self._init_f_model()

        # 初始化模型
        self._init_model()
        # 如果config设置seploss,是否分离损失
        if self.is_sep_loss():
            self._init_sep_optimizer()
        else:
            self._init_optimizer()

        # 设置当前epoch轮数为0,这里不带s
        self.current_epoch = 0
        # 设置运行结果的存储目录
        self.run_save_dir = os.path.join(
            self.save_dir,
            "runs",
            self.model_type,
            self.dataset_type,
            f"w{self.windows}h{self.horizon}s{self.pred_len}",
            # 运行的唯一标识
            self._run_identifier(seed),
        )
        # 设置最好权重文件存储目录
        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "best_model.pth"
        )

        # 当前检查点文件存储目录
        self.run_checkpoint_filepath = os.path.join(
            self.run_save_dir, "run_checkpoint.pth"
        )

        # 早停对象
        # 传入patience: int = 5，verbose为true，还传入最好权重文件存储的目录

        self.early_stopper = EarlyStopping(
            self.patience, verbose=True, path=self.best_checkpoint_filepath
        )

        # 设置run_setuped属性为True
        self.run_setuped = True

    # 根据dataset_type返回对应的类
    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            print(str_or_type)
            # 返回该字符串对应的类
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            # 都不是报错
            raise RuntimeError(f"{str_or_type} should be string or type")

    def _save(self, seed=0):
        self.checkpoint_path = os.path.join(
            self.save_dir, f"{self.model_type}/{self.dataset_type}"
        )
        self.checkpoint_filepath = os.path.join(
            self.checkpoint_path, f"{self.experiment_label}.pth"
        )
        # 检查目录是否存在
        if not os.path.exists(self.checkpoint_path):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.checkpoint_path)
            print(f"Directory '{self.checkpoint_path}' created successfully.")

        self.app_state = {
            "model": self.model,
            # "n_model": self.n_model,
            "optimizer": self.model_optim,
            # "norm_model_optim": self.norm_model_optim,

        }

        self.app_state.update(asdict(self))

        # now only save the newest state
        torch.save(self.app_state, f"{self.checkpoint_filepath}")

    def get_run_state(self):
        # 运行状态
        if self.is_sep_loss():
            run_state = {
                # "n_model": self.n_model.state_dict(),
                # "norm_model_optim": self.norm_model_optim.state_dict(),
                "model": self.model.state_dict(),
                "current_epoch": self.current_epoch,
                "n_optimizer": self.n_model_optim.state_dict(),
                "f_optimizer": self.f_model_optim.state_dict(),
                "rng_state": torch.get_rng_state(),
                "early_stopping": self.early_stopper.get_state(),
            }

        else:
            run_state = {
                # "n_model": self.n_model.state_dict(),
                # "norm_model_optim": self.norm_model_optim.state_dict(),
                "model": self.model.state_dict(),
                "current_epoch": self.current_epoch,
                "optimizer": self.model_optim.state_dict(),
                "rng_state": torch.get_rng_state(),
                "early_stopping": self.early_stopper.get_state(),
            }
        return run_state

    def _save_run_check_point(self, seed):
        # 检查目录是否存在
        if not os.path.exists(self.run_save_dir):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = self.get_run_state()
        # if not isinstance(self.n_model, No):
        #     self.run_state['n_model'] =  self.n_model.state_dict()
        # self.run_state['norm_model_optim'] =  self.norm_model_optim.state_dict()

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")

    def reproducible(self, seed):
        # for reproducibility
        # torch.set_default_dtype(torch.float32)
        # 查看当前torch数据类型
        print("torch.get_default_dtype()", torch.get_default_dtype())
        # 设置数据类型为FloatTensor
        torch.set_default_tensor_type(torch.FloatTensor)
        # 就是为了可重现实验结果
        # 根据传入的seed设置pytorch的随机种子
        torch.manual_seed(seed)
        # 设置pythonhash的随机种子
        os.environ["PYTHONHASHSEED"] = str(seed)
        # 设置GPU随机种子
        torch.cuda.manual_seed_all(seed)
        # 设置numpy和random的随机种子
        np.random.seed(seed)
        random.seed(seed)
        # 设置cuDNN，确保可复现性
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determinstic = True

    def _process_input(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, dec_inp=None, dec_input_date=None):
        # inputs:
        # batch_x:  (B, T, N)
        # batch_y:  (B, Steps,T)
        # batch_x_date_enc:  (B, T, N)
        # batch_y_date_enc:  (B, T, Steps)
        # outputs:
        # pred: (B, O, N)
        raise NotImplementedError()
        # batch_x = batch_x.transpose(1,2) # (B, N, T)
        # batch_x_date_enc = batch_x_date_enc.transpose(1,2) # (B, N, T)
        # pred = self.model(batch_x) # (B, O, N)
        # pred = pred.transpose(1,2) # (B, O, N)
        # return pred

    # 子类会重写该方法
    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x:  (B, T, N)
        # batch_y:  (B, Steps,T)
        # batch_x_date_enc:  (B, T, N)
        # batch_y_date_enc:  (B, T, Steps)

        # outputs:
        # pred: (B, O, N)
        # label:  (B,O,N)
        raise NotImplementedError()
        # label_len = 48
        # dec_inp_pred = torch.zeros(
        #     [batch_x.size(0), self.pred_len, self.dataset.num_features]
        # ).to(self.device)
        # dec_inp_label = batch_x[:, label_len :, :].to(self.device)

        # dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        # dec_inp_date_enc = torch.cat(
        #     [batch_x_date_enc[:, label_len :, :], batch_y_date_enc], dim=1
        # )

        # pred = self._process_input(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, dec_inp, dec_inp_date_enc)

        # return pred, batch_y # (B, O, N), (B, O, N)

    def _ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def visualize_and_compare_vmd(self, model_instance, test_loader, device, epoch_num,
                                  dataset_name="CustomVaryFreq", num_samples_to_plot=1):
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import torch
        import torch.nn.functional as F
        import scipy.signal  # For spectrogram (though we're moving to FFT)
        from vmdpy import VMD
        model_instance.eval()  # Set model to evaluation mode

        # 创建不同的输出目录
        # Create distinct output directories
        output_base_dir = "vmd_comparison_plots"
        d_vmd_output_dir = os.path.join(output_base_dir, "d_vmd_results")
        pyvmd_output_dir = os.path.join(output_base_dir, "pyvmd_results")
        self._ensure_dir(d_vmd_output_dir)
        self._ensure_dir(pyvmd_output_dir)

        # *** 明确指定要可视化的样本和特征索引 ***
        # 您可以根据需要修改这些值
        sample_index_to_visualize = 0  # 批次中的第几个样本 (0-indexed)
        feature_index_to_visualize = 0  # 该样本中的第几个特征 (0-indexed)

        with torch.no_grad():
            # 我们只获取 test_loader 中的第一个批次进行可视化
            # 因为只可视化一个样本的一个特征，所以只取第一个批次就足够了
            batch_x, batch_y, _, _, _, _ = next(iter(test_loader))

            # 将整个批次移动到设备上
            batch_x_on_device = batch_x.to(device, dtype=torch.float32)  # Shape: [B_current, T, N_total]
            current_batch_size = batch_x_on_device.shape[0]

            # 检查指定的样本和特征索引是否有效
            if sample_index_to_visualize >= current_batch_size:
                print(f"错误: 指定的样本索引 {sample_index_to_visualize} 超出批次大小 ({current_batch_size})。请选择一个有效的样本索引。")
                return
            if feature_index_to_visualize >= batch_x_on_device.shape[2]:
                print(
                    f"错误: 指定的特征索引 {feature_index_to_visualize} 超出数据特征范围 ({batch_x_on_device.shape[2]} 个特征)。请选择一个有效的特征索引。")
                return

            # --- 获取 Differentiable VMD 学习到的 k 和 alpha ---
            d_vmd_module = model_instance.vmd.vmd  # 访问 DifferentiableVMD 实例

            # alpha 是一个单独的学习参数，直接使用它
            learned_alpha = F.softplus(d_vmd_module.alpha_raw).item()

            print(f"\n--- 正在处理批次 (大小: {current_batch_size}) ---")
            print(f"PyVMD 的学习到的 Alpha: {learned_alpha:.2f}")
            print(f"--- 仅可视化样本索引: {sample_index_to_visualize}, 特征索引: {feature_index_to_visualize} ---")

            # 对整个批次进行一次 Differentiable VMD 分解
            # 这将为批次中的每个样本返回 k_cont 和 IMF
            imfs_d_vmd_batch_full, x_recon_d_vmd_batch_full, _, _, _, _, _, _, k_cont_batch = d_vmd_module(
                batch_x_on_device
            )
            # imfs_d_vmd_batch_full shape: [B_current, K_learned, T, N_total]
            # k_cont_batch shape: [B_current]

            # --- 提取指定样本和特征的数据进行可视化 ---

            # 提取该特定样本的学习到的 k
            learned_k_float_for_sample = k_cont_batch[sample_index_to_visualize].item()
            learned_k_int_for_sample = int(round(learned_k_float_for_sample))
            learned_k_int_for_sample = max(1, learned_k_int_for_sample)  # 确保 PyVMD 的 k 至少为 1

            print(
                f"  学习到的 K (浮点数) for chosen sample: {learned_k_float_for_sample:.2f}, PyVMD 使用的 K (整数): {learned_k_int_for_sample}")

            # 提取指定样本指定特征的原始信号
            current_original_signal_np = batch_x_on_device[sample_index_to_visualize, :,
                                         feature_index_to_visualize].cpu().numpy()  # [T]

            # 提取 Differentiable VMD 分解结果中对应样本和指定特征的 IMF
            imfs_d_vmd_np = imfs_d_vmd_batch_full[sample_index_to_visualize, :, :,
                            feature_index_to_visualize].cpu().numpy()  # 从 [B, K, T, N] 变为 [K, T]

            print(f"正在处理指定特征 {feature_index_to_visualize + 1}...")

            # --- 2. 使用原始 PyVMD 处理 ---
            print(f"正在使用原始 PyVMD 处理指定特征 {feature_index_to_visualize + 1}...")

            # PyVMD 使用从 D-VMD 的 k_cont 推导出的整数 k
            imfs_pyvmd, u_hat, omega = VMD(current_original_signal_np, learned_alpha, 0., learned_k_int_for_sample, 0,
                                           1, 1e-7)
            # imfs_pyvmd 形状: [K, T]

            # --- 绘图和比较 ---

            # Differentiable VMD 的绘图
            self.plot_vmd_results(current_original_signal_np, imfs_d_vmd_np,
                                  learned_k_float_for_sample, learned_alpha,
                                  f"Differentiable VMD (K={learned_k_float_for_sample:.2f}, A={learned_alpha:.2f})",
                                  os.path.join(d_vmd_output_dir,
                                               f'epoch_{epoch_num}_sample_{sample_index_to_visualize}_feature_{feature_index_to_visualize + 1}'))

            # 原始 PyVMD 的绘图
            self.plot_vmd_results(current_original_signal_np, imfs_pyvmd,
                                  learned_k_int_for_sample, learned_alpha,
                                  f"Original PyVMD (K={learned_k_int_for_sample}, A={learned_alpha:.2f})",
                                  os.path.join(pyvmd_output_dir,
                                               f'epoch_{epoch_num}_sample_{sample_index_to_visualize}_feature_{feature_index_to_visualize + 1}'))

    def plot_vmd_results(self, original_signal, imfs,
                         k_val=None, alpha_val=None, title_prefix="", output_path_base=""):
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import torch
        import torch.nn.functional as F
        import scipy.signal  # For spectrogram (though we're moving to FFT)
        from vmdpy import VMD

        # Ensure num_imfs_to_plot doesn't exceed available IMFs
        if k_val is not None:  # Use integer k_val if available
            num_imfs_to_plot = min(int(k_val), imfs.shape[0])
        else:  # Fallback, plot all if k_val isn't explicitly an integer count
            num_imfs_to_plot = imfs.shape[0]

        T = imfs.shape[1]  # Time steps

        # Reconstruct signal from *all* provided IMFs for accurate reconstruction comparison
        # (even if we only plot a subset, the reconstruction should ideally use all components that have energy)
        reconstructed_signal = np.sum(imfs, axis=0)

        # --- Plot 1: Original Signal, Reconstructed Signal, and Individual IMFs ---
        num_rows_time_series = 2 + num_imfs_to_plot  # Original + Reconstructed + Selected IMFs

        fig, axes = plt.subplots(num_rows_time_series, 1, figsize=(15, 2.5 * num_rows_time_series), sharex=True)

        current_row = 0
        axes[current_row].plot(original_signal, label='Original Signal', color='blue', linewidth=1.5)
        axes[current_row].set_title(f'{title_prefix} - Original Signal Time Series')
        axes[current_row].legend()
        axes[current_row].grid(True)
        current_row += 1

        axes[current_row].plot(reconstructed_signal, label='Reconstructed Signal', color='green', linestyle='--',
                               linewidth=1.5)
        axes[current_row].set_title(f'Reconstructed Signal Time Series')
        axes[current_row].legend()
        axes[current_row].grid(True)
        current_row += 1

        # Plot only the relevant IMFs based on num_imfs_to_plot
        for i in range(num_imfs_to_plot):
            axes[current_row].plot(imfs[i, :], label=f'IMF {i + 1}', color='orange', alpha=0.8)
            axes[current_row].set_title(f'IMF {i + 1} Time Series')
            axes[current_row].legend()
            axes[current_row].grid(True)
            current_row += 1

        plt.xlabel('Time Step')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(f'{output_path_base}_imfs_time_series{self.current_epoch}.png', dpi=300)
        plt.close()

        # --- Plot 2: FFT Amplitude Spectrum of Original and each IMF ---
        num_rows_fft = 1 + num_imfs_to_plot  # Original + Selected IMFs
        fig_fft, axes_fft = plt.subplots(num_rows_fft, 1, figsize=(10, 3 * num_rows_fft), sharex=True)
        fs = 1  # Assuming sampling frequency of 1 unit per time step

        # FFT of Original Signal
        yf_orig = np.fft.fft(original_signal)
        xf_orig = np.fft.fftfreq(T, 1 / fs)[:T // 2]  # Frequencies for single-sided spectrum

        axes_fft[0].plot(xf_orig, 2.0 / T * np.abs(yf_orig[0:T // 2]), color='blue')
        axes_fft[0].set_title(f'{title_prefix} - Original Signal FFT Spectrum')
        axes_fft[0].set_ylabel('Amplitude')
        axes_fft[0].grid(True)
        axes_fft[0].set_xlim([0, fs / 2])
        axes_fft[0].set_ylim(bottom=0)

        # FFT of relevant IMFs
        for i in range(num_imfs_to_plot):
            yf_imf = np.fft.fft(imfs[i, :])
            axes_fft[i + 1].plot(xf_orig, 2.0 / T * np.abs(yf_imf[0:T // 2]), color='orange', alpha=0.8)
            axes_fft[i + 1].set_title(f'IMF {i + 1} FFT Spectrum')
            axes_fft[i + 1].set_ylabel('Amplitude')
            axes_fft[i + 1].grid(True)
            axes_fft[i + 1].set_xlim([0, fs / 2])
            axes_fft[i + 1].set_ylim(bottom=0)

        plt.xlabel('Frequency (Hz)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(f'{output_path_base}_fft_spectrums{self.current_epoch}.png', dpi=300)
        plt.close()

    def _evaluate(self, dataloader):
        # 设置模型为评估模式
        self.model.eval()
        # self.n_model.eval()
        # 重置之前的评估指标结果
        self.metrics.reset()
        # 获取所要评估的数据集的长度
        length = 0
        if dataloader is self.train_loader:
            length = self.dataloader.train_size
        elif dataloader is self.val_loader:
            length = self.dataloader.val_size
        elif dataloader is self.test_loader:
            length = self.dataloader.test_size

        # y_truths = []
        # y_preds = []
        # 评估不需要梯度
        with torch.no_grad():
            # if self.model_type == "VMD_DLinear" and self.visual:
            #     print("\n--- Final Test Set Visualization and VMD Comparison ---")
            #     self.visualize_and_compare_vmd(
            #         model_instance=self.model.fm,
            #         test_loader=self.test_loader,
            #         device=self.device,
            #         epoch_num="final",  # Use a string like "final" or a large epoch number for final test
            #         dataset_name="CustomVaryFreq",
            #     )
            with tqdm(total=length, position=0, leave=True) as progress_bar:
                # 也是遍历获得每一个批次的缩放过的x、y、原始y、以及x和y对应的时间编码
                for batch_x, batch_y, batch_origin_y, batch_x_date_enc, batch_y_date_enc, batch_cycle in dataloader:

                    batch_size = batch_x.size(0)
                    # 移到设备上
                    batch_x = batch_x.to(self.device, dtype=torch.float32)
                    batch_y = batch_y.to(self.device, dtype=torch.float32)
                    batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                    batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                    if torch.all(batch_cycle != -1):
                        batch_cycle = batch_cycle.int().to(self.device)

                    start = time.time()
                    # 可视化效果

                    if torch.all(batch_cycle != -1):
                        preds, truths = self.process_cycle(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc,
                                                           batch_cycle)
                    else:

                        preds, truths = self._process_batch(
                            batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                        )
                    # if self.model_type == "VMD_DLinear" and isinstance(self.dataset, CustomVaryFreq) and self.visual:

                    # import matplotlib.pyplot as plt
                    # import os
                    # # 选择第一个样本和第一个特征维度进行绘图
                    # sample_idx = 0  # 批次中的第0个样本
                    # feature_idx = 0  # 样本的第0个特征维度
                    #
                    # # 确保 'visual' 目录存在
                    # visual_dir = 'visual'
                    # os.makedirs(visual_dir, exist_ok=True)
                    #
                    # # --- 获取要绘制的数据 ---
                    # # 将 Tensor 移到 CPU 并转换为 NumPy 数组
                    # input_x_series = batch_x[sample_idx, :, feature_idx].detach().cpu().numpy()
                    # true_y_series = batch_origin_y[sample_idx, :, feature_idx].detach().cpu().numpy()
                    #
                    # # VMD 分解输入 X
                    # # imfs_x 形状预期是 (B, num_imfs, L, D)
                    # imfs_x, x_recon, _, _, _, _, _, _, _ = self.model.fm.vmd.vmd(batch_x)
                    #
                    # imfs_x_shape = imfs_x.shape  # 获取 imfs_x 的完整形状，特别是 IMF 数量
                    # num_imfs_total = imfs_x_shape[1]  # 获取 VMD 分解出的总 IMF 数量
                    #
                    # # 计算 IMF 能量占比并筛选主要部分 (来自输入 X)
                    # total_energy_x = batch_x.pow(2).sum(dim=[1, 2])  # 针对每个样本的能量
                    # imfs_energy_x = (imfs_x ** 2).sum(dim=[2, 3])  # [B, k_used] 每个模态在每个样本的能量
                    #
                    # # 为了保持批次维度，我们独立处理每个样本的能量筛选
                    # # 这里简单起见，我们对第一个样本进行处理
                    # sample_energy_x = total_energy_x[sample_idx]
                    # sample_imfs_energy_x = imfs_energy_x[sample_idx]  # [k_used]
                    #
                    # energy_ratio_x = sample_imfs_energy_x / (sample_energy_x + 1e-8)
                    #
                    # # 找出能量占比 > 5% 的模态索引 (针对当前样本)
                    # keep_indices_x = torch.nonzero(energy_ratio_x > 0.05).squeeze(-1)
                    # if keep_indices_x.numel() == 0:
                    #     # 如果没有高于阈值的，就取能量最大的一个
                    #     keep_indices_x = torch.tensor([energy_ratio_x.argmax()], device=self.device)
                    #
                    # # keep_indices_x = torch.arange(num_imfs_total, device=self.device)  # 生成从0到num_imfs_total-1的索引
                    # # 提取 VMD 主要部分和剩余部分 (来自输入 X)
                    # main_imfs_x_sample = imfs_x[sample_idx, keep_indices_x, :,
                    #                      feature_idx].detach().cpu().numpy()  # [k_filtered, L]
                    # main_part_x = main_imfs_x_sample.sum(axis=0)  # 形状 (L,)
                    #
                    # # 剩余部分 = 原始信号 - 主要部分
                    # # 这里使用 VMD 重构信号作为基准可能更准确，但根据你的代码，似乎是基于原始输入X
                    # # 如果你的 VMD 模块有返回完整的重构信号（x_recon），用 x_recon - main_part_x 更精确
                    # # 这里为了简单，我们用原始输入减去主成分来近似剩余部分
                    # residual_part_x = input_x_series - main_part_x  # 形状 (L,)
                    #
                    # # 预测的主要部分（由模型给出）
                    # # 假设 pred_main_signal 形状为 (B, H, D) 且已经设置
                    # predicted_main_signal = self.model.fm.vmd.pred_main_signal[sample_idx, :,
                    #                         feature_idx].detach().cpu().numpy()
                    #
                    # predicted_residual_signal_series = self.model.fm.vmd.pred_residual[sample_idx, :,
                    #                                    feature_idx].detach().cpu().numpy()
                    # # # --- 绘图 ---
                    # # # 计算子图数量：1(输入) + 1(真实值) + num_imfs_x + 1(主要部分) + 1(剩余部分)
                    # # num_imfs_x = main_imfs_x_sample.shape[0]  # 实际保留的 IMF 数量
                    # # total_plots = 2 + num_imfs_x + 2+1  # 输入、真实值、IMF、主要部分、剩余部分
                    # #
                    # # plt.figure(figsize=(15, 3 * total_plots))
                    # #
                    # # plot_idx = 1
                    # #
                    # # # 1. 绘制输入 X (原始信号)
                    # # plt.subplot(total_plots, 1, plot_idx)
                    # # plt.plot(input_x_series, label=f'Original Input X (L={len(input_x_series)})', color='blue')
                    # # plt.title(f'Sample {sample_idx}, Feature {feature_idx}: Original Input X')
                    # # plt.legend()
                    # # plt.grid(True)
                    # # plot_idx += 1
                    # #
                    # # # 2. 绘制真实值 Y (原始信号)
                    # # plt.subplot(total_plots, 1, plot_idx)
                    # # plt.plot(true_y_series, label=f'True Y (H={len(true_y_series)})', color='green')
                    # # plt.title(f'Sample {sample_idx}, Feature {feature_idx}: True Y Values')
                    # # plt.legend()
                    # # plt.grid(True)
                    # # plot_idx += 1
                    # #
                    # # # 3. 绘制 VMD 分解的各个 IMF (来自输入 X)
                    # # for i_imf in range(num_imfs_x):
                    # #     plt.subplot(total_plots, 1, plot_idx)
                    # #     plt.plot(main_imfs_x_sample[i_imf, :], label=f'IMF {i_imf + 1} (from Input X)',
                    # #              color='purple')
                    # #     plt.title(
                    # #         f'Sample {sample_idx}, Feature {feature_idx}: Decomposed IMF {i_imf + 1} from Input X')
                    # #     plt.legend()
                    # #     plt.grid(True)
                    # #     plot_idx += 1
                    # #
                    # # # 4. 绘制从输入 X 中提取的 VMD 主要部分
                    # # plt.subplot(total_plots, 1, plot_idx)
                    # # plt.plot(main_part_x, label=f'VMD Main Part (from Input X)', color='red')
                    # # plt.title(f'Sample {sample_idx}, Feature {feature_idx}: VMD Main Part from Input X')
                    # # plt.legend()
                    # # plt.grid(True)
                    # # plot_idx += 1
                    # #
                    # # # 5. 绘制从输入 X 中提取的 VMD 剩余部分
                    # # plt.subplot(total_plots, 1, plot_idx)
                    # # plt.plot(residual_part_x, label=f'VMD Residual Part (from Input X)', color='orange')
                    # # plt.title(f'Sample {sample_idx}, Feature {feature_idx}: VMD Residual Part from Input X')
                    # # plt.legend()
                    # # plt.grid(True)
                    # # plot_idx += 1
                    # #
                    # # # 6. 绘制预测的主要部分
                    # # if self.model.fm.vmd.pred_main_signal is not None:
                    # #     plt.subplot(total_plots, 1, plot_idx)
                    # #     plt.plot(predicted_main_signal,
                    # #              label=f'Predicted Main Signal (H={len(predicted_main_signal)})', color='darkcyan',
                    # #              linestyle='--')
                    # #     plt.title(f'Sample {sample_idx}, Feature {feature_idx}: Predicted Main Signal')
                    # #     plt.legend()
                    # #     plt.grid(True)
                    # #     plot_idx += 1
                    # # else:
                    # #     print(
                    # #         "Warning: self.model.fm.vmd.pred_main_signal is None. Cannot plot predicted main signal.")
                    # #
                    # # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
                    # #
                    # # # 保存图像
                    # # save_file_name = f'{self.model_type}_{self.dataset_type}_{self.pred_len}_sample{sample_idx}_feature{feature_idx}_vmd_decomposition.png'
                    # # save_path = os.path.join(visual_dir, save_file_name)
                    # # plt.savefig(save_path, dpi=300)
                    # # plt.close()  # 关闭图形，释放内存
                    # # --- 定义所有要绘制的线条及其颜色和名称 ---
                    # lines_to_plot = [
                    #     {'data': input_x_series, 'color': 'blue', 'linestyle': '-', 'name': 'Original_Input_X'},
                    #     {'data': true_y_series, 'color': 'green', 'linestyle': '-', 'name': 'True_Y'},
                    # ]
                    #
                    # # 添加所有的 IMF
                    # for i_imf in range(main_imfs_x_sample.shape[0]):
                    #     lines_to_plot.append({
                    #         'data': main_imfs_x_sample[i_imf, :],
                    #         'color': 'purple',  # 所有IMF使用相同颜色，或您可以定义一个IMF颜色列表
                    #         'linestyle': '-',
                    #         'name': f'Decomposed_IMF_{i_imf + 1}_from_Input_X'
                    #     })
                    #
                    # lines_to_plot.extend([
                    #     {'data': main_part_x, 'color': 'red', 'linestyle': '-',
                    #      'name': 'VMD_Main_Part_from_Input_X'},
                    #     {'data': residual_part_x, 'color': 'orange', 'linestyle': '-',
                    #      'name': 'VMD_Residual_Part_from_Input_X'},
                    # ])
                    #
                    # if predicted_main_signal is not None:
                    #     lines_to_plot.append({
                    #         'data': predicted_main_signal,
                    #         'color': 'darkcyan',
                    #         'linestyle': '--',
                    #         'name': 'Predicted_Main_Signal'
                    #     })
                    # if predicted_residual_signal_series is not None:
                    #     lines_to_plot.append({
                    #         'data': predicted_residual_signal_series,
                    #         'color': 'brown',  # 选择一个新颜色，例如棕色
                    #         'linestyle': ':',  # 选择一个新线条样式，例如点线
                    #         'name': 'Predicted_Residual_Signal'
                    #     })
                    # # --- 循环绘制并保存每一条线为单独的图片文件 ---
                    # for line_info in lines_to_plot:
                    #     data = line_info['data']
                    #     color = line_info['color']
                    #     linestyle = line_info['linestyle']
                    #     name = line_info['name']
                    #
                    #     fig = plt.figure(figsize=(8, 2))  # 调整每个图的大小，使其更紧凑
                    #     ax = fig.add_subplot(111)
                    #
                    #     ax.plot(data, color=color, linewidth=2, linestyle=linestyle)
                    #
                    #     # --- 移除所有不必要的图表元素 ---
                    #     ax.set_xticks([])
                    #     ax.set_yticks([])
                    #     ax.set_xticklabels([])
                    #     ax.set_yticklabels([])
                    #     ax.set_xlabel('')
                    #     ax.set_ylabel('')
                    #     ax.set_title('')
                    #     if ax.get_legend() is not None:
                    #         ax.get_legend().remove()
                    #     ax.grid(False)
                    #     plt.box(False)  # 移除边框
                    #
                    #     # 调整子图的布局，使其占据整个画布，无边距
                    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    #
                    #     # 保存图像为 SVG 格式
                    #     # 文件名包含样本和特征信息，以及线条名称
                    #     save_file_name = f'sample{sample_idx}_feature{feature_idx}_{name}.svg'
                    #     save_path = os.path.join(visual_dir, save_file_name)
                    #
                    #     plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
                    #     plt.close(fig)  # 关闭当前图形，释放内存
                    #
                    #     print(f"Saved: {save_path}")

                    batch_origin_y = batch_origin_y.to(self.device)
                    # 如果考虑逆转换的loss，则pred要转回原始值
                    if self.invtrans_loss:
                        preds = self.scaler.inverse_transform(preds)
                        truths = batch_origin_y
                    # 计算预测值和真实值之间的评估指标
                    if self.pred_len == 1:
                        self.metrics.update(preds.view(batch_size, -1), truths.view(batch_size, -1))
                    else:
                        self.metrics.update(preds.contiguous(), truths.contiguous())
                    # 更新进度条
                    progress_bar.update(batch_x.shape[0])
            # y_preds.append(preds)
            # y_truths.append(truths)

            # y_preds = torch.concat(y_preds, dim=0)
            # y_truths = torch.concat(y_truths, dim=0)
            # result会存储所有评估指标的结果
            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result

    def _test(self) -> Dict[str, float]:
        print("Testing .... ")
        # self.model.fm.vmd.visual = True
        self.visual = True
        # 类似_val
        test_result = self._evaluate(self.test_loader)
        for name, metric_value in test_result.items():
            if self._use_wandb():
                wandb.run.summary["test_" + name] = metric_value
                # result = {}
                # for name,value in test_result.items():
                #     result['val_' + name] = value
                # wandb.log(result, step=self.current_epoch)
        # self.model.fm.vmd.visual = False
        self.visual = False
        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        print("Evaluating .... ")
        self.visual = False
        # 对验证集进行评测
        val_result = self._evaluate(self.val_loader)
        # 逐行获取评价指标
        for name, metric_value in val_result.items():
            # 如果用wandb，就设键值对
            if self._use_wandb():
                wandb.run.summary["val_" + name] = metric_value
        # 把评估指标结果打印到output.log
        self._run_print(f"vali_results: {val_result}")
        # 返回评估指标结果
        return val_result

    def _train(self):
        # with torch.no_grad():
        #     if self.model_type == "VMD_DLinear"  and self.current_epoch == 0:
        #         print("\n--- First Set Visualization and VMD Comparison ---")
        #         self.visualize_and_compare_vmd(
        #             model_instance=self.model.fm,
        #             test_loader=self.test_loader,
        #             device=self.device,
        #             epoch_num="0",  # Use a string like "final" or a large epoch number for final test
        #             dataset_name="CustomVaryFreq",
        #         )
        with torch.enable_grad(), tqdm(total=self.train_steps, position=0, leave=True) as progress_bar:
            self.model.train()
            # self.n_model.train()
            # import pdb;pdb.set_trace()
            times = []
            train_loss = []
            k_train = []
            alpha_train = []
            for i, (
                    batch_x,
                    batch_y,
                    origin_y,
                    batch_x_date_enc,
                    batch_y_date_enc, batch_cycle
            ) in enumerate(self.train_loader):

                origin_y = origin_y.to(self.device)
                if self.is_vmd_sep_loss():
                    pass
                # self.n_vmd_model_optim.zero_grad()
                self.model_optim.zero_grad()
                bs = batch_x.size(0)
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                if torch.all(batch_cycle != -1):
                    batch_cycle = batch_cycle.int().to(self.device)

                start = time.time()

                if torch.all(batch_cycle != -1):
                    pred, true = self.process_cycle(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, batch_cycle)
                else:
                    pred, true = self._process_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    )
                    # k训练过程可视化
                # if self.model_type == "VMD_DLinear":
                #     import torch.nn.functional as F
                #     with torch.no_grad():
                #         imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, alpha, k_cont = self.model.fm.vmd.vmd(
                #             batch_x)
                #         k_train.append(k_cont.mean().cpu().numpy())
                #         # 获取 alpha 值
                #         current_alpha = F.softplus(self.model.fm.vmd.vmd.alpha_raw).detach().cpu().numpy()
                #         alpha_train.append(current_alpha)

                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y

                #

                if isinstance(self.model.nm, SAN):
                    mean = self.model.pred_stats[:, :, :self.dataset.num_features]
                    std = self.model.pred_stats[:, :, self.dataset.num_features:]
                    sliced_true = true.reshape(bs, -1, 12, self.dataset.num_features)
                    loss = self.loss_func(pred, true) + self.loss_func(mean, sliced_true.mean(2)) + self.loss_func(std,
                                                                                                                   sliced_true.std(
                                                                                                                       2))
                else:
                    # 相比于分开计算,这里就是记录总的loss,一起更新
                    loss = self.loss_func(pred, true) + self.model.nm.loss(true)
                if self.is_vmd_sep_loss():
                    loss = self.loss_func(pred, true) + self.model.fm.vmd.loss(true) + self.model.fm.vmd.vmd_loss(
                        batch_x) + self.model.nm.loss(true)
                    # loss_vmd = self.model.nm.vmd_loss(batch_x)
                    # loss_vmd.backward(retain_graph=True)
                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()

                # mlp_params_changed = False
                # print("\n--- 检查 MLP 参数变化（与初始值对比）---")
                # for name, param in self.model.fm.vmd.vmd.mlp.named_parameters():
                #     initial_val = self.initial_mlp_params[name]
                #     current_val = param.data  # param.data 是参数的实际值
                #
                #     # 计算参数变化的绝对差值的范数
                #     change_norm = torch.norm(current_val - initial_val).item()
                #
                #     if change_norm > 1e-6:  # 设置一个小的阈值来判断变化
                #         mlp_params_changed = True
                #         print(f"  MLP 参数 '{name}' 已变化 (范数差: {change_norm:.8f})")
                #     else:
                #         print(f"  MLP 参数 '{name}' 无明显变化 (范数差: {change_norm:.8f})")
                #
                # if not mlp_params_changed:
                #     print("警告：MLP 参数在训练后无明显变化。请检查学习率或损失贡献。")
                # print("--- 参数变化检查结束 ---")
                end = time.time()

                times.append(end - start)
            # if self.model_type == "VMD_DLinear":
            #     import matplotlib.pyplot as plt
            #     # 绘图
            #     plt.figure(figsize=(10, 4))
            #     plt.plot(k_train, label='Average k per batch', color='blue')
            #     plt.xlabel('Batch Index')
            #     plt.ylabel('Average k')
            #     plt.title('k Value over Training Batches')
            #     plt.grid(True)
            #     plt.legend()
            #     plt.tight_layout()
            #     # 保存图片
            #     save_file = f"{self.model_type}{self.dataset_type}{self.pred_len}{self.current_epoch}_k_train.png"
            #     plt.savefig("visual/" + save_file, dpi=300)  # dpi=300 提高清晰度
            #     plt.close()  # 释放内存资源，适合写在训练代码里
            #
            #     plt.figure(figsize=(10, 4))
            #     plt.plot(alpha_train, label='Learned Alpha per batch', color='red')  # Using red for alpha
            #     plt.xlabel('Batch Index')
            #     plt.ylabel('Alpha Value')
            #     plt.title('Alpha Value over Training Batches')
            #     plt.grid(True)
            #     plt.legend()
            #     plt.tight_layout()
            #
            #     save_file_alpha = f"{self.model_type}{self.dataset_type}{self.pred_len}{self.current_epoch}_alpha_train.png"
            #     plt.savefig("visual/" + save_file_alpha, dpi=300)  # dpi=300 提高清晰度
            #     plt.close()  # Release memory resources

            print(f"average iter: {np.mean(times) * 1000}ms")

            return train_loss

    # 分开的训练

    def _sep_train(self):
        # 开启梯度计算   并对训练过程以进度条显示
        with torch.enable_grad(), tqdm(total=self.train_steps, position=0, leave=True) as progress_bar:
            # 把模型设置为训练模式，在此过程中一些模块的实现与评估是不一样的
            self.model.train()
            # self.n_model.train()
            # import pdb;pdb.set_trace()
            # 记录每个批次处理时间
            times = []
            # 每个批次损失值,分开即用的是预测模型的loss
            train_loss = []
            # 遍历每一个批次的输入、输出y经过缩放的、实际原始y、x和y的时间编码
            for i, (
                    batch_x,
                    batch_y,
                    origin_y,
                    batch_x_date_enc,
                    batch_y_date_enc, batch_cycle
            ) in enumerate(self.train_loader):
                # 把张量放到device上
                origin_y = origin_y.to(self.device)
                # 梯度清零，每一个批次都要重新求梯度
                if self.is_vmd_sep_loss():
                    pass
                    # self.n_vmd_model_optim.zero_grad()
                self.n_model_optim.zero_grad()
                self.f_model_optim.zero_grad()

                bs = batch_x.size(0)

                # 把每一次的内容都放到device上:::
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                if torch.all(batch_cycle != -1):
                    batch_cycle = batch_cycle.int().to(self.device)

                start = time.time()

                if torch.all(batch_cycle != -1):
                    pred, true = self.process_cycle(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, batch_cycle)
                else:
                    pred, true = self._process_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    )
                # 判断是否考虑逆变换损失  如果考虑就全都反缩放回去
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    # true也变成实际值
                    true = origin_y

                # 分开计算损失
                # 通过MSELoss计算损失
                if self.is_vmd_sep_loss():
                    # loss_vmd = self.model.nm.vmd_loss(batch_x)
                    # loss_vmd.backward(retain_graph=True)
                    loss += self.vmd.loss(true)

                loss = self.loss_func(pred, true)
                # 计算规范化的损失
                lossn = self.model.nm.loss(true)

                # 反向传播
                loss.backward(retain_graph=True)
                lossn.backward(retain_graph=True)

                # torch.nn.utils.clip_grad_norm_(
                #     self.model.parameters(), self.max_grad_norm
                # )

                progress_bar.update(batch_x.size(0))
                # 计算损失值
                train_loss.append(loss.item())
                # 进度条设计
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.f_model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                # 更新参数模型
                if self.is_vmd_sep_loss():
                    pass
                    # self.n_vmd_model_optim.step()
                self.n_model_optim.step()
                self.f_model_optim.step()

                # 计算训练时间并加入
                end = time.time()
                times.append(end - start)

            print("average iter: {}ms", np.mean(times) * 1000)
            # 最后返回记录损失值的
            return train_loss

    # 查看当前run是否存在
    def _check_run_exist(self, seed: str):
        # 判断该seed对应的路径目录是否存在
        if not os.path.exists(self.run_save_dir):
            # 如果目录不存在，则创建新目录
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        # 否则就存在
        else:
            print(f"result directory exists: {self.run_save_dir}")

        # 写入一些参数到args.json
        with codecs.open(os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)
        # 返回run_checkpoint_filepath是否存在
        exists = os.path.exists(self.run_checkpoint_filepath)

        return exists

    # 从指定检查点恢复运行
    def _resume_run(self, seed):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        # 检查点路径
        run_checkpoint_filepath = os.path.join(self.run_save_dir, f"run_checkpoint.pth")

        print(f"resuming from {run_checkpoint_filepath}")
        # 加载检查点的内容
        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        # if not isinstance(self.n_model, No):
        #     self.n_model.load_state_dict(check_point["n_model"])
        #     self.norm_model_optim.load_state_dict(check_point["norm_model_optim"])
        # 把检查点中的内容赋值到对应的地方，实现模型训练的恢复
        self.model.load_state_dict(check_point["model"])

        if self.is_sep_loss():
            self.f_model_optim.load_state_dict(check_point["f_optimizer"])
            self.n_model_optim.load_state_dict(check_point["n_optimizer"])
        else:
            self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

    def _resume_from(self, path):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        run_checkpoint_filepath = os.path.join(path, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

    def _load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.device))

    def single_step_forecast(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self.experiment_label = f"{self.model_type}-w{self.windows}-h{self.horizon}"

    def run_more_epochs(self, seed=42, epoches=200) -> Dict[str, float]:
        self._setup_run(seed)

        if self._check_run_exist(seed):
            self._resume_run(seed)

        self.epoches = epoches

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        self.model_parameters_num = self.count_parameters(self._run_print)

        self._run_print(
            f"model parameters: {self.model_parameters_num}"
        )
        if self._use_wandb():
            wandb.run.summary["parameters"] = self.model_parameters_num
        # for resumable reproducibility

        epoch_time = time.time()
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"loss no decreased for {self.patience} epochs,  early stopping ...."
                )
                break

            if self._use_wandb():
                wandb.run.summary["at_epoch"] = self.current_epoch
            # for resumable reproducibility
            self.reproducible(seed + self.current_epoch)
            if self.is_sep_loss():
                train_losses = self._sep_train()
            else:
                train_losses = self._train()

            self._run_print(
                "Epoch: {} cost time: {}".format(
                    self.current_epoch + 1, time.time() - epoch_time
                )
            )
            self._run_print(
                f"Traininng loss : {np.mean(train_losses)}"
            )

            val_result = self._val()

            # test
            test_result = self._test()

            if self._use_wandb():
                result = {'train_loss': float(np.mean(train_losses))}
                for name, value in val_result.items():
                    result['val_' + name] = value
                for name, value in test_result.items():
                    result['test_' + name] = value
                wandb.log(result, step=self.current_epoch)

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result[self.loss_func_type], model=self.model)

            self._save_run_check_point(seed)

            self.scheduler.step()

            # if self._use_wandb():
            #     wandb.log(result, step=self.current_epoch)

        self._load_best_model()
        best_test_result = self._test()
        self.run_setuped = False
        return best_test_result

    def count_parameters(self, print_fun):
        # 创建一个表格对象table，表头设为 “Modules” 和 “Parameters”。
        table = PrettyTable(["Modules", "Parameters"])
        # 初始化总参数为0
        total_params = 0
        # 遍历模型的参数名称和张量
        for name, parameter in self.model.named_parameters():
            # 如果参数不要求梯度更新，则跳过，不计算
            if not parameter.requires_grad:
                continue
            # 计算张量元素数量
            params = parameter.numel()
            # 名称和参数数量加入到表格
            table.add_row([name, params])
            # 加入总参数数量
            total_params += params
            # 把表格和计算出的总参数数量输出到ouput.log中
        print_fun(table)
        print_fun(f"Total Trainable Params: {total_params}")

        # 返回总参数数量
        return total_params

    # 运行函数，结果用一个字典存储
    def run(self, seed=42) -> Dict[str, float]:
        # 判断是否完成，完成的话就返回一个空字典
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return {}
        # 运行的配置
        self._setup_run(seed)

        # 判断run_checkpoint_filepath是否存在
        if self._check_run_exist(seed):
            # 如果存在，则取其中的内容然后恢复模型训练
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")
        # 计算模型参数数量，传入runprint方法
        self.model_parameters_num = self.count_parameters(self._run_print)
        # 再次向output.log中打印参数数量
        self._run_print(
            f"model parameters: {self.model_parameters_num}"
        )

        # 如果与wandb集成，添加一组键值对  parameters对参数总数量
        if self._use_wandb():
            wandb.run.summary["parameters"] = self.model_parameters_num
        # for resumable reproducibility
        # 记录当前轮数时间
        epoch_time = time.time()
        # 训练至设定的轮数
        while self.current_epoch < self.epochs:
            # 判断是否早停
            if self.early_stopper.early_stop is True:
                # 如果早停，并将对应内容输出到output.log之中
                self._run_print(
                    f"loss no decreased for {self.patience} epochs,  early stopping ...."
                )
                # 退出循环，即结束训练
                break
            # 如果用wandb，记录at_epoch
            if self._use_wandb():
                wandb.run.summary["at_epoch"] = self.current_epoch
            # for resumable reproducibility
            # 每个epoch都有不同seed，并确保可重现性
            self.reproducible(seed + self.current_epoch)

            if self.is_sep_loss():
                # 如果损失分开计算，则分开训练
                train_losses = self._sep_train()

            else:
                # 不分开的话
                train_losses = self._train()

            # 记录一些信息到output.log
            self._run_print(
                "Epoch: {} cost time: {}".format(
                    self.current_epoch + 1, time.time() - epoch_time
                )
            )
            self._run_print(
                f"Traininng loss : {np.mean(train_losses)}"
            )

            # self._run_print(f"Val on train....")
            # trian_val_result = self._evaluate(self.train_loader)
            # self._run_print(f"Val on train result: {trian_val_result}")

            # 在验证集评测
            # evaluate on val set
            result = self._val()
            # test
            test_result = self._test()
            # epoch轮次加1
            self.current_epoch = self.current_epoch + 1

            # 设置早停，从评估指标获得的数值判早停
            self.early_stopper(result[self.loss_func_type], model=self.model)
            # 记录点
            self._save_run_check_point(seed)
            # 根据训练控制学习率变化
            self.scheduler.step()

            # if self._use_wandb():
            #     wandb.log(result, step=self.current_epoch)
        # 加载最好的模型参数
        self._load_best_model()
        # 计算最好模型参数对应的测试评估结果
        best_test_result = self._test()
        # run完，再把那个setuped设为false
        self.run_setuped = False
        # 返回测试结果
        return best_test_result

    def dp_run(self, seed=42, device_ids: List[int] = [0, 2, 4, 6], output_device=0):
        self._setup_dp_run(seed, device_ids, output_device)
        print(f"run : {self.current_run} in seed: {seed}")
        print(
            f"model parameters: {sum([p.nelement() for p in self.model.parameters()])}"
        )
        epoch_time = time.time()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            if self._use_wandb():
                wandb.run.summary["at_epoch"] = epoch
            self._train()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # evaluate on vali set
            self._val()

            self._save(seed=seed)

        return self._test()

    # 调用该函数
    # 以run_fan_wandb为例，传入seeds[1,2,3,4,5]
    def runs(self, seeds: List[int] = [42, 43, 44]):
        # 如果有finished为True，说明该对象对应的实验已经完成
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return
        # 如果与wandb集成完
        if self._use_wandb():
            # 更新seeds的配置
            wandb.config.update({"seeds": seeds})
        # 存结果
        results = []
        # 遍历，索引以及对应的种子
        for i, seed in enumerate(seeds):
            # 每次运行的序列
            self.current_run = i
            # 如果与wandb集成完
            if self._use_wandb():
                # 添加实验次数加入到wandb的summary之中
                wandb.run.summary["at_run"] = i
            # 清显存缓存
            torch.cuda.empty_cache()
            # 调用run方法并存储测试最好结果的评估结果
            result = self.run(seed=seed)
            # 清显存缓存
            torch.cuda.empty_cache()
            # 放入results
            results.append(result)
            # 如果用wandb,还是设内容到里面
            if self._use_wandb():
                for name, metric_value in result.items():
                    wandb.run.summary["test_" + name] = metric_value

        # 计算评估指标的均值和标准差
        df = pd.DataFrame(results)

        self.metric_mean_std = df.agg(["mean", "std"]).T
        # 打印对应的结果
        print(
            self.metric_mean_std.apply(
                lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
            )
        )
        # 如果用wandb,还是将那些均值以及标准差放入
        if self._use_wandb():
            for index, row in self.metric_mean_std.iterrows():
                wandb.run.summary[f"{index}_mean"] = row["mean"]
                wandb.run.summary[f"{index}_std"] = row["std"]
                wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"
        # wandb结束
        wandb.finish()
        # return self.metric_mean_std

    def process_cycle(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, batch_cycle):
        pass


def main():
    exp = Experiment(
        dataset_type="ETTm1",
        data_path="./data",
        optm_type="Adam",
        model_type="Informer",
        batch_size=32,
        device="cuda:3",
        windows=10,
        epochs=1,
        lr=0.001,
        pred_len=3,
        scaler_type="MaxAbsScaler",
    )

    # exp = Experiment(settings)
    # exp.run()


# This function forcibly kills the remaining wandb process.
def force_finish_wandb():
    with open(os.path.join(os.path.dirname(__file__), './wandb/latest-run/logs/debug-internal.log'), 'r') as f:
        last_line = f.readlines()[-1]
    match = re.search(r'(HandlerThread:|SenderThread:)\s*(\d+)', last_line)
    if match:
        pid = int(match.group(2))
        print(f'wandb pid: {pid}')
    else:
        print('Cannot find wandb process-id.')
        return

    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Process with PID {pid} killed successfully.")
    except OSError:
        print(f"Failed to kill process with PID {pid}.")


# Start wandb.finish() and execute force_finish_wandb() after 60 seconds.
def try_finish_wandb():
    threading.Timer(5, force_finish_wandb).start()
    wandb.finish()


if __name__ == "__main__":

    # main()
    dataset: TimeSeriesDataset = Electricity(root="./data")
    train_dataset = MultiStepTimeFeatureSet(
        dataset,
        scaler=StandarScaler(),
        time_enc=0,
        window=96,
        horizon=1,
        steps=96,
        freq="h",
        scaler_fit=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=3,
    )
    for i, (
            batch_x,
            batch_y,
            origin_y,
            batch_x_date_enc,
            batch_y_date_enc,
    ) in enumerate(train_loader):
        print(f"Batch {i + 1}")
        print("batch_x:", batch_x.shape)  # 输出 batch_x 的值
        print("batch_y:", batch_y.shape)  # 输出 batch_y 的值
        print("batch_x_date_enc:", batch_x_date_enc.shape)  # 输出 batch_x_date_enc 的值
        print("batch_y_date_enc:", batch_y_date_enc.shape)  # 输出 batch_y_date_enc 的值
        if i == 0:
            break
