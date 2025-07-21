"""
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""
import numpy as np
import torch


# 早停实现
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    # 传入patience: int = 5，verbose为true，还传入最好权重文件存储的目录
    def __init__(
            self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        # 变量的赋值
        self.patience = patience
        # verbose是否打印详细信息
        self.verbose = verbose
        # 记录没有改善的轮数
        self.counter = 0
        # 最佳得分
        self.best_score = None
        # 是否早停
        self.early_stop = False
        # 模型保存路径
        self.path = path
        # 验证损失的最小值
        self.val_loss_min = np.Inf
        # 两参数
        self.delta = delta
        self.trace_func = trace_func

    def get_state(self):
        # 获得一系列的状态
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "val_loss_min": self.val_loss_min,
            "early_stop": self.early_stop
        }
    # 设置一系列状态
    def set_state(self, state):
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.val_loss_min = state["val_loss_min"]
        self.early_stop = state["early_stop"]

    # 重置 重新赋值
    def reset(self):

        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
    # 该方法直接即可通过对象(参数)去调用该call方法

    def __call__(self, val_loss, model):
        # 损失值取负为得分
        score = -val_loss

        # 第一次调用则设置改最好得分为score
        if self.best_score is None:
            self.best_score = score
            # 并将参数进行保存
            self.save_checkpoint(val_loss, model)
        # 如果得分要低的话,说明这一次是损失值不减的一次
        elif score < self.best_score + self.delta:
            # 增加次数
            self.counter += 1

            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 如果次数大于等于patienct,则设置早停为True
            if self.counter >= self.patience:
                self.early_stop = True
        # 得分高的话,则需要进行更新
        else:

            self.best_score = score
            self.save_checkpoint(val_loss, model)
            # 并重置轮数为0
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        """Saves model when validation loss decrease."""
        # 如果要详细显示内容
        if self.verbose:
            # 实际上这里为打印
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # 将参数保存在对应path
        torch.save(model.state_dict(), self.path)
        # 并更新对应的val_loss_min
        self.val_loss_min = val_loss
