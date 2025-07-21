from dataclasses import dataclass, field

import fire
import torch

from torch_timeseries.models import FEDformer
from torch_timeseries.norm_experiments.experiment import NormExperiment


@dataclass
class FEDformerExperiment(NormExperiment):
    # 设置model类型为FEDformer
    model_type: str = "FEDformer"
    # label_len : int = 48
    # 默认使用傅里叶变换
    version: str = 'Fourier'

    # 模型深度默认设置为3
    L: int = 3
    # 前馈神经网络维度
    d_ff: int = 2048
    # 激活函数
    activation: str = 'gelu'
    # 编码层的数量
    e_layers: int = 2
    # 解码层的数量
    d_layers: int = 1
    # 模式选择方法,默认是随机选择
    mode_select: str = 'random'
    # 模式数量
    modes: int = 64
    # 是否输出注意力信息
    output_attention: bool = True
    # 移动平均参数
    moving_avg: list = field(default_factory=lambda: [24])
    # 多头注意力有多少头
    n_heads: int = 8
    # 交叉激活函数，默认是tanh
    cross_activation: str = 'tanh'
    # 模型特征维度
    d_model: int = 512
    # 嵌入方式
    embed: str = 'timeF'
    # 时间频率为h
    freq: str = 'h'
    # dropout率
    dropout: float = 0.0
    # 基础函数 legendre多项式
    base: str = 'legendre'
    # 把基类的l2重新赋默认值
    l2_weight_decay: float = 0.0

    def _init_f_model(self):

        # 设置标签长度为输入窗口的长度
        self.label_len = self.windows

        # 创建该预测模型的对象
        self.f_model = FEDformer(
            # 根据数据集的特征数初始化编码器和解码器的输入
            enc_in=self.dataset.num_features,
            dec_in=self.dataset.num_features,
            # 输入序列长度
            seq_len=self.windows,
            # 预测长度
            pred_len=self.pred_len,
            # 标签长度
            label_len=self.label_len,
            # 输出特征的数量
            c_out=self.dataset.num_features,

            # 将之前类的变量作为参数传入
            version=self.version,
            L=self.L,
            d_ff=self.d_ff,
            activation=self.activation,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            mode_select=self.mode_select,
            modes=self.modes,
            output_attention=self.output_attention,
            moving_avg=self.moving_avg,
            n_heads=self.n_heads,
            cross_activation=self.cross_activation,
            d_model=self.d_model,
            embed=self.embed,
            freq=self.freq,
            dropout=self.dropout,
            base=self.base
        )
        # 把这个模型部署到cuda上
        self.f_model = self.f_model.to(self.device)

    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x:  (B, T, N)
        # batch_y:  (B, Steps,T)
        # batch_x_date_enc:  (B, T, N)
        # batch_y_date_enc:  (B, T, Steps)

        # outputs:
        # pred: (B, O, N)
        # label:  (B,O,N)
        # 初始化解码器输入
        # 预测的占位符
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)

        # 解码器的另一部分，将x最后一部分数据作为输入，为我们预测的依据
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        # 两者连接起来作为解码器的输入
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        # 跨注意力机制（CrossAttentionMechanism）主要是解码器通过这个模块从编码器的输出中“关注”到与当前解码位置相关的重要特征信息

        # 获得解码器输入对应的时间编码
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len:, :], batch_y_date_enc], dim=1
        )

        # 对两种输入都进行规范化，这里即可以选择归一化方式进行相应的规范化，而针对于FAN的话，只对x规范化，dec_inp不规范化
        # 获得去除前k个频率的x以及dec_inp这里不操作
        batch_x, dec_inp = self.model.normalize(batch_x, dec_inp=dec_inp)  # (B, T, N)   # (B,L,N)


        # 通过预测模型预测
        # 获得预测数据
        pred = self.model.fm(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)[0]
        # 逆归一化就是加上FAN那里计算的前k个的预测
        pred = self.model.denormalize(pred)

        # 返回
        return pred, batch_y  # (B, O, N), (B, O, N)

        # pred = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc) # (B, O, N)
        # return pred


def cli():
    # 会自动将通过命令行运行该python文件时的参数映射到FEDformerExperiment的构造函数，默认会运行其中的run函数
    fire.Fire(FEDformerExperiment)


def main():
    exp = FEDformerExperiment(
        dataset_type="ExchangeRate",
        data_path="./data",
        norm_type='RevIN',  # No  DishTS
        optm_type="Adam",
        batch_size=128,
        device="cuda:1",
        windows=96,
        pred_len=96,
        horizon=1,
        epochs=100,
        dropout=0.05,
        d_ff=256,
    )

    exp.run()


if __name__ == "__main__":
    # main()
    # print(1)
    cli()
