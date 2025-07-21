from dataclasses import dataclass, field

import fire
import torch

from torch_timeseries.models.MultiTaskEMDLearning import MultiTaskEMDLearning
from torch_timeseries.norm_experiments.experiment import NormExperiment


@dataclass
class MultiTaskEMDExperiment(NormExperiment):
    model_type: str = "MultiTaskEMDLearning"
    # label_len : int = 48

    # 前馈神经网络维度
    d_ff: int = 2048
    # 激活函数
    activation: str = 'gelu'
    # 编码层的数量
    e_layers: int = 2
    # 解码层的数量
    d_layers: int = 1
    # 是否输出注意力信息
    output_attention: bool = True
    # 多头注意力有多少头
    n_heads: int = 8
    # 模型特征维度
    d_model: int = 512
    # 嵌入方式
    embed: str = 'timeF'
    # 时间频率为h
    freq: str = 'h'
    # dropout率
    dropout: float = 0.0
    # 把基类的l2重新赋默认值
    l2_weight_decay: float = 0.0

    k_user_defined: int = 3

    hidden_dim: int = 64

    def _init_f_model(self):
        self.label_len = self.windows

        self.f_model = MultiTaskEMDLearning(
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

            d_ff=self.d_ff,
            activation=self.activation,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            output_attention=self.output_attention,
            n_heads=self.n_heads,
            d_model=self.d_model,
            embed=self.embed,
            freq=self.freq,
            dropout=self.dropout,
            hidden_dim=self.hidden_dim
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

        batch_x, dec_inp = self.model.normalize(batch_x, dec_inp=dec_inp)  # (B, T, N)   # (B,L,N)

        # 通过预测模型预测
        # 获得预测数据
        pred = self.model.fm(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc, self.k_user_defined, self.device)
        # 逆归一化就是加上FAN那里计算的前k个的预测
        pred = self.model.denormalize(pred)

        # 返回
        return pred, batch_y  # (B, O, N), (B, O, N)

        # pred = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc) # (B, O, N)
        # return pred


def cli():
    # 会自动将通过命令行运行该python文件时的参数映射到的构造函数，默认会运行其中的run函数
    fire.Fire(MultiTaskEMDExperiment)


if __name__ == "__main__":
    cli()
