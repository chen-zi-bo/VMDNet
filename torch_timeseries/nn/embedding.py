import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    # 512   最大长度
    def __init__(self, d_model, max_len=5000):
        # Transformer的位置编码通常使用正弦和余弦函数来生成。其核心思想是：通过对不同位置进行正弦和余弦的组合，得到独特的、平滑变化的编码。
        # 位置编码中的奇数和偶数维度会分别使用正弦和余弦函数进行编码。
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.

        # 创建张量，存储位置编码
        pe = torch.zeros(max_len, d_model)
        # 设置该张量不计算梯度，不进行更新
        pe.requires_grad_(False)
        # max_len，1的一个张量
        position = torch.arange(0, max_len).unsqueeze(1)

        # 计算缩放因子，调整正弦和余弦的频率
        div_term = (torch.arange(0, d_model, 2)
                    * -(math.log(10000.0) / d_model)).exp()
        # 分别和position乘在求sin或cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加一个维度，(1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 注册为缓冲区变量，GPU中固定该位置编码
        self.register_buffer('pe', pe)

    # 返回与输入长度相匹配的部分的位置编码
    def forward(self, x):
        """return positional embedding of multivariate input x

        Args:
            x (torch.Tensor): size of (B, L, d_model), B is the batch_size, L is the sequence length

        Returns:
            torhc.Tensor: size of (1, L, d_model)
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        # 数据特征数   512

        super(TokenEmbedding, self).__init__()
        # 填充量
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 定义一维卷积层
        #                 使用循环填充
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 对一维卷积层的权重进行Kaiming正态初始化
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        # 传播只需要通过一次卷积进行计算即可
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model)
        position = torch.arange(0, c_in).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2)
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    # 512           timeF           h
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        # 频率字典
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        # 获得频率对应值
        d_inp = freq_map[freq]

        # 全连接层,对应值映射到512
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        # 传播就是走一次全连接层
        return self.embed(x)


class DataEmbedding(nn.Module):
    # TODO: freq d,m embeddings
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, time_embed=True):
        super(DataEmbedding, self).__init__()

        self.time_embed = time_embed

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.time_embed:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                        freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)

        self.d_model = d_model
        self.c_in = c_in
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if self.time_embed:
            x = self.value_embedding(
                x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        else:
            x = self.value_embedding(
                x) + self.position_embedding(x)

        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        # 数据特征数   512        timeF      h         0.0
        super(DataEmbedding_wo_pos, self).__init__()

        # 创建三个嵌入模块

        # 该模块进行一次卷积操作
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 位置编码的模型
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # 如果不等于timeF，用TemporalEmbedding
        # 如果是timeF，就创建TimeFeatureEmbedding模块

        # 该模块根据时间频率设置一个全连接层,输出512维度
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        # 一个dropout层,神经元有p的概率把输出置为0
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 传播,首先输入   数据以及时间标记
        # try:
        #     对输入数据求卷积操作            对时间标记设置全连接层     两者加起来
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        # 应用一个dropout层
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
