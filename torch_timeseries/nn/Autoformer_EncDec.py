import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        # 归一化模块
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        # 这一步是将每个特征通道的均值和方差进行标准化，使得每个通道的输出均值为0，方差为1。
        x_hat = self.layernorm(x)
        # 求偏差
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        # 同时还能更好捕捉特征变化和季节性趋势
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 创建了一维平均池化层,窗口大小为kernerl_size，步长为stride，padding=0表示不填充
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 往前后填充对应内容，然后利用平均池化层进行池化
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    # 只会传入一个移动窗口大小
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        # 只有一个移动平均层
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # 直接计算即可
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        # 传入移动的窗口大小
        super(series_decomp_multi, self).__init__()
        # 根据kernel列表情况创建对应的moving_avg实例
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        # 创建一个全连接层，1-列表长度
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        # 存储每一个移动平均层的计算结果
        moving_mean = []
        for func in self.moving_avg:
            # 计算对应的池化结果
            moving_avg = func(x)
            # 加入结果中，不过添加一个维度
            moving_mean.append(moving_avg.unsqueeze(-1))
        # 在最后一个维度连接起来
        moving_mean = torch.cat(moving_mean, dim=-1)
        # 求了一个加权平均池化
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)

        res = x - moving_mean
        # 返回去掉后的剩余结果以及移动平均值
        return res, moving_mean


class FourierDecomp(nn.Module):
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    # """  # AutoCorrelationLayer  512    2048     24       0.0        gelu'

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        # 全连接层维度
        d_ff = d_ff or 4 * d_model
        # 注意力模块
        self.attention = attention
        # 两个1维卷积
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            # 返回的是一个模块，该模块实现平均池化层进行池化的结果，返回去掉后的剩余结果以及移动平均值
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 激活函数
        self.activation = F.relu if activation == "relu" else F.gelu

    # mask没用上的
    def forward(self, x, attn_mask=None):

        # 把x看作q、k、v，强调频率
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )

        # 进行残差连接并使用dropout
        x = x + self.dropout(new_x)
        # 去掉趋势成分
        x, _ = self.decomp1(x)

        y = x
        # 对y卷积再激活然后再卷积回来，提取特征
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 残差链接，然后再次去掉趋势成分
        res, _ = self.decomp2(x + y)

        return res, attn


# 编码器模块
class Encoder(nn.Module):
    """
    Autoformer encoder
    """  # 注意力层列表     卷积层         归一化层 FEdformer中没有用卷积层

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        # 创建注意力层
        self.attn_layers = nn.ModuleList(attn_layers)
        # 有则创，无则为null
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        # 一个归一化层
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # 存储注意力权重
        attns = []
        # 有卷积层的话,每次走一遍自注意层和卷积层
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                # 注意力加进去
                attns.append(attn)
            # 没有卷积，剩余的自注意力层也要走，加入对应的注意力
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        # 没有的话，就只走所有的自注意力层

        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        # 如果有归一化层，再归一化一次
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):

        super(DecoderLayer, self).__init__()
        # 前馈网络维度
        d_ff = d_ff or 4 * d_model
        # 模块赋值
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            # 返回的是一个模块，该模块实现平均池化层进行池化的结果，返回去掉后的剩余结果以及移动平均值
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 投影到目标维度
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):

        # 自注意力计算然后dropout，同时实现残差链接
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        # 提取残差和趋势
        x, trend1 = self.decomp1(x)

        # 交叉注意力并进行dropout以及残差链接
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        # 再次趋势分解
        x, trend2 = self.decomp2(x)

        # 对x特征提取
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # 再次趋势分解
        x, trend3 = self.decomp3(x + y)
        # 总体趋势
        residual_trend = trend1 + trend2 + trend3
        # 投影到目标维度
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    #                   解码器层        归一化层         投影层
    def __init__(self, layers, norm_layer=None, projection=None):

        super(Decoder, self).__init__()
        # 解码器层列表
        self.layers = nn.ModuleList(layers)
        # 后面处理模块
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):

        for layer in self.layers:
            # 每一层计算趋势，求出找出的总趋势
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

            trend = trend + residual_trend
        # 同时最后计算的x就是季节部分的内容
        if self.norm is not None:
            # 对剩余的内容归一化
            x = self.norm(x)

        if self.projection is not None:
            # 对剩余的内容投影
            x = self.projection(x)

        return x, trend
