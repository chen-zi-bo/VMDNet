# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn


# 模式对应着频率分量
# 96
def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    # 确保模式数量小于频率范围
    modes = min(modes, seq_len // 2)
    # 如果random，随机选取模式
    if mode_select_method == 'random':
        # 随机选一系列模式，用索引数代表选取第几个模式
        index = list(range(0, seq_len // 2))
        # 打乱
        np.random.shuffle(index)
        # 选前modes个，实现随机选取
        index = index[:modes]
    else:
        # 否则就直接找前modes个,从低到高了
        index = list(range(0, modes))

    # 最后排个序,获得要找的频率分量对应的索引
    index.sort()

    return index


# 该模块是在做FFT,线性变换以及逆FFT
# ########## fourier layer #############
class FourierBlock(nn.Module):
    # 512          512          96      64,              random
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """

        # get modes on frequency domain
        #                                 96        64                    random
        # 获取要得到的模式的索引
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)

        print('modes={}, index={}'.format(modes, self.index))

        # 设置了一个用于权重缩放的常数值
        self.scale = (1 / (in_channels * out_channels))
        # 设置了一个四维权重矩阵
        self.weights1 = nn.Parameter(  # 复数类型
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    # 实现复数一维卷积
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # 原理上是自注意力的实现
        # 传播过程走的是
        # 首先对数据进行傅里叶变换

        # size = [B, L, H, E]
        # batch length  特征维度 特征向量的维度
        B, L, H, E = q.shape
        # (B, H, E, L)
        x = q.permute(0, 2, 3, 1)
        # 进行傅里叶变换，(B, H, E, L // 2 + 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)

        # Perform Fourier neural operations
        # 提前准备存储结果
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        # 傅里叶变换后的结果和weight相乘获得对应结果(按照前面找到的模式，即index)

        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        # 逆傅里叶变换转回(B, H, E, L)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


# ########## Fourier Cross Former ####################
# 跨注意力模块
class FourierCrossAttention(nn.Module):
    # 512         512           查询序列长度self.seq_len // 2 + self.pred_len, 键值序列长度self.seq_len,
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):

        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        # 三个变量的赋值
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # get modes for queries and keys (& values) on frequency domain
        # 找到查询和键值的模式的序列
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)

        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        # 缩放以及权重矩阵
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

        # 输入与权重的复杂乘法。

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # 此时传播要传入三个q,k,v
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        # 维度变化
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        # q的fft变换
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        # xq_ft_只包含对应模式的频率
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        # k也是类似的操作
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        # q和k的模型的频率乘积，实际上类似于求相关性，获得一个注意力矩阵
        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        # 根据激活函数类型实现对应操作
        if self.activation == 'tanh':
            # 变成一个个实数
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            # 变成实数
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            # 实部设为该实数，复部为0，变回复数矩阵
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        # 将加权和k的频域乘积
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        # 然后进行一个线性变换
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        # 创建存储结果的张量
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        # 找出对应频率的每一个输出
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # 转成时域
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)
