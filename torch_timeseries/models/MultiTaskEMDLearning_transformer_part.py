# 嵌入层

# 添加时间周期编码的嵌入层
import time

import torch
from torch import nn

from torch_timeseries.nn.embedding import TokenEmbedding, TemporalEmbedding, TimeFeatureEmbedding, FixedEmbedding


class DataEmbedding_cycle_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        # 数据特征数   512        timeF      h         0.0
        super(DataEmbedding_cycle_pos, self).__init__()

        # 创建三个嵌入模块

        # 该模块进行一次卷积操作
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        self.cycle_position_embedding = Cycle_PositionalEmbedding(d_model=d_model)

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
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.cycle_position_embedding(x)
        # except:
        #     a = 1
        # 应用一个dropout层
        return self.dropout(x)


class Cycle_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', k=1):
        super(Cycle_PositionalEmbedding, self).__init__()

        self.d_model = d_model
        self.k = k  # 获取前k个频率成分
        self.embed_type = embed_type

        # 使用固定的周期编码还是学习型嵌入（基于固定模式的周期编码）
        self.periodic_embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding


    def forward(self, x):
        """
               输入：x, 形状 (b, t, n)，b为batch_size, t为时间步数, n为特征维度
               输出：周期编码后的张量，形状为 (b, t, n, d_model)
               """

        # 记录开始时间
        start_time = time.time()
        device = x.device
        b, t, n = x.size()
        top_k_periods = torch.zeros((b, self.k, n), device=x.device)  # 预分配结果张量
        for i in range(n):
            x_i = x[:, :, i]  # 获取第i个维度的时间序列数据

            # 进行快速傅里叶变换（rfft）
            fft_result = torch.fft.rfft(x_i)
            fft_abs = fft_result.abs()

            # 获取前k个频率的索引
            top_k_vals, top_k_indices = torch.topk(fft_abs, self.k, dim=-1)

            # 从频率中提取周期
            periods = torch.fft.fftfreq(t, d=1.0)  # 假设采样间隔d=1.0

            # 确保 periods 和 top_k_indices 在相同的设备上
            periods = periods.to(device)
            top_k_indices = top_k_indices.to(device)
            # 计算主周期
            top_k_periods[:, :, i] = (t / periods[top_k_indices]).clamp(min=1, max=t)

            # 相同形状周期信息数据
        # 获取主周期 (b, n)
        periods = top_k_periods[:, self.k - 1, :].clamp(min=1)  # 避免周期为0
        # 生成时间索引 (t,)
        time_indices = torch.arange(t, device=top_k_periods.device).unsqueeze(0).unsqueeze(-1)  # (1, t, 1)
        # 计算循环索引 (b, t, n)
        periodic_indices = time_indices % periods.unsqueeze(1)  # (b, t, n)

        end_time = time.time()
        # 计算并打印运行时间
        elapsed_time = end_time - start_time
        print(f"代码运行时间b: {elapsed_time:.4f} 秒")

        encoded_periods = []

        for dim_idx in range(n):  # 遍历每个维度
            # 获取该维度的最大周期，保证 embedding 不超界
            max_period = int(periods[:, dim_idx].max().item())

            # 创建该维度的 FixedEmbedding
            embedding_layer = FixedEmbedding(c_in=max_period, d_model=self.d_model).to(device)

            # 获取当前维度的周期索引 (b, t)
            periodic_indices_i = periodic_indices[:, :, dim_idx].long()

            # 进行周期编码 (b, t, d_model)
            encoded_dim = embedding_layer(periodic_indices_i)

            # 存储该维度的编码
            encoded_periods.append(encoded_dim.unsqueeze(2))  # 变成 (b, t, 1, d_model)
        # 组合所有维度，得到 (b, t, n, d_model)
        encoded_periods = torch.cat(encoded_periods, dim=2)

        # 维度上求均值 (b, t, d_model)
        encoded_periods = encoded_periods.mean(dim=2)

        # 记录运行时间
        elapsed_time = time.time() - start_time
        print(f"代码运行时间c: {elapsed_time:.4f} 秒")
        return encoded_periods
        #
        #
         # 对每个batch和每个维度分别进行周期编码
        # for batch_idx in range(b):
        #     # 汇总当前batch中所有维度的周期编码
        #     batch_encoded = []
        #     for dim_idx in range(n):
        #         # 获取该维度的周期
        #         period_n = periods[batch_idx, dim_idx].item()  # 当前维度的周期
        #         period_n = int(period_n)
        #         # 创建 FixedEmbedding 层，为当前batch和维度生成周期编码
        #         embedding_layer = FixedEmbedding(c_in=period_n, d_model=self.d_model).to(device)
        #
        #         # 获取该维度的周期索引
        #         periodic_indices_i = periodic_indices[batch_idx, :, dim_idx].long().to(device)  # (t,)
        #
        #         # 对该维度进行周期编码
        #         embedded_i = embedding_layer(periodic_indices_i)  # (t, d_model)
        #
        #         batch_encoded.append(embedded_i.unsqueeze(1))  # 变成 (t, 1, d_model)
        #
        #     # 组合所有维度，得到 (t, n, d_model)
        #     batch_encoded = torch.cat(batch_encoded, dim=1)
        #
        #     encoded_periods.append(batch_encoded.unsqueeze(0))  # 变成 (1, t, n, d_model)
        #
        # # 组合所有 batch，得到 (b, t, n, d_model)
        # encoded_periods = torch.cat(encoded_periods, dim=0)
        #
        # # 将 n 维度上的所有编码取平均或求和，得到 (b, t, d_model)
        # encoded_periods = encoded_periods.mean(dim=2)
        # end_time = time.time()
        # # 计算并打印运行时间
        # elapsed_time = end_time - start_time
        # print(f"代码运行时间c: {elapsed_time:.4f} 秒")
        # return encoded_periods

