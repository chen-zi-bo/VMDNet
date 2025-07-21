import time

import numpy as np
import torch
from PyEMD import EMD
# 第一部分内容：
# 通过EMD首先分成之后要用的IMF和趋势项。

# 有一个问题，如果通过EMD分解后的IMF数量（除去趋势项）比k小时，这时候energy的数据维度不一致了
#                 将趋势项通过fft分解获取对应数据
from torch import nn
from vmdpy import VMD

from torch_timeseries.models.Testformer import Testformer


# def emd_decompose(batch_x, k_user_defined, device='cuda'):
#     # 如果归一化之后
#     # print(batch_x)
#     # avg = torch.mean(batch_x, axis=1, keepdim=True).detach()  # b*1*d
#     # var = torch.var(batch_x, axis=1, keepdim=True).detach()  # b*1*d
#     # epsilon = 1e-6  # 避免除以零
#     # batch_x = (batch_x - avg) / torch.sqrt(var + epsilon)
#     # print(batch_x)
#     B, T, N = batch_x.shape  # B: batch size, T: 长度, N: 维度
#     emd = EMD()
#
#     # 存储最后多任务的输入
#     final_input = [[] for _ in range(k_user_defined + 1)]
#     for b in range(B):  # 遍历 batch 中的每个样本
#         sample_imfs = []  # 存储该样本选出的 IMF
#         # imf_counts = torch.zeros(N, dtype=torch.int32, device=device)
#         for n in range(N):
#             signal = batch_x[b, :, n].cpu().numpy()  # 将 signal 转换为 NumPy 数组
#             imfs = emd(signal)  # 进行 EMD 分解，返回 (M, T) 形式的 IMF
#             imfs = torch.tensor(imfs, dtype=torch.float32, device=device)  # 转为 GPU 张量
#             # print("begin")
#             # print(imfs.shape)
#             # print(imfs)
#             # reconstructed_signals存储补齐的时序数据
#             reconstructed_signals = []
#             # 如果某一维IMF（除趋势项）个数小于k
#             # print(imfs.shape[0])
#             if imfs.shape[0] - 1 < k_user_defined:
#                 trend = imfs[-1, :]
#                 # print(trend.shape)
#                 fft_result = torch.fft.rfft(trend)  # 进行 FFT 变换
#                 # print(fft_result)
#                 comp_values = torch.topk(fft_result.abs(), k_user_defined - (imfs.shape[0] - 1))
#                 # print(comp_values)
#                 # 获得要补齐的频率的索引
#                 indices = comp_values.indices
#
#                 # 获取剩余的频率成分（除去前k个）
#                 mask = torch.ones_like(fft_result)
#                 mask[indices] = 0  # 将前k个频率成分的索引位置置为0
#                 remaining_freqs = fft_result * mask  # 剩余的频率成分
#                 # print(remaining_freqs)
#                 # 对剩余频率成分进行傅里叶逆变换，恢复时域信号
#                 remaining_signal = torch.fft.irfft(remaining_freqs, n=trend.shape[0]).real.float()
#                 # print(remaining_signal)
#                 for idx in indices:
#                     # 创建一个全零的复数张量
#                     single_freq = torch.zeros_like(fft_result)
#                     # 仅保留当前频率成分
#                     single_freq[idx] = fft_result[idx]
#                     # 进行傅里叶逆变换，恢复时域信号
#                     single_signal = torch.fft.irfft(single_freq, n=trend.shape[0]).real.float()
#                     reconstructed_signals.append(single_signal)
#
#                     # 接下来重新编排imfs
#                 imfs_without_trend = imfs[:-1, :]  # 去掉最后一个趋势项
#                 # print(imfs_without_trend.shape)
#
#                 reconstructed_signals = torch.stack(reconstructed_signals, dim=0)  # 将多个时序信号堆叠成张量
#                 # print(reconstructed_signals.shape)
#                 remaining_signal = remaining_signal.view(1, -1)  # 转为张量
#                 # print(remaining_signal)
#                 # print(remaining_signal.shape)
#                 final_imfs = torch.cat((imfs_without_trend, reconstructed_signals, remaining_signal), dim=0)
#             # print(final_imfs[0])
#             else:
#                 final_imfs = imfs
#             sample_imfs.append(final_imfs)
#             # print(final_imfs)
#             # print("end")
#         selected_imfs = []
#         for n in range(N):
#             imfs = sample_imfs[n]  # 该维度的所有 IMF，形状 (M, T)
#             # 分离趋势项
#             trend_imf = imfs[-1:].clone()  # 取最后一个 IMF 作为趋势项
#             imfs_without_trend = imfs[:-1]  # 其余 IMF 作为备选 IMF
#             # 计算非趋势 IMF 的能量（平方和）
#             energies = torch.sum(imfs_without_trend ** 2, dim=1)  # 形状 (M-1,)
#             # **找出前 k 个最大能量 IMF**
#             top_k_indices = torch.argsort(energies, descending=True)[:k_user_defined]  # 获取索引
#             top_k_imfs = imfs_without_trend[top_k_indices]  # 选取对应 IMF
#             # 合并最终选择的 IMF（先高能 IMF，最后趋势项）
#             final_imfs = torch.cat([top_k_imfs, trend_imf], dim=0)  # 仍在 GPU 上
#             # print(final_imfs.shape)
#             selected_imfs.append(final_imfs)
#         # print(selected_imfs.shape)
#         # 接下来我们应该转成k+1个，实现分别学习。
#         energy_groups = [[] for _ in range(k_user_defined + 1)]  # 创建一个列表来存储不同能量级别的 IMF
#         for n in range(N):
#             imfs = selected_imfs[n]  # 该维度的所有 IMF，形状 (M, T)
#             # 直接将已排序的 IMF 分到能量组中
#             for idx, imf in enumerate(imfs):  # 包括趋势项
#                 energy_groups[idx].append(imf)  # 按能量级别分类存储
#         for i in range(len(energy_groups)):
#             energy_groups[i] = torch.stack(energy_groups[i], dim=0)  # (N, T)
#         # 现在实现energy_group有k+1个，每一个存对应位置能量的IMF的每个维度的内容
#         # 即第一个存的就是每个维度能量最大的IMF，之后将它进行预测所要预测区间的维度能量最大的IMF区间。作为一个任
#         # 现在实现了有k+1个所要预测的数据
#         for i in range(len(energy_groups)):
#             final_input[i].append(energy_groups[i])
#
#     for i in range(len(final_input)):
#         final_input[i] = torch.stack(final_input[i], dim=0)
#
#     final_input = torch.stack(final_input, dim=0)
#
#     # final_input (k+1,B,Dimension,Length)
#     input_x = final_input.transpose(2, 3)
#     # input_x(k+1,B,Length,Dimension)
#     print(input_x.shape)
#     return input_x

# def vmd_decompose(batch_x, k_user_defined, device='cuda', alpha=2000, tau=0, DC=0, init=1, tol=1e-6):
#     """
#     使用 VMD 进行时间序列分解，替换 EMD
#     :param batch_x: 输入数据，形状 (B, T, N) (batch_size, length, dimensions)
#     :param k_user_defined: 选取的前 K 个高能 IMF
#     :param device: 计算设备（默认 GPU）
#     :param alpha: VMD 平滑因子
#     :param tau: 终止阈值
#     :param DC: 是否保留 DC 分量
#     :param init: 初始模式选择方式
#     :param tol: 计算收敛阈值
#     :return: 分解后的时序数据 (k+1, B, Length, Dimension)
#     """
#     B, T, N = batch_x.shape  # B: batch size, T: 时序长度, N: 维度
#     final_input = [[] for _ in range(k_user_defined + 1)]
#
#     for b in range(B):  # 遍历 batch 中的每个样本
#         sample_imfs = []  # 存储该样本选出的 IMF
#         for n in range(N):
#             signal = batch_x[b, :, n].cpu().numpy()  # 转换为 NumPy
#             # 进行 VMD 分解，返回 U (K, T)
#             imfs, _, _ = VMD(signal, alpha, tau, k_user_defined + 1, DC, init, tol)
#             imfs = torch.tensor(imfs, dtype=torch.float32, device=device)  # 转为 GPU 张量
#             sample_imfs.append(imfs)
#
#         selected_imfs = []
#         for n in range(N):
#             imfs = sample_imfs[n]  # (K, T)
#             energies = torch.sum(imfs ** 2, dim=1)  # 计算能量
#             top_k_indices = torch.argsort(energies, descending=True)[:k_user_defined]  # 取前 k_user_defined 个高能 IMF
#             top_k_imfs = imfs[top_k_indices]  # 选取对应 IMF
#             trend_imf = imfs[-1:].clone()  # 取趋势项
#             final_imfs = torch.cat([top_k_imfs, trend_imf], dim=0)  # 组合成最终结果
#             selected_imfs.append(final_imfs)
#
#         # 组织数据到 k+1 组
#         energy_groups = [[] for _ in range(k_user_defined + 1)]
#         for n in range(N):
#             imfs = selected_imfs[n]
#             for idx, imf in enumerate(imfs):
#                 energy_groups[idx].append(imf)
#         for i in range(len(energy_groups)):
#             energy_groups[i] = torch.stack(energy_groups[i], dim=0)
#
#         for i in range(len(energy_groups)):
#             final_input[i].append(energy_groups[i])
#
#     for i in range(len(final_input)):
#         final_input[i] = torch.stack(final_input[i], dim=0)
#     final_input = torch.stack(final_input, dim=0)
#
#     input_x = final_input.transpose(2, 3)  # (k+1, B, Length, Dimension)
#     # print(input_x.shape)
#     return input_x
def vmd_decompose(batch_x, k_user_defined, device='cuda', alpha=2000, tau=0, DC=0, init=1, tol=1e-6):
    """
    使用 VMD 进行时间序列分解，替换 EMD
    :param batch_x: 输入数据，形状 (B, T, N) (batch_size, length, dimensions)
    :param k_user_defined: 选取的前 K 个高能 IMF
    :param device: 计算设备（默认 GPU）
    :param alpha: VMD 平滑因子
    :param tau: 终止阈值
    :param DC: 是否保留 DC 分量
    :param init: 初始模式选择方式
    :param tol: 计算收敛阈值
    :return: 分解后的时序数据 (k+1, B, Length, Dimension)
    """
    B, T, N = batch_x.shape  # B: batch size, T: 时序长度, N: 维度
    K = k_user_defined + 1  # 选取的 IMF 数量

    # 预分配存储空间 (k+1, B, T, N)
    final_input = torch.zeros((K, B, T, N), dtype=torch.float32, device=device)

    eps = 1e-8  # 避免除零错误的小数

    for b in range(B):  # 遍历 batch 中的每个样本
        for n in range(N):  # 遍历每个维度
            signal = batch_x[b, :, n].cpu().numpy()  # 转换为 NumPy

            # 进行 VMD 分解，返回 U (K, T)
            imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)  # 直接获取 K 个 IMF

            # **避免 NaN 和 Inf**
            imfs = np.nan_to_num(imfs, nan=0.0, posinf=1.0, neginf=-1.0)  # 替换 NaN 和 Inf
            imfs = torch.tensor(imfs, dtype=torch.float32, device=device)  # 转为 GPU 张量

            # 计算 IMF 能量
            energies = torch.sum(imfs ** 2 + eps, dim=1)  # (K,) 避免除零错误
            top_k_indices = torch.argsort(energies, descending=True)[:k_user_defined]  # 取前 k_user_defined 个高能 IMF

            # 选择高能 IMF + 趋势项
            top_k_imfs = imfs[top_k_indices]  # (k_user_defined, T)
            trend_imf = imfs[-1:].clone()  # 取趋势项 (1, T)
            final_imfs = torch.cat([top_k_imfs, trend_imf], dim=0)  # (k+1, T)

            # 存储到 final_input
            final_input[:, b, :, n] = final_imfs

    return final_input  # (k+1, B, Length, Dimension)

def vmd_decompose(batch_x, k_user_defined, device='cuda', alpha=2000, tau=0, DC=0, init=1, tol=1e-6):
    """
    使用 VMD 进行时间序列分解，替换 EMD
    :param batch_x: 输入数据，形状 (B, T, N) (batch_size, length, dimensions)
    :param k_user_defined: 选取的前 K 个高能 IMF
    :param device: 计算设备（默认 GPU）
    :param alpha: VMD 平滑因子
    :param tau: 终止阈值
    :param DC: 是否保留 DC 分量
    :param init: 初始模式选择方式
    :param tol: 计算收敛阈值
    :return: 分解后的时序数据 (k+1, B, Length, Dimension)
    """
    B, T, N = batch_x.shape  # B: batch size, T: 时序长度, N: 维度
    K = k_user_defined + 1  # 选取的 IMF 数量

    # 预分配存储空间 (k+1, B, T, N)
    final_input = torch.zeros((K, B, T, N), dtype=torch.float32, device=device)

    # 将数据从 (B, T, N) 转换为 (B * N, T)，以便批量处理
    batch_x_reshaped = batch_x.view(-1, T).cpu().numpy()

    # 执行 VMD 分解，并且一次性处理所有信号
    imfs_list = []
    for signal in batch_x_reshaped:
        imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)  # 获取所有 IMF
        imfs_list.append(imfs)

    # 将所有 IMF 结果转为张量 (B * N, K, T)
    imfs_tensor = torch.tensor(np.array(imfs_list), dtype=torch.float32, device=device)

    # 计算每个 IMF 的能量 (B * N, K)
    energies = torch.sum(imfs_tensor ** 2, dim=2)  # (B * N, K)

    # 获取每个信号的前 k_user_defined 个高能 IMF
    top_k_indices = torch.argsort(energies, descending=True)[:, :k_user_defined]  # (B * N, k_user_defined)

    # 构造最终的 IMF 结果 (B * N, k_user_defined, T)
    top_k_imfs = torch.gather(imfs_tensor, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, T))  # (B * N, k_user_defined, T)

    # 取趋势项（最后一个 IMF）
    trend_imf = imfs_tensor[:, -1:, :]  # (B * N, 1, T)

    # 合并高能 IMF 和趋势项 (B * N, k_user_defined + 1, T)
    final_imfs = torch.cat([top_k_imfs, trend_imf], dim=1)

    # 将 (B * N, k_user_defined + 1, T) 转换为 (k+1, B, T, N)
    final_input = final_imfs.view(K, B, T, N)

    return final_input  # (k+1, B, T, N)

# 接下来实现多任务学习
# 整体多任务学习模型
class MultiTaskEMDLearning(nn.Module):

    def __init__(self, seq_len, pred_len, enc_in, dec_in, label_len, c_out, hidden_dim=64, d_ff=2048,
                 activation='gelu', e_layers=2, d_layers=1, output_attention=True, factor=5,
                 n_heads=8, d_model=512, embed='timeF', freq='h', dropout=0.0, attn='prob'):
        super(MultiTaskEMDLearning, self).__init__()
        # 初始化两个预测模型
        self.mlp_model = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len)  # 输出维度 = 输入维度
        )
        self.label_len = label_len
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.transformer = Testformer(
            # 根据数据集的特征数初始化编码器和解码器的输入
            enc_in=enc_in,
            dec_in=dec_in,
            # 输入序列长度
            seq_len=seq_len,
            # 预测长度
            pred_len=pred_len,
            # 标签长度
            label_len=label_len,
            # 输出特征的数量
            c_out=c_out,

            # 将之前类的变量作为参数传入
            d_ff=d_ff,
            activation=activation,
            e_layers=e_layers,
            d_layers=d_layers,
            output_attention=output_attention,
            n_heads=n_heads,
            d_model=d_model,
            embed=embed,
            freq=freq,
            dropout=dropout,
            factor=factor,
            attn=attn
        )

    def forward(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, k_user_defined, device):
        # 记录开始时间
        # start_time = time.time()
        # input_x = emd_decompose(batch_x, k_user_defined, device)
        k_plus_1 = k_user_defined + 1  # k+1 个相同的 batch_x

        # input_x = batch_x.unsqueeze(0).repeat(k_plus_1, 1, 1, 1)  # 变为 (k+1, b, t, n)
        input_x = vmd_decompose(batch_x, k_user_defined, device)
        # end_time = time.time()
        # 计算并打印运行时间
        # elapsed_time = end_time - start_time
        # print(f"代码运行时间a: {elapsed_time:.4f} 秒")
        k_plus_1, B, Length, Dimension = input_x.shape
        task_outputs = []

        for task_id in range(k_plus_1):
            # print(f"训练任务 {task_id + 1}/{k_plus_1}")

            task_input = input_x[task_id]  # Shape: (B, Length, Dimension)

            if task_id == k_plus_1 - 1 and self.mlp_model:  # 处理趋势项
                # (B, Length, Dimension)
                task_input_tensor = task_input.float()  # 不需要使用torch.tensor
                # b,l,d 转换之后通过mlp求完之后又转回去
                task_output = self.mlp_model(task_input_tensor.transpose(1, 2)).transpose(1, 2)
                # print(task_output.shape)
            else:
                # 其他任务使用主模型训练
                task_output = self.transformer(task_input, batch_y, batch_x_date_enc, batch_y_date_enc)[0]

                # print(task_output.shape)
            task_outputs.append(task_output)

        # 直接合成
        # 进行 EMD 合成
        final_output = torch.sum(torch.stack(task_outputs, dim=0), dim=0)
        # print(final_output.shape)
        return final_output


# 再实现一个当获得预测结果，即相当于


if __name__ == '__main__':
    import numpy as np

    batch_x = torch.tensor([[[15, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0],
                             [10.0, 11.0, 12.0],
                             [13.0, 14.0, 15.0]], [[15, 2.0, 3.0],
                                                   [4.0, 5.0, 6.0],
                                                   [7.0, 8.0, 9.0],
                                                   [10.0, 11.0, 12.0],
                                                   [13.0, 14.0, 15.0]]], dtype=torch.float32)
    # 确保 batch_x 在 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_x = batch_x.to(device)
    print(batch_x.shape)

    # 假设 emd_decompose 函数已定义
    input_x = vmd_decompose(batch_x, 3)
    k_plus_1, B, Length, Dimension = input_x.shape
    task_outputs = []

    for task_id in range(k_plus_1):
        print(f"训练任务 {task_id + 1}/{k_plus_1}")

        task_input = input_x[task_id].to(device)  # 确保 task_input 在正确的设备上

        mlp_model = nn.Sequential(
            nn.Linear(batch_x.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)  # 把模型放到 batch_x 的设备上

        task_output = mlp_model(task_input.transpose(1, 2)).transpose(1, 2)
        task_outputs.append(task_output)
        # print(task_output.shape)
    #     if task_id == k_plus_1 - 1 and self.mlp_model:  # 处理趋势项
    #         # (B, Length, Dimension)
    #         task_input_tensor = task_input.float()  # 不需要使用torch.tensor
    #         print(task_input_tensor.shape)
    #         # b,l,d 转换之后通过mlp求完之后又转回去
    #         task_output = self.mlp_model(task_input_tensor.transpose(1, 2)).transpose(1, 2)
    #     else:
    #         # 其他任务使用主模型训练
    #         task_output = self.transformer(task_input, batch_y, batch_x_date_enc, batch_y_date_enc)[0]
    #
    #     task_outputs.append(task_output)
    #
    # # 直接合成
    # # 进行 EMD 合成
    final_output = torch.sum(torch.stack(task_outputs, dim=0), dim=0)
    print(final_output.shape)
