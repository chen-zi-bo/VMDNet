import torch
import torch.nn as nn

from torch_timeseries.nn.decomp import SeriesDecomp

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


# 没有对每一特征都max_k学习 ，多的一个线性层先保留 mlp改为学习到（B，max_k,2） sigmoid修改   sigma初始化方法修改  alpha先去掉   温度改为0.5  fft改为rfft ，带宽惩罚项改为能量

# 后面加一ver 高斯滤波器不再经过训练，就是固定的
class MLPToLearnK(nn.Module):
    # 特征维度   max_k
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLPToLearnK, self).__init__()
        # 设定 MLP 网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 存疑，是否有必要加着一层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim * 2)  # 输出一个标量
        self.max_k = output_dim

    def forward(self, x):
        B, T, N = x.shape
        # 通过 MLP 学习到 k
        # T上求均值变成(B, N) 存疑    忽略了不同特征间的差异                 之后可以改成直接找（B,max_k,N）
        x = x.mean(dim=1)
        # 假设我们现在获得每个特征维度的代表值
        x = F.relu(self.fc1(x))
        # 存疑          这一层可能多余
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        # 求出每个批次每个max_k的logits，每个批次统一
        logits = logits.view(B, self.max_k, 2)
        # logits = torch.sigmoid(logits)
        return logits  # 形状: [B, max_K,2]


class DifferentiableVMD(nn.Module):
    # 输入长度        特征维度
    def __init__(self, signal_length, input_dim, max_K=10):
        super().__init__()
        self.max_K = max_K
        self.N = signal_length

        self.mlp = MLPToLearnK(input_dim, output_dim=max_K)  # 定义你之前的 MLP 网络
        # 可学习的软模态数控制参数（用来生成 gate）
        # self.k_raw = nn.Parameter(torch.tensor(3.0))  # 可学习的模态数

        # 存疑 初始化是否有问题
        # 可学习的中心频率 mu（初始化均匀分布在 [0.05, 0.45]）
        self.mu = nn.Parameter(torch.linspace(0.05, 0.45, max_K))


        self.sigma_raw = nn.Parameter(torch.ones(max_K) * -2.0)


        # self.sigma = nn.Parameter(torch.ones(max_K) * 0.05)

        # 可学习的 alpha，softplus(alpha_raw) 控制强度
        # self.alpha_raw = nn.Parameter(torch.tensor(7.6))
        # alpha问题，存疑，alpha是否还有存在的必要性
        # self.alpha_raw = nn.Parameter(torch.tensor(7.6))

        # 温度也可能需要改变以调整尖锐程度
        self.temperature = 0.5
        # 温度可以随时间变化
        # # 频率轴 [-0.5, 0.5)
        # freqs = torch.fft.fftfreq(self.N)
        # self.register_buffer("freqs", freqs)

    def forward(self, x):
        """
        输入:
            x: [B, T, N] 时序信号
        返回:
            imfs: [B, k_used, T, N]
            x_recon: [B, T, N]
            total_loss: 重建误差 + 惩罚
        """
        B, T, N = x.shape

        # freqs = torch.fft.fftfreq(T, d=1. / T).to(x.device)  # [T]
        # freqs使用存疑，和后面在高斯滤波器的使用上是否是对的
        # fft时的k是代表着频率索引，通过fftfreq是找到真实的频率张量，这里默认采样间隔为1
        # 尽管没有采样率，但他也是相对频率坐标
        freqs = torch.fft.rfftfreq(T).to(x.device)  # 动态生成 [T]
        # freqs=self.freqs
        # x_fft是x在时间维（T）上的快速傅里叶变换（FFT），用于将信号从时域转换到频域。
        x_fft = torch.fft.rfft(x, dim=1)
        # 保证惩罚系数大于0
        # alpha = F.softplus(self.alpha_raw)

        delta_mu = (self.mu[1] - self.mu[0]).detach()
        self.sigma = delta_mu * torch.sigmoid(self.sigma_raw)
        # 存疑

        imf_logits = self.mlp(x)
        # eps = 1e-10  # 用于数值稳定性，避免 log(0)
        # # 正向传播会为0-1
        # imf_selection_mask_list = []
        # for i in range(self.max_K):
        #     # ！！！！！！！！使用gumbel-softmax过程存疑
        #     # 抽取第 i 个 IMF 的“选择”概率
        #     prob_chosen = imf_logits[:, i]  # [B]
        #
        #     # 计算第 i 个 IMF 的“不选择”概率
        #     prob_not_chosen = 1.0 - prob_chosen  # [B]
        #
        #     # 将概率转换回 Logits，以供 Gumbel-softmax 使用
        #     # log(p) 是 Logit 形式，这里要分别计算选择和不选择的 Logit
        #     logit_chosen = torch.log(prob_chosen + eps)  # 避免 log(0)
        #     logit_not_chosen = torch.log(prob_not_chosen + eps)  # 避免 log(0)
        #     # 将“选择”和“不选择”的 Logits 堆叠起来，形成 [B, 2] 的输入
        #     binary_logits = torch.stack([logit_chosen, logit_not_chosen], dim=-1)  # [B, 2]
        #
        #     # 对每个 IMF 独立应用 Gumbel-softmax
        #     # dim=-1 是对这两个类别 [chosen, not_chosen] 进行 softmax
        #     # hard=True 确保前向是二元的，反向是连续的
        #     # [:, :, 0] 抽取选中的概率（即第一个维度，代表选择）
        #     # 传入的logits应该有问题
        #     # 应该传入认为选的logits，然后可以将不选的logits转为0
        #     selected_mask = F.gumbel_softmax(logits=binary_logits, tau=self.temperature, hard=True, dim=-1)[:, 0]  # [B]
        #     # 抽取选中的概率（即第一个维度，代表选择）
        #     imf_selection_mask_list.append(selected_mask.unsqueeze(1))
        #
        # imf_selection_mask = torch.cat(imf_selection_mask_list, dim=1)  # [B, max_K], 包含多个1
        # Gumbel-Softmax
        imf_selection = F.gumbel_softmax(
            imf_logits,
            tau=self.temperature,
            hard=True,
            dim=-1
        )  # [B, max_K, 2]

        # 取“选择”那一维
        imf_selection_mask = imf_selection[..., 0]  # [B, max_K]


        k_cont = imf_selection_mask.sum(dim=1)  # [B]  每个批次每条数据的k值

        # 会获得每个imf激活程度
        imfs_fft = []
        imfs_energy = []
        bandwidth_penalty = 0.0
        imfs_energy_batchwise = []  # 用于收集每个IMF在每个批次中的能量
        eps = 1e-8

        for i in range(self.max_K):
            # 获取中心频率
            mu_i = self.mu[i]

            # 模态带宽
            sigma_i = self.sigma[i].abs() + 1e-4

            # 使用高斯滤波器
            # 这里freqs的使用有问题，应该先取绝对值，正频率和负频率是共轭的，都应该滤波
            filter_i = torch.exp(-0.5 * ((freqs - mu_i) / sigma_i) ** 2).to(x.device)  # [T]

            # broadcast filter: [T, 1] -> [1, T, 1] -> [B, T, N]
            filter_i = filter_i[None, :, None]  # 变成 [1, T, 1] 用于广播
            # 通过对x_fft和filter_i的逐元素相乘，滤波器filter_i作用在输入信号的频域表示上，提取出对应的频率成分。

            filtered_fft = x_fft * filter_i  # [B, T, N]

            # 获得对应的imf
            imf_i = torch.fft.irfft(filtered_fft, dim=1).real.float()  # 在时间维上做逆 FFT

            imfs_energy_batchwise.append(imf_i.pow(2).sum(dim=[1, 2]))

            # soft mask 作用（选择性使用模态）
            # print(soft_mask.shape)

            current_imf_weight = imf_selection_mask[:, i].view(B, 1, 1)  # [B, 1, 1]
            # print(mask_i.shape)
            imf_i = imf_i * current_imf_weight  # 广播乘法 → [B, T, N]
            # [B, 1, T, N]
            # 没选为0的也会append
            imfs_fft.append(imf_i.unsqueeze(1))
            imfs_energy.append((imf_i ** 2).sum())

            # 计算带宽惩罚项
            # 建议修改计算带宽惩罚

            # freqs使用也存在问题，这里我们评判带宽惩罚用的是幅值，还可以用能量，
            # u_hat = filtered_fft.abs().pow(2)  # [B, T, N]
            # num = ((freqs - mu_i) ** 2).view(1, T, 1) * u_hat
            # penalty_i = num.sum(dim=1) / (u_hat.sum(dim=1) + eps)  # [B, N]
            power = filtered_fft.abs().pow(2)  # |U|^2
            num = ((freqs - mu_i) ** 2).view(1, freqs.shape[0], 1) * power
            penalty_i = num.sum(dim=1) / (power.sum(dim=1) + eps)
            bandwidth_penalty += (penalty_i * current_imf_weight.squeeze(1)).mean()

        # 计算每个IMF在每个批次中的能量占比 [B, max_K]
        # 检查是否所有模式都被排除了
        if not imfs_fft or imf_selection_mask.sum() == 0:
            # 如果所有的 IMF 都被 Gumbel-softmax 选为 0，这可能会导致问题
            # 此时可以返回一个零张量或者进行特殊处理
            # 比如，强制选择一个IMF（例如概率最高的那个，或者第一个）
            # 或者抛出错误，让优化器去解决这个问题
            # 在实际训练中，Gumbel-softmax 通常不会让所有输出都是0，除非tau太低或logits非常负
            # 为了避免运行时错误，可以返回零填充的imfs和x_recon
            # 但更合理的做法是让损失函数引导模型至少选择一个IMF
            # 例如，在损失函数中加入一个惩罚项，如果k_cont过低
            print("Warning: All IMFs were selected as zero by Gumbel-softmax.")
            imfs = torch.zeros(B, 1, T, N, device=x.device)  # 至少返回一个维度
            x_recon = torch.zeros_like(x)
        else:
            # 包含不选的imf
            imfs = torch.cat(imfs_fft, dim=1)  # [B, k_active, T, N]
            x_recon = imfs.sum(dim=1)  # [B, T, N]

        # 重构loss
        recon_loss = F.mse_loss(x_recon, x)

        # ---------- Orthogonality penalty ----------
        # 计算正交损失，模态混叠损失值计算
        orth_loss = 0.0
        num_imfs_actual = imfs.shape[1]
        for i in range(num_imfs_actual):
            for j in range(i + 1, num_imfs_actual):
                imf_i = imfs[:, i]  # [B, T, N]
                imf_j = imfs[:, j]  # [B, T, N]

                # 只有当两个IMF都被选择时才计算正交性惩罚
                # 这种方式更直接地反映了在选择后的IMF集合中的正交性
                dot_product = (imf_i * imf_j).sum(dim=[1, 2])  # [B]
                norm_i = imf_i.norm(dim=[1, 2])
                norm_j = imf_j.norm(dim=[1, 2])

                # 避免除以零
                # 过滤掉那些 norm 为0 的情况 (即imf_i或imf_j被完全清零的情况)
                valid_indices = (norm_i > eps) & (norm_j > eps)
                if valid_indices.sum() > 0:
                    orth_term = (dot_product[valid_indices] / (
                            norm_i[valid_indices] * norm_j[valid_indices] + eps)) ** 2
                    orth_loss += orth_term.mean()  # 对有效项求平均
        energy_penalty = 0.0

        total_loss = recon_loss + bandwidth_penalty + 0.25 * orth_loss

        return imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, k_cont


class VMD_Change(nn.Module):

    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        # 四个变量的赋值
        # 例96  96 321
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._build_model()

    # 创建可微VMD模型
    def _build_model(self):
        # 传入输入窗口长度，预测长度，以及特征数
        self.vmd = DifferentiableVMD(signal_length=self.seq_len, input_dim=self.enc_in)

        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)
        # self.model_freq=

    def vmd_loss(self, x):
        imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, k = self.vmd(x)
        return total_loss

    def loss(self, true):
        # freq normalization
        B, O, N = true.shape
        # 把true分开
        imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, k = self.vmd(true)

        pred_main = imfs.sum(dim=1)  # [B, T, N]

        residual = true - pred_main

        lf = nn.functional.mse_loss
        # 计算前k个频率的loss和残差部分的损失
        return lf(self.pred_main_signal, pred_main) + lf(residual, self.pred_residual)

    # 对输入x进行规范化
    def normalize(self, input):
        # (B, T, N)
        # 获取输入数据的批次数量、序列长度以及维度
        bs, len, dim = input.shape

        imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, k = self.vmd(input)

        main_sum = imfs.sum(dim=1)  # [B, T, N]
        # if self.visual:
        #     plot_and_save_vmd_results(input, imfs, main_imfs, main_sum)

        norm_input = input - main_sum

        # 转换1 2 维
        # （B,N,T）                                          再转回去
        # 接收通过前k个频率的预测结果
        self.pred_main_signal = self.model_freq(main_sum.transpose(1, 2), input.transpose(1, 2)).transpose(1, 2)
        # 返回的是去除前k个主要频率的输入x
        return norm_input.reshape(bs, len, dim)

    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_signal
        return output.reshape(bs, len, dim)

    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x)


class MLPfreq(nn.Module):
    # 例如传入96 96 321
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        # 变量的赋值
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        # model_freq该模块包括一个全连接层以及一个激活函数
        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )

        # model_all该模块包括一个全连接层，一个激活函数以及最后的输出层
        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )
        self.linear = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    # 前向传播
    def forward(self, main_freq, x):
        # （batch_size, channels, 64)
        # return self.linear(main_freq)
        self.model_freq(main_freq)
        # # self.model_freq的输出形状是[batch_size, 64]，x原本数据
        # # 最终输出的张量inp的形状是[batch_size, 64 + seq_len]
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        # # 最后使用model_all进行预测
        # # 然后进过该模块获得# （batch_size, channels, pred_len)
        return self.model_all(inp)


class VMD_DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, seq_len, pred_len, enc_in, individual: bool = False):
        super(VMD_DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.vmd = VMD_Change(seq_len=seq_len, pred_len=self.pred_len, enc_in=enc_in)

        self.linear = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        x = self.vmd.normalize(x)
        # x: [Batch, Input length, Channel]
        # x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        # pred = self.vmd.denormalize(x)

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        pred = self.vmd.denormalize(x.permute(0, 2, 1))
        return pred  # to [Batch, Output length, Channel]





