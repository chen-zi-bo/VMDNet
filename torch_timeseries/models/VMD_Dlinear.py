import torch
import torch.nn as nn

from torch_timeseries.nn.decomp import SeriesDecomp

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class MLPToLearnK(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLPToLearnK, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim * 2)  
        self.max_k = output_dim

    def forward(self, x):
        B, T, N = x.shape
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        logits = logits.view(B, self.max_k, 2)
        # logits = torch.sigmoid(logits)
        return logits  


class DifferentiableVMD(nn.Module):

    def __init__(self, signal_length, input_dim, max_K=10):
        super().__init__()
        self.max_K = max_K
        self.N = signal_length

        self.mlp = MLPToLearnK(input_dim, output_dim=max_K) 

        self.mu = nn.Parameter(torch.linspace(0.05, 0.45, max_K))


        self.sigma_raw = nn.Parameter(torch.ones(max_K) * -2.0)



        self.temperature = 0.5


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

        freqs = torch.fft.rfftfreq(T).to(x.device) 

        x_fft = torch.fft.rfft(x, dim=1)


        delta_mu = (self.mu[1] - self.mu[0]).detach()
        self.sigma = delta_mu * torch.sigmoid(self.sigma_raw)
        

        imf_logits = self.mlp(x)
        imf_selection = F.gumbel_softmax(
            imf_logits,
            tau=self.temperature,
            hard=True,
            dim=-1
        )  # [B, max_K, 2]


        imf_selection_mask = imf_selection[..., 0]  


        k_cont = imf_selection_mask.sum(dim=1)  


        imfs_fft = []
        imfs_energy = []
        bandwidth_penalty = 0.0
        imfs_energy_batchwise = [] 
        eps = 1e-8

        for i in range(self.max_K):

            mu_i = self.mu[i]


            sigma_i = self.sigma[i].abs() + 1e-4


            filter_i = torch.exp(-0.5 * ((freqs - mu_i) / sigma_i) ** 2).to(x.device)  # [T]

            # broadcast filter: [T, 1] -> [1, T, 1] -> [B, T, N]
            filter_i = filter_i[None, :, None] 


            filtered_fft = x_fft * filter_i  # [B, T, N]


            imf_i = torch.fft.irfft(filtered_fft, dim=1).real.float()

            imfs_energy_batchwise.append(imf_i.pow(2).sum(dim=[1, 2]))



            current_imf_weight = imf_selection_mask[:, i].view(B, 1, 1)  # [B, 1, 1]
            # print(mask_i.shape)
            imf_i = imf_i * current_imf_weight 
            # [B, 1, T, N]

            imfs_fft.append(imf_i.unsqueeze(1))
            imfs_energy.append((imf_i ** 2).sum())


            power = filtered_fft.abs().pow(2)  # |U|^2
            num = ((freqs - mu_i) ** 2).view(1, freqs.shape[0], 1) * power
            penalty_i = num.sum(dim=1) / (power.sum(dim=1) + eps)
            bandwidth_penalty += (penalty_i * current_imf_weight.squeeze(1)).mean()


        if not imfs_fft or imf_selection_mask.sum() == 0:
            print("Warning: All IMFs were selected as zero by Gumbel-softmax.")
            imfs = torch.zeros(B, 1, T, N, device=x.device) 
            x_recon = torch.zeros_like(x)
        else:

            imfs = torch.cat(imfs_fft, dim=1)  # [B, k_active, T, N]
            x_recon = imfs.sum(dim=1)  # [B, T, N]


        recon_loss = F.mse_loss(x_recon, x)


        orth_loss = 0.0
        num_imfs_actual = imfs.shape[1]
        for i in range(num_imfs_actual):
            for j in range(i + 1, num_imfs_actual):
                imf_i = imfs[:, i]  # [B, T, N]
                imf_j = imfs[:, j]  # [B, T, N]


                dot_product = (imf_i * imf_j).sum(dim=[1, 2])  # [B]
                norm_i = imf_i.norm(dim=[1, 2])
                norm_j = imf_j.norm(dim=[1, 2])


                valid_indices = (norm_i > eps) & (norm_j > eps)
                if valid_indices.sum() > 0:
                    orth_term = (dot_product[valid_indices] / (
                            norm_i[valid_indices] * norm_j[valid_indices] + eps)) ** 2
                    orth_loss += orth_term.mean()  
        energy_penalty = 0.0

        total_loss = recon_loss + bandwidth_penalty + 0.25 * orth_loss

        return imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, k_cont


class VMD_Change(nn.Module):

    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self._build_model()


    def _build_model(self):

        self.vmd = DifferentiableVMD(signal_length=self.seq_len, input_dim=self.enc_in)

        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)


    def vmd_loss(self, x):
        imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, k = self.vmd(x)
        return total_loss

    def loss(self, true):

        B, O, N = true.shape

        imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, k = self.vmd(true)

        pred_main = imfs.sum(dim=1)  # [B, T, N]

        residual = true - pred_main

        lf = nn.functional.mse_loss

        return lf(self.pred_main_signal, pred_main) + lf(residual, self.pred_residual)


    def normalize(self, input):

        bs, len, dim = input.shape

        imfs, x_recon, total_loss, recon_loss, bandwidth_penalty, orth_loss, energy_penalty, k = self.vmd(input)

        main_sum = imfs.sum(dim=1)  # [B, T, N]


        norm_input = input - main_sum


        self.pred_main_signal = self.model_freq(main_sum.transpose(1, 2), input.transpose(1, 2)).transpose(1, 2)

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

    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )


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

        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)

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





