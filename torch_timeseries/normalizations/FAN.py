import torch
import torch.nn as nn


def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()

    if rfft:
        # 默认使用rfft对输入x傅里叶变换，按着维度去变换
        # xf(B, T // 2 + 1, N)   每个元素是一个复数
        xf = torch.fft.rfft(x, dim=1)

    else:
        xf = torch.fft.fft(x, dim=1)
    # 返回按幅度大小排序的前k个元素
    k_values = torch.topk(xf.abs(), k, dim=1)

    # 通过indice即可获得前k个元素 (B, k, N)
    indices = k_values.indices
    # 和xf相同形状的零张量，之后其实就是作为面罩过滤的作用
    mask = torch.zeros_like(xf)
    # 根据上面获得的索引填充1
    mask.scatter_(1, indices, 1)
    # 相乘，即可过滤xf，只获得前k个频率成分
    xf_filtered = xf * mask

    if rfft:
        # 如果用的是rfft，然后把前面获得的前k个频率成分再转换回去
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()
    # 然后计算出过滤掉前k个频率成分之后的x
    norm_input = x - x_filtered

    # print(f"decompose take:{ time.time() - start} s")

    return norm_input, x_filtered


class FAN(nn.Module):
    """FAN first substract bottom k frequecy component from the original series
      

    Args:
        nn (_type_): _description_
    """

    # 生成一个FAN对象赋给n_model    当前例子      96     96 168 336 720各一次     数据集特征数 每个数据集自己定义好了，norm配置{freq_topk:4}
    # 构造函数传入           98      其中一个96   特征数321  默认所要过滤的前20个，这里注意实际并没有通过接收命令行传入的freq_topk去配置
    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, rfft=True, **kwargs):
        # rfft为使用FFT的实数版本，只返回一半
        super().__init__()
        # 四个变量的赋值
        # 例96  96 321
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        self.freq_topk = freq_topk
        self.rfft = rfft
        # 很小量，防除零
        self.epsilon = 1e-8

        print("freq_topk : ", self.freq_topk)

        self._build_model()
        # 给该模型添加了训练参数，应该是权重 ，形状为 [2, enc_in] 的张量
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    # 创建一个模型，简单的MLP
    def _build_model(self):
        # 传入输入窗口长度，预测长度，以及特征数
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)


    def loss(self, true):
        # freq normalization
        B, O, N = true.shape
        # 把true分开
        residual, pred_main = main_freq_part(true, self.freq_topk, self.rfft)

        lf = nn.functional.mse_loss
        # 计算前k个频率的loss和残差部分的损失
        return lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual)

    # 对输入x进行规范化
    def normalize(self, input):
        # (B, T, N)
        # 获取输入数据的批次数量、序列长度以及维度
        bs, len, dim = input.shape
        # 调用main_freq_part  传入输入数据
        # 把x分成了两部分，一个是前k个频率成分
        norm_input, x_filtered = main_freq_part(input, self.freq_topk, self.rfft)
                                                    # 转换1 2 维
                                                    # （B,N,T）                                          再转回去
        # 接收通过前k个频率的预测结果
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1, 2), input.transpose(1, 2)).transpose(1, 2)
        # 返回的是去除前k个主要频率的输入x
        return norm_input.reshape(bs, len, dim)

    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

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

    # 前向传播
    def forward(self, main_freq, x):
        # 对前k个频率简单MLP
        # （batch_size, channels, 64)
        self.model_freq(main_freq)
        # self.model_freq的输出形状是[batch_size, 64]，x原本数据
        # 最终输出的张量inp的形状是[batch_size, 64 + seq_len]
        # 学习后的和去掉前k个的连接在一起
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        # 最后使用model_all进行预测
        # 然后进过该模块获得# （batch_size, channels, pred_len)
        # 把前k个作为学习目标，根据学习前k个去预测
        return self.model_all(inp)
