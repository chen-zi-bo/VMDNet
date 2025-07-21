import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_timeseries.nn.AutoCorrelation import AutoCorrelationLayer
from torch_timeseries.nn.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, \
    series_decomp, series_decomp_multi
# from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from torch_timeseries.nn.FourierCorrelation import FourierBlock, FourierCrossAttention
from torch_timeseries.nn.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from torch_timeseries.nn.embedding import DataEmbedding_wo_pos

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, enc_in, dec_in, seq_len, pred_len, label_len, c_out, version='Fourier', L=3, d_ff=2048,
                 activation='gelu', e_layers=2, d_layers=1, mode_select='random', modes=64, output_attention=True,
                 moving_avg=[24], n_heads=8, cross_activation='tanh', d_model=512, embed='timeF', freq='h', dropout=0.0,
                 base='legendre'):
        super(FEDformer, self).__init__()

        # 部分参数值
        # version: str = 'Fourier'
        # L: int = 3
        # d_ff: int = 2048
        # activation: str = 'gelu'
        # e_layers: int = 2
        # d_layers: int = 1
        # mode_select: str = 'random'
        # modes: int = 64
        # output_attention: bool = True
        # moving_avg: list = field(default_factory=lambda: [24])
        # n_heads: int = 8
        # cross_activation: str = 'tanh'
        # d_model: int = 512
        # embed: str = 'timeF'
        # freq: str = 'h'
        # dropout: float = 0.0
        # base: str = 'legendre'
        # l2_weight_decay: float = 0.0
        # 将输入的参数先进行赋值

        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        # 这个模块求移动平均，返回一个减去移动平均之后的值和移动平均值
        # 这个模块返回给decomp
        if isinstance(kernel_size, list):
            # 如果是列表
            self.decomp = series_decomp_multi(kernel_size)
        else:
            # 如果不是
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.

        # 设置编码器和解码器的嵌入层,为DataEmbedding_wo_pos模块
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq,
                                                  dropout)

        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq,
                                                  dropout)

        # 有两种类别，一种是小波变换，一种是傅里叶变换,默认我们使用的是傅里叶变换的内容

        if version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
            decoder_cross_att = MultiWaveletCross(in_channels=d_model,
                                                  out_channels=d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=modes,
                                                  ich=d_model,
                                                  base=base,
                                                  activation=cross_activation)


        else:

            # 该模块实现频域上特征学习
            # 512 512 96   64, random
            encoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len,
                                            modes=modes,
                                            mode_select_method=mode_select)

            # 第二个seq_len有所不同
            decoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=modes,
                                            mode_select_method=mode_select)

            # 跨注意力特征学习
            decoder_cross_att = FourierCrossAttention(in_channels=d_model,
                                                      out_channels=d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=modes,
                                                      mode_select_method=mode_select)

        # Encoder
        # 求编码器和解码器的模式

        enc_modes = int(min(modes, seq_len // 2))
        dec_modes = int(min(modes, (seq_len // 2 + pred_len) // 2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        # 构建编码器模块
        self.encoder = Encoder(
            [ #有e_layers个EncoderLayer层
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        d_model, n_heads),

                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],

            norm_layer=my_Layernorm(d_model)
        )


        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 一个自注意力
                    AutoCorrelationLayer(
                        decoder_self_att,
                        d_model, n_heads),
                    # 一个交叉注意力
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],

            norm_layer=my_Layernorm(d_model),
            # 线性层，实现投影
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # 创建一个均值初始值   (B,pred_len,N)形状的
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        # [batch_size, pred_len, channels]的一个全零张量  放到设备里
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        # 通过池化获得初始的季节值和趋势值
        seasonal_init, trend_init = self.decomp(x_enc)

        # 解码器的趋势以及季节初始化输入
        # decoder input
        # 最后的那几个trend和mean连起来 [batch_size, label_len + pred_len, channels]
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # [batch_size, label_len + pred_len, channels] 后面填充为0
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # x_enc（输入数据）通过卷积操作嵌入到d_model维度空间。
        # x_mark_enc（时间标记）通过全连接层嵌入到d_model维度空间。
        # 然后两者相加，得到包含时间信息的特征表示。
        # enc
        # [batch_size, seq_len, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 将输入传入编码器层，按照FEDformer的架构输出编码器层的结果，对输入反复自注意过
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # seasonal_init（输入数据）通过卷积操作嵌入到d_model维度空间。
        # x_mark_enc（时间标记）通过全连接层嵌入到d_model维度空间。
        # 然后两者相加，得到包含时间信息的特征表示。
        # 解码器的输入
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # 通过解码器层计算出季节部分和趋势部分
        # 传入一开始的初始趋势
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # 获得最后结果
        # final
        dec_out = trend_part + seasonal_part

        # 只输出最后要的预测部分的数据
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    import sys

    print(sys.path)
#     class Configs(object):
#         ab = 0
#         modes = 32
#         mode_select = 'random'
#         # version = 'Fourier'
#         version = 'Wavelets'
#         moving_avg = [12, 24]
#         L = 1
#         base = 'legendre'
#         cross_activation = 'tanh'
#         seq_len = 96
#         label_len = 48
#         pred_len = 96
#         output_attention = True
#         enc_in = 7
#         dec_in = 7
#         d_model = 16
#         embed = 'timeF'
#         dropout = 0.05
#         freq = 'h'
#         factor = 1
#         n_heads = 8
#         d_ff = 16
#         e_layers = 2
#         d_layers = 1
#         c_out = 7
#         activation = 'gelu'
#         wavelet = 0

#     configs = Configs()
#     model = Model(configs)

#     print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
#     enc = torch.randn([3, seq_len, 7])
#     enc_mark = torch.randn([3, seq_len, 4])

#     dec = torch.randn([3, seq_len//2+pred_len, 7])
#     dec_mark = torch.randn([3, seq_len//2+pred_len, 4])
#     out = model.forward(enc, enc_mark, dec, dec_mark)
#     print(out)
