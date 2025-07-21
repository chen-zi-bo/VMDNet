# transformer模型
import torch
from torch import nn

from torch_timeseries.models.MultiTaskEMDLearning_transformer_part import DataEmbedding_cycle_pos
from torch_timeseries.nn.AutoCorrelation import AutoCorrelationLayer
from torch_timeseries.nn.attention import FullAttention, ProbAttention
from torch_timeseries.nn.decoder import Decoder, DecoderLayer
from torch_timeseries.nn.encoder import Encoder, EncoderLayer


class Testformer(nn.Module):
    # 编码解码特征数和长度
    def __init__(self, enc_in, dec_in, seq_len, pred_len, label_len, c_out, d_ff=2048,
                 activation='gelu', e_layers=2, d_layers=1, output_attention=True, factor=5,
                 n_heads=8, d_model=512, embed='timeF', freq='h', dropout=0.0, attn='prob'):
        super(Testformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # 用传统的transformer模型
        # 先初始化好嵌入层、encoder、decoder三个组件

        # 嵌入层组件

        self.enc_embedding = DataEmbedding_cycle_pos(enc_in, d_model, embed, freq, dropout)

        self.dec_embedding = DataEmbedding_cycle_pos(dec_in, d_model, embed, freq, dropout)

        Attn = ProbAttention if attn=='prob' else FullAttention

        # 构建编码器模块
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],

            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 一个自注意力
                    AutoCorrelationLayer(
                        Attn(True, factor, attention_dropout=dropout),
                        d_model, n_heads),
                    # 一个交叉注意力
                    AutoCorrelationLayer(
                        FullAttention(False, factor, attention_dropout=dropout),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # 线性层，实现投影
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
