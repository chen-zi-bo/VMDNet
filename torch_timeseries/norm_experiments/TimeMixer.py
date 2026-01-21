from dataclasses import dataclass

import fire
import torch
from types import SimpleNamespace
from torch_timeseries.models.TimeMixer.TimeMixer import Model
from torch_timeseries.norm_experiments.experiment import NormExperiment


@dataclass
class TimeMixerExperiment(NormExperiment):
    model_type: str = "TimeMixer"

    def _init_f_model(self):
        # 模型初始化
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=self.windows,
            label_len=0,
            pred_len=self.pred_len,
            down_sampling_layers=3,
            down_sampling_window=2,
            down_sampling_method="avg",
            enc_in=self.dataset.num_features,
            c_out=self.dataset.num_features,
            channel_independence=1,
            e_layers=3,
            moving_avg=25,

            # embedding
            d_model=16,
            dropout=0.1,
            decomp_method='moving_avg',
            d_ff=32,
            use_future_temporal_feature=0,
            embed='timeF',
            freq='h',
            # norm
            use_norm=1,

        )
        if self.dataset.name == "ETTh1" or self.dataset.name == "ETTh2" or self.dataset.name == "ETTm1" or self.dataset.name == "ETTm2":
            configs.e_layers = 2
        if self.dataset.name == "ETTm2" or self.dataset.name == "traffic":
            configs.d_model = 32
        if self.dataset.name == "traffic":
            configs.d_ff = 64
        self.f_model = Model(configs)
        # self.f_model = Informer(
        #     self.dataset.num_features,
        #     self.dataset.num_features,
        #     self.dataset.num_features,
        #     self.pred_len,
        #     factor=self.factor,
        #     d_model=self.d_model,
        #     n_heads=self.n_heads,
        #     e_layers=self.e_layers,
        #     dropout=self.dropout,
        #     attn=self.attn,
        #     embed=self.embed,
        #     activation=self.activation,
        #     distil=self.distil,
        #     mix=self.mix,
        # )
        self.f_model = self.f_model.to(self.device)

    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x:  (B, T, N)
        # batch_y:  (B, Steps,T)
        # batch_x_date_enc:  (B, T, N)
        # batch_y_date_enc:  (B, T, Steps)

        # outputs:
        # pred: (B, O, N)
        # label:  (B,O,N)
        # 赋0是为了mask，不看到之后的信息
        # dec_inp_pred = torch.zeros(
        #     [batch_x.size(0), self.pred_len, self.dataset.num_features]
        # ).to(self.device)
        # # 解码器输入前面那一部分为label_len
        # dec_inp_label = batch_x[:, self.label_len:, :].to(self.device)
        #
        # dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        # dec_inp_date_enc = torch.cat(
        #     [batch_x_date_enc[:, self.label_len:, :], batch_y_date_enc], dim=1
        # )
        dec_inp = None

        batch_x, dec_inp = self.model.normalize(batch_x, dec_inp=dec_inp)  # (B, T, N)   # (B,L,N)

        pred = self.model.fm(batch_x, batch_x_date_enc, dec_inp, batch_y_date_enc)

        pred = self.model.denormalize(pred)

        return pred, batch_y  # (B, O, N), (B, O, N)

        # pred = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc) # (B, O, N)
        # return pred


def cli():
    fire.Fire(TimeMixerExperiment)


if __name__ == "__main__":
    cli()
