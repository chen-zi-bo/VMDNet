from dataclasses import dataclass
from types import SimpleNamespace

import fire
import torch

from torch_timeseries.models.itransformer.iTransformer import Model
from torch_timeseries.norm_experiments.experiment import NormExperiment


@dataclass
class itransformerExperiment(NormExperiment):
    model_type: str = "itransformer"

    def _init_f_model(self):
        # 模型初始化
        configs = SimpleNamespace(
            seq_len=self.windows,
            pred_len=self.pred_len,
            output_attention=False,
            use_norm=True,
            d_ff=128,
            d_model=128,
            dropout=0.1,
            embed='timeF',
            freq='h',
            class_strategy='projection',
            factor=1,
            n_heads=8,
            activation='gelu',
            e_layers=2
        )
        if self.dataset.name == "electricity" or self.dataset.name == "traffic" or self.dataset.name == "weather":
            configs.d_model = 512
            configs.d_ff = 512
        if self.dataset.name == "ETTh1":
            configs.d_model = 256
            configs.d_ff = 256
        if self.dataset.name == "electricity" or self.dataset.name == "weather":
            configs.e_layers = 3
        if self.dataset.name == "traffic":
            configs.e_layers = 4
        self.f_model = Model(configs)
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
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float().to(self.device)

        batch_x, dec_inp = self.model.normalize(batch_x, dec_inp=dec_inp)  # (B, T, N)   # (B,L,N)

        pred = self.model.fm(batch_x, batch_x_date_enc, dec_inp, batch_y_date_enc)

        pred = self.model.denormalize(pred)

        pred = pred[:, -self.pred_len:, 0:]
        batch_y = batch_y[:, -self.pred_len:, 0:].to(self.device)

        return pred, batch_y  # (B, O, N), (B, O, N)

        # pred = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc) # (B, O, N)
        # return pred


def cli():
    fire.Fire(itransformerExperiment)


if __name__ == "__main__":
    cli()
