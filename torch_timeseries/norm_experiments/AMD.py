from dataclasses import dataclass

# import fire
import fire

# from torch_timeseries.models import VMD_DLinear
from torch_timeseries.models.AMD.tsAMD import AMD

from torch_timeseries.norm_experiments.experiment import NormExperiment


@dataclass
class AMDExperiment(NormExperiment):
    model_type: str = "AMD"

    individual: bool = False
    alpha: float = 0.0
    patch: int = 4

    def _init_f_model(self):
        if self.dataset.name == "ETTh2":
            self.alpha = 1.0
        if self.dataset.name == "ETTh1":
            self.patch = 16
        if self.dataset.name == "ETTh2":
            self.patch = 4
        if self.dataset.name == "ETTm1":
            self.patch = 16
        if self.dataset.name == "ETTm2":
            self.patch = 8
        if self.dataset.name == "exchange_rate":
            self.patch = 8
        if self.dataset.name == "traffic":
            self.patch = 16
        if self.dataset.name == "weather":
            self.patch = 16
        if self.dataset.name == "electricity":
            self.patch = 16


        self.f_model = AMD(
            input_shape=(self.windows, self.dataset.num_features),
            pred_len=self.pred_len,
            dropout=0.1,
            n_block=1,
            patch=self.patch,
            k=3,
            c=2,
            alpha=self.alpha,
            target_slice=slice(0, None),
            norm=True,
            layernorm=True)
        # self.f_model = VMD_DLinear(
        #     seq_len=self.windows,
        #     pred_len=self.pred_len,
        #     enc_in=self.dataset.num_features,
        #     individual=self.individual,
        # )

        self.f_model = self.f_model.to(self.device)

    def process_batch_amd(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x, _ = self.model.normalize(batch_x)  # (B, T, N)   # (B,L,N)

        pred, moe_loss = self.model.fm(batch_x)  # (B, O, N)

        pred = self.model.denormalize(pred)

        return pred, batch_y, moe_loss  # (B, O, N), (B, O, N)


def cli():
    fire.Fire(AMDExperiment)


if __name__ == "__main__":
    cli()
