from dataclasses import dataclass

import fire

from torch_timeseries.models.CycleNet import CycleNet
from torch_timeseries.norm_experiments.experiment import NormExperiment


@dataclass
class CycleNetExperiment(NormExperiment):
    model_type: str = "CycleNet"

    inner_model_type: str = 'linear'

    use_revin: int = 1
    d_model: int = 512

    def _init_f_model(self):
        self.f_model = CycleNet(self.windows, self.pred_len, self.dataset.num_features,
                                self.dataloader.cycle, self.inner_model_type, self.d_model, self.use_revin)
        self.f_model = self.f_model.to(self.device)

    def process_cycle(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, batch_cycle):
        batch_x, _ = self.model.normalize(batch_x)  # (B, T, N)   # (B,L,N)

        pred = self.model.fm(batch_x, batch_cycle)  # (B, O, N)

        pred = self.model.denormalize(pred)

        return pred, batch_y  # (B, O, N), (B, O, N)


def cli():
    fire.Fire(CycleNetExperiment)


if __name__ == "__main__":
    cli()
