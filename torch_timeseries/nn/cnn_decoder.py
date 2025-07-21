import torch.nn as nn
import torch.nn.functional as F


class CNNDecoder(nn.Module):
    def __init__(self,  middle_channel=32, out_dim=3) -> None:
        super().__init__()
        self.end_conv_1 = nn.Conv2d(in_channels=middle_channel,
                                             out_channels=middle_channel,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=middle_channel,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        
        
        
    def forward(self, x):
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x