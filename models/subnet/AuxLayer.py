import torch.nn as nn
import torch.nn.functional as F
from .ConvBNReLU import ConvBNReLU

class AuxLayer(nn.Module):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x