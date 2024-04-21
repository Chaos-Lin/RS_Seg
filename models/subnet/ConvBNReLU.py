from torch import nn
from torch.nn import functional as F
class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = nn.SyncBatchNorm(out_channels)


    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return F.relu(x)
