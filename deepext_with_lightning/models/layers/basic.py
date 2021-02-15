import torch
from torch import nn as nn
from torch.nn import functional as F


class Conv2DTrasnposeBatchNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1,
                 bias: bool = False, dropout_rate: float = 0.2):
        super().__init__()
        self._layers = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self._layers(x)


class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        return self.batchnorm(x)

    def init_weights(self):
        init_weights_func(self.conv)


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1, bias: bool = False):
        super().__init__()
        self._conv2d_batch_norm = Conv2DBatchNorm(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                  bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self._conv2d_batch_norm.forward(x)
        return self.relu(x)


class Conv2DBatchNormLeakyRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1, bias: bool = False):
        super().__init__()
        self._conv2d_batch_norm = Conv2DBatchNorm(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                  bias)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self._conv2d_batch_norm.forward(x)
        return self.relu(x)


def init_weights_func(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.bias.data) if m.bias is not None else None


class BottleNeck(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self._layer1 = Conv2DBatchNormRelu(in_channels=in_channels,
                                           out_channels=mid_channels, kernel_size=1, padding=0)
        self._layer2 = Conv2DBatchNormRelu(in_channels=mid_channels,
                                           out_channels=mid_channels, kernel_size=3, padding=dilation,
                                           dilation=dilation, stride=stride)
        self._layer3 = Conv2DBatchNorm(in_channels=mid_channels,
                                       out_channels=out_channels, kernel_size=1, padding=0)

        self._residual_layer = Conv2DBatchNormRelu(in_channels=in_channels,
                                                   out_channels=out_channels, kernel_size=1, padding=0, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        feature = self._layer1(x)
        feature = self._layer2(feature)
        feature = self._layer3(feature)

        residual = self._residual_layer(x)
        return self.relu(feature + residual)


class BottleNeckIdentity(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self._layer1 = Conv2DBatchNormRelu(in_channels=in_channels,
                                           out_channels=out_channels, kernel_size=1, padding=0)
        self._layer2 = Conv2DBatchNormRelu(in_channels=out_channels,
                                           out_channels=out_channels, kernel_size=3, padding=dilation,
                                           dilation=dilation)
        self._layer3 = Conv2DBatchNorm(in_channels=out_channels,
                                       out_channels=in_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feature = self._layer1(x)
        feature = self._layer2(feature)
        feature = self._layer3(feature)
        return self.relu(feature + x)


class FeaturePyramidPooling(nn.Module):
    def __init__(self, in_channels, compress_sizes=(6, 3, 2, 1)):
        super().__init__()
        out_channels = int(in_channels / len(compress_sizes))
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, size) for size in compress_sizes])

    def _make_stage(self, in_channels, out_channels, size):
        compress_pooling = nn.AdaptiveAvgPool2d(output_size=size)
        conv = Conv2DBatchNormLeakyRelu(in_channels, out_channels, kernel_size=1, bias=False, padding=0)
        return nn.Sequential(compress_pooling, conv)

    def forward(self, feats):
        img_size = (feats.shape[2], feats.shape[3])
        priors = [feats, ] + [
            F.interpolate(input=stage(feats), size=img_size, mode='bilinear', align_corners=True) for stage in
            self.stages]
        return torch.cat(priors, 1)


class GlobalAveragePooling(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class DropBlock2d(nn.Module):
    """
    https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py
    """

    def __init__(self, dropout_rate: float = 0.2, block_size: int = 5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.block_size = block_size

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4
        if not self.training or self.dropout_rate == 0.0 or self.block_size == 0:
            return x
        # get gamma value
        threshold = self._compute_threshold()

        # sample mask
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < threshold).float()

        # place mask on input device
        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)

        # apply block mask
        out = x * block_mask[:, None, :, :]

        # scale output
        out = out * block_mask.numel() / block_mask.sum()

        return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_threshold(self):
        return self.dropout_rate / (self.block_size ** 2)
