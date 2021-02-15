from typing import Optional

import torch.nn as nn
from torch.nn import functional as F
import torch
from ..layers.basic import Conv2DTrasnposeBatchNorm, Conv2DBatchNormRelu, BottleNeck, BottleNeckIdentity, \
    DropBlock2d, Conv2DBatchNorm, Conv2DBatchNormLeakyRelu


class ResidualBlock(nn.Module):
    def __init__(self, n_blocks: int, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1,
                 dilation: int = 1):
        super().__init__()
        assert n_blocks > 0

        self._blocks = nn.Sequential()
        self._blocks.add_module(
            f"block1",
            BottleNeck(out_channels=out_channels, mid_channels=mid_channels, in_channels=in_channels, stride=stride,
                       dilation=dilation)
        )
        for i in range(n_blocks - 1):
            self._blocks.add_module(
                f"block{i + 2}",
                BottleNeckIdentity(out_channels=mid_channels, in_channels=out_channels, dilation=dilation)
            )

    def forward(self, x):
        return self._blocks(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks: int = 3):
        super().__init__()

        self._layers = nn.Sequential()

        if n_blocks == 1:
            self._layers.add_module(
                "block1",
                Conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, stride=2)
            )
            return
        self._layers.add_module(
            "block1",
            Conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels)
        )
        for i in range(n_blocks - 2):
            self._layers.add_module(
                f"block{i + 2}",
                Conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels)
            )
        self._layers.add_module(
            f"block_output",
            Conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, stride=2)
        )

    def forward(self, x):
        return self._layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks: int = 3, is_output_layer=False):
        super().__init__()

        self._layers = nn.Sequential()

        if n_blocks == 1 and is_output_layer:
            self._layers.add_module(
                "upblock1_output",
                nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                              nn.Softmax2d())
            )
            return

        self._layers.add_module(
            "upblock1",
            Conv2DTrasnposeBatchNorm(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                     stride=2)
        )
        if n_blocks == 1:
            return

        for i in range(n_blocks - 2):
            self._layers.add_module(
                f"upblock{i + 2}",
                Conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels)
            )

        if is_output_layer:
            self._layers.add_module(
                "upblock1_output",
                nn.Sequential(Conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels),
                              nn.Softmax2d(), )
            )
            return
        self._layers.add_module(
            f"upblock_out",
            Conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self._layers(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class SharedWeightResidualBlock(nn.Module):
    def __init__(self, in_channels: int, dropout_rate=0.1):
        super().__init__()
        self.shared_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                     padding=1)
        self._relu = nn.ReLU()
        self._bn1 = nn.BatchNorm2d(in_channels)
        self._dropout = nn.Dropout2d(dropout_rate)
        self._bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, lateral_input: torch.Tensor, vertical_input: Optional[torch.Tensor]):
        """
        :param lateral_input: (Batch size, in_channel, height, width)
        :param vertical_input: (Batch size, in_channel, height, width)
        :return:  (Batch size, out_channel, height, width)
        """
        input_ = lateral_input + vertical_input if vertical_input is not None else lateral_input
        out = self.shared_conv(input_)
        out = self._relu(self._bn1(out))
        out = self._dropout(out)
        out = self.shared_conv(out)
        out = self._bn2(out)
        out = out + input_
        return self._relu(out)


class SharedWeightResidualBlockWithDropBlock(nn.Module):
    def __init__(self, in_channels: int, dropout_rate=0.3):
        super().__init__()
        self.shared_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                     padding=1)
        self._relu = nn.ReLU()
        self._bn1 = nn.BatchNorm2d(in_channels)
        self._dropblock = DropBlock2d(dropout_rate)
        self._bn2 = nn.BatchNorm2d(in_channels)
        self.init_weight()

    def forward(self, lateral_input: torch.Tensor, vertical_input: Optional[torch.Tensor]):
        """
        :param lateral_input: (Batch size, in_channel, height, width)
        :param vertical_input: (Batch size, in_channel, height, width)
        :return:  (Batch size, out_channel, height, width)
        """
        input_ = lateral_input + vertical_input if vertical_input is not None else lateral_input
        out = self.shared_conv(input_)
        out = self._relu(self._bn1(out))
        out = self._dropblock(out)
        out = self.shared_conv(out)
        out = self._bn2(out)
        out = out + input_
        return self._relu(out)

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)


class ChannelWiseAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = Conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                        stride=1, padding=1)
        self.conv_attention = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.bn_attention = nn.BatchNorm2d(out_channels)
        self.sigmoid_attention = nn.Sigmoid()
        self.init_weight()

    def forward(self, x: torch.Tensor):
        feature = self.conv(x)
        attention = F.avg_pool2d(feature, feature.size()[2:])
        attention = self.conv_attention(attention)
        attention = self.bn_attention(attention)
        attention = self.sigmoid_attention(attention)
        out = torch.mul(feature, attention)
        return out

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
