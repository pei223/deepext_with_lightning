from typing import Tuple

import torch
from torch import nn

from ...layers.basic import BottleNeck
from ...layers.block import DownBlock, UpBlock


class UNetModel(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, first_layer_channels: int = 64):
        super().__init__()
        self._first_layer_channels = first_layer_channels
        self._encoder_layer1 = self.down_sampling_layer(n_input_channels, first_layer_channels)
        self._encoder_layer2 = self.down_sampling_layer(first_layer_channels, first_layer_channels * 2)
        self._encoder_layer3 = self.down_sampling_layer(first_layer_channels * 2, first_layer_channels * 4)
        self._encoder_layer4 = self.down_sampling_layer(first_layer_channels * 4, first_layer_channels * 8)
        self._encoder_layer5 = self.down_sampling_layer(first_layer_channels * 8, first_layer_channels * 16)

        self._decoder_layer1 = self.up_sampling_layer(first_layer_channels * 16, first_layer_channels * 8)
        self._decoder_layer2 = self.up_sampling_layer(first_layer_channels * 16, first_layer_channels * 4)
        self._decoder_layer3 = self.up_sampling_layer(first_layer_channels * 8, first_layer_channels * 2)
        self._decoder_layer4 = self.up_sampling_layer(first_layer_channels * 4, first_layer_channels)
        self._decoder_layer5 = self.up_sampling_layer(first_layer_channels * 2, n_output_channels, is_output_layer=True)

        self.apply(init_weights_func)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        :param x: (batch size, channels, height, width)
        :return:  (batch size, class, height, width)
        """
        enc1 = self._encoder_layer1(x)
        enc2 = self._encoder_layer2(enc1)
        enc3 = self._encoder_layer3(enc2)
        enc4 = self._encoder_layer4(enc3)
        encoded_feature = self._encoder_layer5(enc4)

        x = self._decoder_layer1(encoded_feature)
        x = torch.cat([x, enc4], dim=1)
        x = self._decoder_layer2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self._decoder_layer3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self._decoder_layer4(x)
        x = torch.cat([x, enc1], dim=1)
        output = self._decoder_layer5(x)
        return output

    def down_sampling_layer(self, n_input_channels: int, n_out_channels: int):
        # 継承することでエンコーダーにResnetBlocなど適用可能
        return DownBlock(n_input_channels, n_out_channels)

    def up_sampling_layer(self, n_input_channels: int, n_out_channels: int, is_output_layer=False):
        # 継承することでエンコーダーにResnetBlocなど適用可能
        return UpBlock(n_input_channels, n_out_channels, is_output_layer)


class ResUNetModel(UNetModel):
    def __init__(self, n_classes: int, n_input_channels: int = 3, first_layer_channels: int = 64):
        super().__init__(n_classes, n_input_channels, first_layer_channels)

    def down_sampling_layer(self, n_input_channels: int, n_out_channels: int):
        return BottleNeck(n_input_channels, mid_channels=n_input_channels, out_channels=n_out_channels, stride=2)


def init_weights_func(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
