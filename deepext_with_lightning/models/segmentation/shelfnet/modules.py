from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....models.layers.block import Conv2DBatchNormRelu, SharedWeightResidualBlock, ChannelWiseAttentionBlock


class SegmentationShelf(torch.nn.Module):
    def __init__(self, in_channels_ls: List[int]):
        super().__init__()
        self._decoder = Decoder(in_channels_ls=in_channels_ls)
        self._encoder = Encoder(in_channels_ls=in_channels_ls)
        self._final_decoder = Decoder(in_channels_ls=in_channels_ls)

    def forward(self, inputs: List[torch.Tensor]):
        dec_outputs = self._decoder(inputs)
        enc_outputs = self._encoder(dec_outputs)
        final_output = self._final_decoder(enc_outputs)
        return final_output


class OutLayer(torch.nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, n_classes: int, out_size: Tuple[int, int]):
        super().__init__()
        self._out_size = out_size
        self.mid_conv = Conv2DBatchNormRelu(in_channels=in_channels, out_channels=mid_channels, kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.conv_out = torch.nn.Conv2d(in_channels=mid_channels, out_channels=n_classes, kernel_size=3, stride=1,
                                        padding=1)

    def forward(self, x):
        x = self.mid_conv(x)
        x = self.conv_out(x)
        return F.interpolate(x, size=self._out_size, mode='bilinear', align_corners=True)


class Decoder(nn.Module):
    def __init__(self, in_channels_ls: List[int]):
        super().__init__()

        self._first_sblock = SharedWeightResidualBlock(in_channels=in_channels_ls[-1])

        sblocks = list(
            map(lambda in_channel: SharedWeightResidualBlock(in_channels=in_channel), in_channels_ls[::-1][1:]))
        self._sblocks = nn.ModuleList(sblocks)

        up_conv_list, up_dense_list = [], []

        for i in range(1, len(in_channels_ls)):
            up_conv_list.append(nn.ConvTranspose2d(in_channels=in_channels_ls[i], out_channels=in_channels_ls[i - 1],
                                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))
            up_dense_list.append(SharedWeightResidualBlock(in_channels=in_channels_ls[i - 1]))

        # Low -> High resolution
        self._up_conv_list = nn.ModuleList(up_conv_list[::-1])
        self._up_dense_list = nn.ModuleList(up_dense_list[::-1])

    def forward(self, inputs: List[torch.Tensor]):
        # inputsは高解像度から低解像度の順
        # 低解像度から高解像度へアップサンプリング
        out = self._first_sblock(inputs[-1], None)
        outs = [out, ]
        for i, (up_conv, up_dense) in enumerate(zip(self._up_conv_list, self._up_dense_list)):
            out = up_conv(out)
            pre_input = inputs[-(i + 2)]  # 高解像度から低解像度順のため逆のインデックスになる
            out = up_dense(out, pre_input)
            outs.insert(0, out)
        return outs


class Encoder(nn.Module):
    def __init__(self, in_channels_ls: List[int]):
        super().__init__()

        sblocks = list(map(lambda in_channel: SharedWeightResidualBlock(in_channels=in_channel), in_channels_ls[:-1]))
        self._sblocks = nn.ModuleList(sblocks)

        down_convs = []
        for i in range(1, len(in_channels_ls)):
            down_convs.append(nn.Conv2d(kernel_size=3, stride=2, in_channels=in_channels_ls[i - 1],
                                        out_channels=in_channels_ls[i], padding=1))
        self._down_convs = nn.ModuleList(down_convs)

    def forward(self, inputs: List[torch.Tensor]):
        out = None
        outs = []
        for i, (sblock_layer, down_conv_layer) in enumerate(zip(self._sblocks, self._down_convs)):
            out = sblock_layer(inputs[i], out)
            outs.append(out)
            out = down_conv_layer(out)
        outs.append(out)
        return outs
