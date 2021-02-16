from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import nn as nn

from ...layers.backbone_key import BackBoneKey, BACKBONE_CHANNEL_COUNT_DICT
from ...layers.subnetwork import create_backbone
from ...layers.block import Conv2DBatchNormRelu, SharedWeightResidualBlock


class ShelfNetModel(nn.Module):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], backbone: BackBoneKey = BackBoneKey.RESNET_18,
                 pretrained=True):
        super().__init__()

        backbone_out_channel_ls = BACKBONE_CHANNEL_COUNT_DICT[backbone]
        assert backbone_out_channel_ls is not None, "Invalid backbone type."
        if backbone in [BackBoneKey.RESNET_18]:
            mid_channel_ls = [64, 128, 256, 512]
        elif backbone in [BackBoneKey.RESNET_34, BackBoneKey.RESNET_50, BackBoneKey.RESNEST_50,
                          BackBoneKey.RESNEXT_50, ]:
            mid_channel_ls = [128, 256, 512, 1024]
        elif backbone in [BackBoneKey.RESNET_101, BackBoneKey.RESNET_152,
                          BackBoneKey.RESNEXT_101, BackBoneKey.RESNEST_101,
                          BackBoneKey.RESNEST_200, BackBoneKey.RESNEST_269]:
            mid_channel_ls = [256, 512, 1024, 2048]
        else:
            assert False, "Invalid backbone type."

        self._feature_and_layer_diff = len(backbone_out_channel_ls) - len(mid_channel_ls)
        assert self._feature_and_layer_diff >= 0, "Shelfのレイヤー数はBackBoneのレイヤー数以下の必要があります."
        reducers = []
        for i in range(len(mid_channel_ls)):
            reducers.append(
                Conv2DBatchNormRelu(kernel_size=1,
                                    in_channels=backbone_out_channel_ls[self._feature_and_layer_diff + i],
                                    out_channels=mid_channel_ls[i], padding=0, bias=False))
        self._reducers = nn.ModuleList(reducers)

        self._multi_scale_backbone = create_backbone(backbone_key=backbone, pretrained=pretrained)
        self._segmentation_shelf = SegmentationShelf(in_channels_ls=mid_channel_ls)

        out_convs = []
        for i in range(len(mid_channel_ls)):
            out_convs.append(OutLayer(in_channels=mid_channel_ls[i], mid_channels=mid_channel_ls[0],
                                      n_classes=n_classes, out_size=out_size))
        self._out_convs = nn.ModuleList(out_convs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_list = self._multi_scale_backbone(x)
        reduced_x_list = []

        for i, reducer_layer in enumerate(self._reducers):
            reduced_x_list.append(reducer_layer(x_list[i + self._feature_and_layer_diff]))
        x_list = self._segmentation_shelf(reduced_x_list)
        outs = []
        for i, out_conv_layer in enumerate(self._out_convs):
            outs.append(out_conv_layer(x_list[i]))
        return outs


class ShelfNetModelWithEfficientNet(nn.Module):
    def __init__(self, n_classes: int, out_size: Tuple[int, int], backbone: BackBoneKey = BackBoneKey.EFFICIENTNET_B0,
                 pretrained=True):
        super().__init__()

        if backbone in [BackBoneKey.EFFICIENTNET_B0, BackBoneKey.EFFICIENTNET_B1, BackBoneKey.EFFICIENTNET_B2]:
            mid_channel_ls = [64, 128, 256, 512]
        elif backbone in [BackBoneKey.EFFICIENTNET_B3, BackBoneKey.EFFICIENTNET_B4, BackBoneKey.EFFICIENTNET_B5]:
            mid_channel_ls = [128, 256, 512, 1024]
        elif backbone in [BackBoneKey.EFFICIENTNET_B6, BackBoneKey.EFFICIENTNET_B7]:
            mid_channel_ls = [128, 256, 512, 1024]
        else:
            assert False, "Invalid backbone type."
        backbone_out_channel_ls = BACKBONE_CHANNEL_COUNT_DICT[backbone][1:5]

        self._feature_and_layer_diff = len(backbone_out_channel_ls) - len(mid_channel_ls)
        assert self._feature_and_layer_diff >= 0, "Shelfのレイヤー数はBackBoneのレイヤー数以下の必要があります."
        reducers = []
        for i in range(len(mid_channel_ls)):
            reducers.append(
                Conv2DBatchNormRelu(kernel_size=1,
                                    in_channels=backbone_out_channel_ls[self._feature_and_layer_diff + i],
                                    out_channels=mid_channel_ls[i], padding=0))
        self._reducers = nn.ModuleList(reducers)

        self._multi_scale_backbone = create_backbone(backbone_key=backbone, pretrained=pretrained)
        self._segmentation_shelf = SegmentationShelf(in_channels_ls=mid_channel_ls)

        out_convs = []
        for i in range(len(mid_channel_ls)):
            out_convs.append(OutLayer(in_channels=mid_channel_ls[i], mid_channels=mid_channel_ls[0],
                                      n_classes=n_classes, out_size=out_size))
        self._out_convs = nn.ModuleList(out_convs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_list = self._multi_scale_backbone(x, start_idx=1, end_idx=5)  # BackBoneの後ろからレイヤー数分取得する.
        reduced_x_list = []

        for i, reducer_layer in enumerate(self._reducers):
            reduced_x_list.append(reducer_layer(x_list[i]))

        x_list = self._segmentation_shelf(reduced_x_list)

        outs = []
        for i, out_conv_layer in enumerate(self._out_convs):
            outs.append(out_conv_layer(x_list[i]))
        return outs


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
            out = up_dense(pre_input, out)
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
            out = F.relu(out)
        outs.append(out)
        return outs
