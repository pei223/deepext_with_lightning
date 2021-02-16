from .backbone_key import BackBoneKey
from .block import ResidualBlock, UpSamplingBlock
from .basic import Conv2DBatchNormRelu, BottleNeck, BottleNeckIdentity, GlobalAveragePooling
import torch.nn as nn
import torchvision

__all__ = ['FeatureDecoder', 'FeatureMapBackBone', 'create_backbone', 'ClassifierHead',
           'ResNetBackBone', 'ResNextBackBone', 'AttentionClassifierBranch', 'EfficientNetBackBone',
           'ResNeStSubnetwork']

from .efficientnet_subnetwork import EfficientNetFeatureExtractor
from .resnest_subnetwork import ResNeStSubnetwork


class FeatureMapBackBone(nn.Module):
    def __init__(self, in_channels=3, first_layer_channels=32):
        super().__init__()
        n_block_list = [3, 4, 6, 3]

        self._feature_map_layer1 = Conv2DBatchNormRelu(in_channels, first_layer_channels, stride=2)
        self._feature_map_layer2 = Conv2DBatchNormRelu(first_layer_channels, first_layer_channels)
        self._feature_map_layer3 = Conv2DBatchNormRelu(first_layer_channels, first_layer_channels * 2)
        self._pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._residual_layer1 = ResidualBlock(in_channels=first_layer_channels * 2,
                                              mid_channels=first_layer_channels,
                                              stride=1,
                                              out_channels=first_layer_channels * 4, n_blocks=n_block_list[0])

        self._residual_layer2 = ResidualBlock(in_channels=first_layer_channels * 4,
                                              mid_channels=first_layer_channels * 2,
                                              out_channels=first_layer_channels * 8,
                                              stride=2,
                                              n_blocks=n_block_list[1])

        self._dilated_layer1 = ResidualBlock(in_channels=first_layer_channels * 8,
                                             mid_channels=first_layer_channels * 4,
                                             out_channels=first_layer_channels * 16, dilation=2,
                                             n_blocks=n_block_list[2])

        self._dilated_layer2 = ResidualBlock(in_channels=first_layer_channels * 16,
                                             mid_channels=first_layer_channels * 8,
                                             out_channels=first_layer_channels * 32, dilation=4,
                                             n_blocks=n_block_list[3])

        self._out_channels = first_layer_channels * 32

    def forward(self, x):
        x = self._feature_map_layer1(x)
        x = self._feature_map_layer2(x)
        x = self._feature_map_layer3(x)
        x = self._pool(x)
        x = self._residual_layer1(x)
        x = self._residual_layer2(x)
        x = self._dilated_layer1(x)
        x = self._dilated_layer2(x)
        return x

    def output_filter_num(self):
        return self._out_channels


class FeatureDecoder(nn.Module):
    def __init__(self, in_channels: int, heigh: int, width: int, n_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self._size = (heigh, width)
        self._upsampling1 = UpSamplingBlock(in_channels=in_channels, out_channels=int(in_channels / 4))
        self._upsampling2 = UpSamplingBlock(in_channels=int(in_channels / 4), out_channels=int(in_channels / 8))
        self._upsampling3 = UpSamplingBlock(in_channels=int(in_channels / 8), out_channels=int(in_channels / 16))

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.classification = Conv2DBatchNormRelu(in_channels=int(in_channels / 16), out_channels=n_classes,
                                                  kernel_size=1, padding=0)
        self._upsampling = nn.UpsamplingBilinear2d(size=self._size)

    def forward(self, x):
        x = self._upsampling1(x)
        x = self._upsampling2(x)
        x = self._upsampling3(x)
        x = self.dropout1(x)
        x = self.classification(x)
        output = self._upsampling(x)
        return output


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_blocks=3):
        super().__init__()
        self.model = nn.Sequential()
        for i in range(n_blocks - 1):
            if i == 0:
                self.model.add_module(f"block{i + 1}",
                                      BottleNeck(in_channels=in_channels,
                                                 mid_channels=in_channels,
                                                 out_channels=in_channels, stride=2))
                continue
            self.model.add_module(f"block{i + 1}",
                                  BottleNeckIdentity(in_channels=in_channels,
                                                     out_channels=in_channels))
        self.model.add_module(f"block{n_blocks}",
                              nn.Conv2d(kernel_size=1, padding=0, in_channels=in_channels,
                                        out_channels=n_classes))
        self.model.add_module("gap", GlobalAveragePooling())

    def forward(self, x):
        y = self.model(x)
        return y.view(y.shape[0], -1)  # (Batch size, class num)


class AttentionClassifierBranch(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, n_blocks=3):
        super().__init__()
        assert n_blocks > 0
        self.conv_layers = nn.Sequential()
        for i in range(n_blocks):
            self.conv_layers.add_module(f"block{i + 1}",
                                        BottleNeckIdentity(in_channels=in_channels, out_channels=in_channels))

        self.conv_layers.add_module("bn", nn.BatchNorm2d(in_channels))
        self.conv_layers.add_module("1x1conv",
                                    nn.Conv2d(in_channels=in_channels, out_channels=n_classes, kernel_size=1,
                                              padding=0))
        self.conv_layers.add_module("relu", nn.ReLU())

        self.attention_mapper = nn.Sequential(
            nn.Conv2d(in_channels=n_classes, out_channels=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=n_classes, out_channels=n_classes, kernel_size=1, padding=0),
            GlobalAveragePooling(),
        )

    def forward(self, feature):
        feature = self.conv_layers(feature)
        attention_map = self.attention_mapper(feature)
        classification_label = self.classifier(feature).view(feature.shape[0], -1)
        return classification_label, attention_map


class ResNetBackBone(nn.Module):
    def __init__(self, resnet_type: BackBoneKey = BackBoneKey.RESNET_18, pretrained=True):
        super().__init__()
        assert resnet_type in [BackBoneKey.RESNET_18, BackBoneKey.RESNET_34, BackBoneKey.RESNET_50,
                               BackBoneKey.RESNET_101, BackBoneKey.RESNET_152, ]
        if resnet_type == BackBoneKey.RESNET_18:
            self.resnet_model = torchvision.models.resnet18(pretrained=pretrained)
        elif resnet_type == BackBoneKey.RESNET_34:
            self.resnet_model = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet_type == BackBoneKey.RESNET_50:
            self.resnet_model = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet_type == BackBoneKey.RESNET_101:
            self.resnet_model = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet_type == BackBoneKey.RESNET_152:
            self.resnet_model = torchvision.models.resnet152(pretrained=pretrained)

    def forward(self, x):
        """
        :param x: (Batch size, 3, height, width)
        :return: (Batch size, 64, height/2, width/2), (Batch size, 128, height/4, width/4),
        (Batch size, 256, height/8, width/8), (Batch size, 512, height/16, width/16)
        """
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        out1 = self.resnet_model.layer1(x)
        out2 = self.resnet_model.layer2(out1)
        out3 = self.resnet_model.layer3(out2)
        out4 = self.resnet_model.layer4(out3)
        return out1, out2, out3, out4


class ResNextBackBone(nn.Module):
    def __init__(self, resnext_type: BackBoneKey = BackBoneKey.RESNEXT_50, pretrained=True):
        super().__init__()
        assert resnext_type in [BackBoneKey.RESNEXT_50, BackBoneKey.RESNEXT_101]
        if resnext_type == BackBoneKey.RESNEXT_50:
            self.resnet_model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif resnext_type == BackBoneKey.RESNEXT_101:
            self.resnet_model = torchvision.models.resnext101_32x8d(pretrained=pretrained)

    def forward(self, x):
        """
        :param x: (Batch size, 3, height, width)
        :return: (Batch size, 64, height/2, width/2), (Batch size, 128, height/4, width/4),
        (Batch size, 256, height/8, width/8), (Batch size, 512, height/16, width/16)
        """
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        out1 = self.resnet_model.layer1(x)
        out2 = self.resnet_model.layer2(out1)
        out3 = self.resnet_model.layer3(out2)
        out4 = self.resnet_model.layer4(out3)
        return out1, out2, out3, out4


class EfficientNetBackBone(nn.Module):
    def __init__(self, model_type: BackBoneKey = BackBoneKey.EFFICIENTNET_B0, pretrained=True):
        super().__init__()
        assert model_type in [BackBoneKey.EFFICIENTNET_B0, BackBoneKey.EFFICIENTNET_B1, BackBoneKey.EFFICIENTNET_B2,
                              BackBoneKey.EFFICIENTNET_B3, BackBoneKey.EFFICIENTNET_B4, BackBoneKey.EFFICIENTNET_B5,
                              BackBoneKey.EFFICIENTNET_B6, BackBoneKey.EFFICIENTNET_B7, ]
        if pretrained:
            self.model = EfficientNetFeatureExtractor.from_pretrained(model_name=model_type.value)
        else:
            self.model = EfficientNetFeatureExtractor.from_name(model_name=model_type.value)

    def forward(self, x, start_idx=0, end_idx=-1):
        return self.model(x, start_idx, end_idx)


def create_backbone(backbone_key: BackBoneKey, pretrained=True) -> nn.Module:
    if backbone_key in [BackBoneKey.RESNET_18, BackBoneKey.RESNET_34, BackBoneKey.RESNET_50, BackBoneKey.RESNET_101,
                        BackBoneKey.RESNET_152]:
        return ResNetBackBone(resnet_type=backbone_key, pretrained=pretrained)
    elif backbone_key in [BackBoneKey.RESNEXT_50, BackBoneKey.RESNEXT_101]:
        return ResNextBackBone(resnext_type=backbone_key, pretrained=pretrained)
    elif backbone_key in [BackBoneKey.RESNEST_50, BackBoneKey.RESNEST_101,
                          BackBoneKey.RESNEST_200, BackBoneKey.RESNEST_269]:
        return ResNeStSubnetwork(resnest_backbone=backbone_key)
    elif backbone_key in [BackBoneKey.EFFICIENTNET_B0, BackBoneKey.EFFICIENTNET_B1, BackBoneKey.EFFICIENTNET_B2,
                          BackBoneKey.EFFICIENTNET_B3, BackBoneKey.EFFICIENTNET_B4, BackBoneKey.EFFICIENTNET_B5,
                          BackBoneKey.EFFICIENTNET_B6, BackBoneKey.EFFICIENTNET_B7, ]:
        return EfficientNetBackBone(model_type=backbone_key, pretrained=pretrained)
