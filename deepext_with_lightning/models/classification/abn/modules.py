from typing import Tuple

import torch
from torch import nn as nn

from ...layers.backbone_key import BackBoneKey, BACKBONE_CHANNEL_COUNT_DICT
from ...layers.subnetwork import create_backbone, AttentionClassifierBranch, ClassifierHead


class ABNModel(nn.Module):
    def __init__(self, n_classes: int, pretrained=True,
                 backbone: BackBoneKey = BackBoneKey.RESNET_50, n_blocks=3):
        super().__init__()
        self.feature_extractor = create_backbone(backbone_key=backbone, pretrained=pretrained)
        feature_channel_num = BACKBONE_CHANNEL_COUNT_DICT[backbone][-1]
        self.attention_branch = AttentionClassifierBranch(in_channels=feature_channel_num, n_classes=n_classes,
                                                          n_blocks=n_blocks)
        self.perception_branch = ClassifierHead(in_channels=feature_channel_num, n_blocks=n_blocks,
                                                n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (batch size, channels, height, width)
        :return: (batch size, class), (batch size, class), heatmap (batch size, 1, height, width)
        """
        origin_feature = self.feature_extractor(x)[-1]
        attention_output, attention_map = self.attention_branch(origin_feature)
        # 特徴量・Attention mapのSkip connection
        perception_feature = (origin_feature * attention_map) + origin_feature
        perception_output = self.perception_branch(perception_feature)
        return perception_output, attention_output, attention_map
