import torch.nn as nn
import torch

from ..layers.backbone_key import BackBoneKey


class ResNeStSubnetwork(nn.Module):
    def __init__(self, resnest_backbone: BackBoneKey):
        super().__init__()
        assert resnest_backbone in [BackBoneKey.RESNEST_50, BackBoneKey.RESNEST_101,
                                    BackBoneKey.RESNEST_200, BackBoneKey.RESNEST_269]
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        self._net = torch.hub.load('zhanghang1989/ResNeSt', resnest_backbone.value, pretrained=True)

    def forward(self, x):
        x = self._net.conv1(x)
        x = self._net.bn1(x)
        x = self._net.relu(x)
        x = self._net.maxpool(x)

        x1 = self._net.layer1(x)
        x2 = self._net.layer2(x1)
        x3 = self._net.layer3(x2)
        x4 = self._net.layer4(x3)
        return x1, x2, x3, x4
