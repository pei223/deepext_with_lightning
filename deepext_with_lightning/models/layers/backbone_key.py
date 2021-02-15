from enum import Enum


class BackBoneKey(Enum):
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"
    RESNEXT_50 = "resnext50"
    RESNEXT_101 = "resnext101"
    RESNEST_50 = "resnest50"
    RESNEST_101 = "resnest101"
    RESNEST_200 = "resnest200"
    RESNEST_269 = "resnest269"
    EFFICIENTNET_B0 = "efficientnet-b0"
    EFFICIENTNET_B1 = "efficientnet-b1"
    EFFICIENTNET_B2 = "efficientnet-b2"
    EFFICIENTNET_B3 = "efficientnet-b3"
    EFFICIENTNET_B4 = "efficientnet-b4"
    EFFICIENTNET_B5 = "efficientnet-b5"
    EFFICIENTNET_B6 = "efficientnet-b6"
    EFFICIENTNET_B7 = "efficientnet-b7"

    @staticmethod
    def from_val(val: str):
        for key in BACKBONE_KEYS:
            if key.value == val:
                return key
        return None

    @staticmethod
    def is_efficentnet(backbone_key: 'BackBoneKey') -> bool:
        return "efficientnet" in backbone_key.value

    @staticmethod
    def is_resnet_architecture(backbone_key: 'BackBoneKey') -> bool:
        return "resn" in backbone_key.value


BACKBONE_CHANNEL_COUNT_DICT = {
    BackBoneKey.RESNET_18: [64, 128, 256, 512],
    BackBoneKey.RESNET_34: [64, 128, 256, 512],
    BackBoneKey.RESNET_50: [256, 512, 1024, 2048],
    BackBoneKey.RESNET_101: [256, 512, 1024, 2048],
    BackBoneKey.RESNET_152: [256, 512, 1024, 2048],
    BackBoneKey.RESNEXT_50: [256, 512, 1024, 2048],
    BackBoneKey.RESNEXT_101: [256, 512, 1024, 2048],
    BackBoneKey.RESNEST_50: [256, 512, 1024, 2048],
    BackBoneKey.RESNEST_101: [256, 512, 1024, 2048],
    BackBoneKey.RESNEST_200: [256, 512, 1024, 2048],
    BackBoneKey.RESNEST_269: [256, 512, 1024, 2048],
    BackBoneKey.EFFICIENTNET_B0: [16, 24, 40, 80, 112, 192, 320],
    BackBoneKey.EFFICIENTNET_B1: [16, 24, 40, 80, 112, 192, 320],
    BackBoneKey.EFFICIENTNET_B2: [16, 24, 48, 88, 120, 208, 352],
    BackBoneKey.EFFICIENTNET_B3: [24, 32, 48, 96, 136, 232, 384],
    BackBoneKey.EFFICIENTNET_B4: [24, 32, 56, 112, 160, 272, 448],
    BackBoneKey.EFFICIENTNET_B5: [24, 40, 64, 128, 176, 304, 512],
    BackBoneKey.EFFICIENTNET_B6: [32, 40, 72, 144, 200, 344, 576],
    BackBoneKey.EFFICIENTNET_B7: [32, 48, 80, 160, 224, 384, 640],
}
BACKBONE_DOWNSAMPLING_RATE = {
    BackBoneKey.RESNET_18: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNET_34: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNET_50: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNET_101: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNET_152: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNEXT_50: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNEXT_101: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNEST_50: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNEST_101: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNEST_200: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.RESNEST_269: [1 / 4, 1 / 8, 1 / 16, 1 / 32],
    BackBoneKey.EFFICIENTNET_B0: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],
    BackBoneKey.EFFICIENTNET_B1: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],
    BackBoneKey.EFFICIENTNET_B2: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],
    BackBoneKey.EFFICIENTNET_B3: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],
    BackBoneKey.EFFICIENTNET_B4: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],
    BackBoneKey.EFFICIENTNET_B5: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],
    BackBoneKey.EFFICIENTNET_B6: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],
    BackBoneKey.EFFICIENTNET_B7: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128],
}
BACKBONE_KEYS = [BackBoneKey.RESNET_18, BackBoneKey.RESNET_34, BackBoneKey.RESNET_50, BackBoneKey.RESNET_101,
                 BackBoneKey.RESNET_152, BackBoneKey.RESNEXT_50, BackBoneKey.RESNEXT_101,
                 BackBoneKey.RESNEST_50, BackBoneKey.RESNEST_101, BackBoneKey.RESNEST_200, BackBoneKey.RESNEST_269,
                 BackBoneKey.EFFICIENTNET_B0, BackBoneKey.EFFICIENTNET_B1, BackBoneKey.EFFICIENTNET_B2,
                 BackBoneKey.EFFICIENTNET_B3, BackBoneKey.EFFICIENTNET_B4, BackBoneKey.EFFICIENTNET_B5,
                 BackBoneKey.EFFICIENTNET_B6, BackBoneKey.EFFICIENTNET_B7]
