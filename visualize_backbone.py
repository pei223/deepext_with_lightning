import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext_with_lightning.models.layers.backbone_key import BackBoneKey
from deepext_with_lightning.models.layers.subnetwork import create_backbone
from deepext_with_lightning.image_process.drawer import combine_heatmap
import cv2
import numpy as np

backbone_key = BackBoneKey.RESNET_50

origin_image = cv2.imread("<Image file path>")
origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

image_size = (512, 512)

transforms = A.Compose([
    A.Resize(image_size[0], image_size[1]),
    ToTensorV2(),
])

image = transforms(image=origin_image)["image"]
image = image.view(-1, image.shape[0], image.shape[1], image.shape[2]).float()
image /= 255.

subnet = create_backbone(backbone_key, pretrained=True)
subnet.eval()

for i in range(3):
    heatmap = subnet.forward(image)[i]

    heatmap = torch.sum(heatmap, dim=1).view(heatmap.shape[-2], heatmap.shape[-1])

    heatmap = cv2.resize(heatmap.cpu().detach().numpy(), origin_image.shape[:2][::-1])

    min_val = np.min(heatmap)
    max_val = np.max(heatmap)
    heatmap = (heatmap - min_val) / (max_val - min_val)
    heatmap = (heatmap * 255).astype("uint8")

    result_image = combine_heatmap(origin_image, heatmap, origin_alpha=0.0)
    cv2.imwrite(f"{backbone_key.value}_result_{i + 1}.png", result_image)
    # cv2.imshow("", result_image)
    # cv2.waitKey()
