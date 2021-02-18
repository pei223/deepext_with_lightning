import albumentations as A
import torch
import numpy as np
from PIL import Image

from ..image_process.convert import pil_to_cv


class ImageToOneHot:
    def __init__(self, class_num: int, ignore_index: int = None):
        self._class_num = class_num
        self._ignore_index = ignore_index

    def __call__(self, img: torch.Tensor):
        assert img.ndim == 3
        img = img.permute(1, 2, 0).long()
        if self._ignore_index:
            img[img == self._ignore_index] = 0
        img = torch.eye(self._class_num)[img]
        return img.view(img.shape[0], img.shape[1], img.shape[3]).permute(2, 0, 1)


class PilToTensor:
    def __call__(self, img: Image.Image):
        img_array = np.array(img)
        if img_array.ndim == 2:
            img_array = img_array[:, :, None]
        img_array = img_array.transpose((2, 0, 1))
        return torch.from_numpy(img_array) % 255


class SegmentationLabelSmoothing:
    def __init__(self, class_num: int, epsilon=0.1):
        self._class_num = class_num
        self._epsilon = epsilon
        self._to_onehot = ImageToOneHot(self._class_num)

    def __call__(self, label_img: torch.Tensor):
        assert label_img.ndim == 2 or label_img.ndim == 3
        if label_img.ndim == 2:
            label_img = self._to_onehot(label_img)
        return label_img * (1. - self._epsilon) + self._epsilon


class LabelToOneHot:
    def __init__(self, class_num: int):
        self._class_num = class_num

    def __call__(self, label: torch.Tensor):
        assert label.ndim == 1
        return torch.eye(self._class_num)[label]


class ClassificationLabelSmoothing:
    def __init__(self, class_num: int, epsilon=0.1):
        self._class_num = class_num
        self._epsilon = epsilon
        self._to_onehot = LabelToOneHot(self._class_num)

    def __call__(self, label: torch.Tensor):
        assert label.ndim == 1 or label.ndim == 2
        if label.ndim == 1:
            label = self._to_onehot(label)
        return label * (1. - self._epsilon) + self._epsilon


class AlbumentationsOnlyImageWrapperTransform:
    def __init__(self, albumentations_transforms: A.Compose, is_image_normalize=True):
        self._albumentations_transforms = albumentations_transforms
        self._is_image_normalize = is_image_normalize

    def __call__(self, image: Image.Image or np.ndarray):
        if isinstance(image, Image.Image):
            image = pil_to_cv(image)

        result_dict = self._albumentations_transforms(image=image)
        result_image = result_dict["image"]
        if self._is_image_normalize:
            result_image = result_image / 255.
        return result_image
