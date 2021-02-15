from typing import List
import numpy as np
from PIL import Image
import albumentations as A

from ..image_process.convert import pil_to_cv
from ..transforms import ImageToOneHot


class AlbumentationsSegmentationWrapperTransform:
    def __init__(self, albumentations_transforms: A.Compose, class_num: int, is_image_normalize=True,
                 require_onehot=True, ignore_indices: List[int] = None):
        self._albumentations_transforms = albumentations_transforms
        self._require_onehot = require_onehot
        self._to_onehot = ImageToOneHot(class_num)
        self._is_image_normalize = is_image_normalize
        self._ignore_indices = ignore_indices

    def __call__(self, image: Image.Image or np.ndarray, teacher: Image.Image or np.ndarray):
        if isinstance(image, Image.Image):
            image = pil_to_cv(image)
        if isinstance(teacher, Image.Image):
            teacher = np.array(teacher)
        for ignore_index in self._ignore_indices:
            teacher[teacher == ignore_index] = 0

        result_dict = self._albumentations_transforms(image=image, mask=teacher)
        image, teacher = result_dict["image"], result_dict["mask"]
        if teacher.ndim == 2:
            teacher = teacher.expand([1, ] + list(teacher.shape))
        if self._require_onehot:
            teacher = self._to_onehot(teacher)
        if self._is_image_normalize:
            image = image.float() / 255.
        return image, teacher
