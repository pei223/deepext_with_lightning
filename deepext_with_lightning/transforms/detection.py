from typing import List
import numpy as np
from PIL import Image
import albumentations as A

from ..image_process.convert import pil_to_cv


class AlbumentationsDetectionWrapperTransform:
    def __init__(self, albumentations_transform_ls: List, annotation_transform=None, is_image_normalize=True,
                 data_format='pascal_voc'):
        self._albumentations_transforms = A.Compose(albumentations_transform_ls,
                                                    bbox_params=A.BboxParams(label_fields=['category_ids'],
                                                                             format=data_format))
        self._annotation_transform = annotation_transform
        self._is_image_normalize = is_image_normalize

    def __call__(self, image: Image.Image or np.ndarray, teacher: Image.Image or np.ndarray):
        if self._annotation_transform is not None:
            teacher = self._annotation_transform(teacher)
        if isinstance(image, Image.Image):
            image = pil_to_cv(image)
        if isinstance(teacher, Image.Image):
            teacher = np.array(teacher)
        bboxes = teacher[:, :4] if teacher.shape[0] > 0 else []
        labels = teacher[:, 4] if teacher.shape[0] > 0 else []
        result_dict = self._albumentations_transforms(image=image, bboxes=bboxes, category_ids=labels)
        image = result_dict["image"]
        bboxes = result_dict["bboxes"]
        labels = result_dict["category_ids"]
        teacher = np.concatenate([bboxes, np.array(labels).reshape(-1, 1)], axis=1) if len(bboxes) > 0 else []
        if self._is_image_normalize:
            image = image.float() / 255.
        return image, teacher
