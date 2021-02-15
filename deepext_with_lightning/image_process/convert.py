from typing import Tuple
from warnings import warn

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

to_tensor_transform = ToTensor()
to_pil_transform = ToPILImage()


def to_4dim(tensor):
    return tensor.view(-1, tensor.shape[0], tensor.shape[1], tensor.shape[2])


def pil_to_cv(image: Image.Image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        return new_image
    elif new_image.shape[2] == 3:  # カラー
        return cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        return cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)


def pil_to_tensor(image: Image):
    return to_tensor_transform(image)


def cv_to_pil(image: np.ndarray):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        return Image.fromarray(new_image)
    elif new_image.shape[2] == 3:  # カラー
        return Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    elif new_image.shape[2] == 4:  # 透過
        return Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA))
    assert False, f"Invalid shape {new_image.shape}"


def cv_to_tensor(cv2image: np.ndarray):
    img_array = cv2image.transpose(2, 0, 1)
    return torch.FloatTensor(torch.from_numpy(img_array.astype('float32')))


def tensor_to_cv(tensor: torch.Tensor):
    if tensor.ndim == 2:
        return tensor.cpu().detach().numpy()
    if tensor.ndim == 3:
        return tensor.permute(1, 2, 0).cpu().detach().numpy()
    raise ValueError(f"Invalid tensor shape: {tensor.shape}")


def tensor_to_pil(tensor: torch.Tensor):
    return to_pil_transform(tensor)


def resize_image(img: Image.Image or torch.Tensor or np.ndarray, size: Tuple[int, int]):
    """
    NOTE Tensorオブジェクトならリサイズ不能のためそのまま返す.
    :param img:
    :param size:
    """
    if isinstance(img, Image.Image):
        return img.resize(size)
    elif isinstance(img, np.ndarray):
        return cv2.resize(img, size)
    warn("Resizing tensor is not enable.")
    return img


def get_image_size(img: Image.Image or torch.Tensor or np.ndarray):
    if isinstance(img, Image.Image):
        return img.size
    elif isinstance(img, np.ndarray):
        return img.shape[:2]
    return img.shape[1:]


def img_to_tensor(img: Image.Image or torch.Tensor or np.ndarray):
    if isinstance(img, Image.Image):
        return pil_to_tensor(img)
    if isinstance(img, np.ndarray):
        return cv_to_tensor(img)
    if isinstance(img, torch.Tensor):
        return img
    raise ValueError(f"Invalid data type: {type(img)}")


def img_to_pil(img: Image.Image or torch.Tensor or np.ndarray):
    if isinstance(img, torch.Tensor):
        return tensor_to_pil(img)
    if isinstance(img, np.ndarray):
        return cv_to_pil(img)
    if isinstance(img, Image.Image):
        return img
    raise ValueError(f"Invalid data type: {type(img)}")


def img_to_cv(img: Image.Image or torch.Tensor or np.ndarray):
    if isinstance(img, torch.Tensor):
        return tensor_to_cv(img)
    if isinstance(img, Image.Image):
        return pil_to_cv(img)
    if isinstance(img, np.ndarray):
        return img
    raise ValueError(f"Invalid data type: {type(img)}")


def normalize255(img: np.ndarray):
    return (img * 255).astype("uint8")


def normalize1(img: np.ndarray):
    return (img / 255).astype("float32")


def try_cuda(e):
    if torch.cuda.is_available() and hasattr(e, "cuda"):
        return e.cuda()
    return e
