import warnings
from typing import Tuple

warnings.simplefilter('ignore')
import torch


def gen_random_tensor(batch_size: int, img_size: Tuple[int, int], channels: int = 3):
    return torch.randn([batch_size, channels, img_size[0], img_size[1]])


def assert_tensor_shape(result_tensor: torch.Tensor, expected_shape: Tuple, message: str):
    assert result_tensor.shape == expected_shape, message
