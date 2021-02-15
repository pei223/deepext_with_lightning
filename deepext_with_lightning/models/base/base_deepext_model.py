from abc import ABCMeta, abstractmethod
from warnings import warn
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils import mobile_optimizer


class BaseDeepextModel(LightningModule, metaclass=ABCMeta):
    def save_model_for_mobile(self, width: int, height: int, out_filepath: str, for_os="cpu"):
        torch_model = self.to("cpu")
        torch_model.eval()

        if for_os == "cpu":
            example = torch.rand(1, 3, width, height).to("cpu")
            traced_script_module = torch.jit.trace(torch_model, example)
            traced_script_module.save(out_filepath)
            return

        script_model = torch.jit.script(torch_model)
        if for_os == "android":
            mobile_optimizer.optimize_for_mobile(script_model, backend="Vulkan")
        elif for_os == "ios":
            mobile_optimizer.optimize_for_mobile(script_model, backend="metal")
        torch.jit.save(script_model, out_filepath)

        # scripted_model = torch.jit.script(torch_model)
        # opt_model = mobile_optimizer.optimize_for_mobile(scripted_model)
        # torch.jit.save(opt_model, out_filepath, _use_new_zipfile_serialization=False)

    def generate_model_name(self, suffix: str = "") -> str:
        return f"{self.__class__.__name__}{suffix}"

    def cv_image_to_tensor(self, img: torch.Tensor or np.ndarray, require_normalize=False):
        """
        :param img: 3 dimention tensor or ndarray
        :param require_normalize:
        :return: 4 dimention tensor, origin image
        """
        assert img.ndim == 3, f"Invalid data shape: {img.shape}. Expected 3 dimension"
        if isinstance(img, np.ndarray):
            img_tensor = torch.tensor(img.transpose(2, 0, 1))
            origin_img = img
        elif isinstance(img, torch.Tensor):
            img_tensor = img
            origin_img = img.cpu().numpy().transpose(1, 2, 0)
        else:
            assert False, f"Invalid data type: {type(img)}"

        if require_normalize:
            img_tensor = img_tensor.float() / 255.
        else:
            origin_img = (origin_img * 255).astype('uint8')

        img_tensor = img_tensor.view(-1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        return img_tensor, origin_img
