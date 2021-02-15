from typing import List
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

from .common import create_filepath_ls


class IndexImageDataset(Dataset):
    @staticmethod
    def create(image_dir_path: str, index_image_dir_path: str, transforms, valid_suffixes: List[str] = None):
        image_path_ls = create_filepath_ls(image_dir_path, valid_suffixes)
        return IndexImageDataset(image_path_ls, image_dir_path, index_image_dir_path, transforms)

    def __init__(self, image_filename_ls: List[str], image_dir: str, index_image_dir: str,
                 transform):
        self._transform = transform
        self._image_dir = Path(image_dir)
        self._index_image_dir = Path(index_image_dir)
        self._image_filename_ls = image_filename_ls

    def __len__(self):
        return len(self._image_filename_ls)

    def __getitem__(self, idx: int):
        image_name = self._image_filename_ls[idx]
        image_path = self._image_dir.joinpath(image_name)
        index_image_path = self._index_image_dir.joinpath(f"{Path(image_name).stem}.png")
        image = Image.open(str(image_path)).convert("RGB")
        index_image = Image.open(str(index_image_path))
        if self._transform:
            return self._transform(image, index_image)
        return image, index_image
