from typing import Tuple, List, TypeVar
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

T_ROOT_DATASET = TypeVar("T_ROOT_DATASET", bound=Dataset)


class TransformsWrapperDataset(Dataset):
    def __init__(self, root_dataset: T_ROOT_DATASET, transforms):
        super().__init__()
        self._root_dataset = root_dataset
        self._transforms = transforms

    def __len__(self):
        return len(self._root_dataset)

    def __getitem__(self, idx):
        image, label = self._root_dataset[idx]
        if self._transforms is not None:
            return self._transforms(image, label)
        return image, label

    @property
    def root_dataset(self) -> T_ROOT_DATASET:
        return self._root_dataset


def create_filepath_ls(image_dir_path: str, valid_suffixes: List[str] = None) -> List[Path]:
    valid_suffixes = valid_suffixes or ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_path_ls = []
    image_dir = Path(image_dir_path)
    for suffix in valid_suffixes:
        image_path_ls += list(image_dir.glob(suffix))
    image_path_ls.sort()
    return image_path_ls


class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir: str, image_transform):
        self._image_transform = image_transform
        image_dir_path = Path(image_dir)
        self._image_file_path_ls = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.jpeg")) + list(
            image_dir_path.glob("*.png")) + list(image_dir_path.glob("*.bmp"))
        self._current_image_size = None
        self._current_file_path = None

    def __len__(self):
        return len(self._image_file_path_ls)

    def current_image_size(self) -> Tuple[int, int]:
        """
        :return: (width, height)
        """
        return self._current_image_size

    def current_file_path(self) -> str:
        return str(self._current_file_path)

    def __getitem__(self, idx):
        file_path = self._image_file_path_ls[idx]
        self._current_file_path = file_path
        img = Image.open(str(file_path))
        img = img.convert("RGB")
        width, height = img.size[:2]
        self._current_image_size = width, height
        if self._image_transform:
            img, label = self._image_transform(img, None)
        return img
