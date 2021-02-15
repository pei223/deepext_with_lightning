from collections import OrderedDict
from typing import Dict, List
from warnings import warn
from torch.utils.data import Dataset
import csv
from PIL import Image
from pathlib import Path


class CSVAnnotationDataset(Dataset):
    @staticmethod
    def create(image_dir: str, annotation_csv_filepath: str, transforms,
               label_dict: Dict[str, int] = None) -> 'CSVAnnotationDataset':
        """
        Create dataset from CSV file.
        If CSV column 2 value is string, required label_dict arg.
        :param label_dict:
        :param image_dir:
        :param annotation_csv_filepath: 
        :param transforms:
        :return:
        """
        filename_label_dict = CSVAnnotationDataset._create_filename_label_dict(annotation_csv_filepath, label_dict)
        return CSVAnnotationDataset(image_dir, filename_label_dict=filename_label_dict, transforms=transforms)

    @staticmethod
    def _create_filename_label_dict(annotation_csv_filepath: str, label_dict: Dict[str, int] = None) -> OrderedDict:
        filename_label_dict = OrderedDict()
        with open(annotation_csv_filepath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                filename = Path(row[0]).name
                label = row[1]
                if not label.isdigit():
                    if label_dict is None:
                        raise RuntimeError("Required dict transform label name to class number.")
                    label_num = label_dict.get(label)
                    if label_num is None:
                        warn(f"Invalid label name: {label},  {row}")
                        continue
                    filename_label_dict[filename] = label_num
                    continue
                filename_label_dict[filename] = int(label)
        return filename_label_dict

    def __init__(self, image_dir: str, filename_label_dict: OrderedDict, transforms):
        self._image_dir = image_dir
        self._transforms = transforms
        self._filepath_label_dict = filename_label_dict

    def labels_distribution(self) -> List[int]:
        """
        summarize labels distribution.
        :return: result[label] = count
        """
        result = [0 for _ in range(max(self._filepath_label_dict.values()))]
        for label in self._filepath_label_dict.values():
            result[label] += 1
        return result

    def __len__(self):
        return len(self._filepath_label_dict)

    def __getitem__(self, idx):
        filename, label = list(self._filepath_label_dict.items())[idx]
        filepath = Path(self._image_dir).joinpath(filename)
        img = Image.open(str(filepath))
        img = img.convert("RGB")
        if self._transforms:
            return self._transforms(img, label)
        return img, label
