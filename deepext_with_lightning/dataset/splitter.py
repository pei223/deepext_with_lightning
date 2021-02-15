from typing import Tuple, Generator
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

from .common import TransformsWrapperDataset


class DatasetSplitter:
    def split_train_test(self, test_ratio: float, root_dataset: Dataset, train_transforms, test_transforms) \
            -> Tuple[Dataset, Dataset]:
        data_len = len(root_dataset)
        test_num = int(data_len * test_ratio)
        indices = np.random.permutation(data_len)
        test_indices = indices[:test_num]
        train_indices = indices[test_num:]
        return TransformsWrapperDataset(Subset(root_dataset, train_indices),
                                        transforms=train_transforms), TransformsWrapperDataset(
            Subset(root_dataset, test_indices), transforms=test_transforms)

    def create_kfold_generator(self, n_fold: int, root_dataset: Dataset, train_transforms, test_transforms) -> \
            Generator[Tuple[Dataset, Dataset], None, None]:
        assert 1 < n_fold < 100
        kf = KFold(n_fold)
        dummy = np.zeros(len(root_dataset), 1)
        for _fold, (train_indices, test_indices) in enumerate(kf.split(dummy)):
            train_dataset = TransformsWrapperDataset(Subset(root_dataset, train_indices),
                                                     transforms=train_transforms)
            test_dataset = TransformsWrapperDataset(Subset(root_dataset, test_indices),
                                                    transforms=test_transforms)
            yield train_dataset, test_dataset
