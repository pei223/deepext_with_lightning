from typing import Tuple, Generator, TypeVar, Callable
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

from .common import TransformsWrapperDataset

T_DATASET = TypeVar('T_DATASET', bound=Dataset)


def root_dataset_to_subset(root_dataset: T_DATASET, indices: np.ndarray):
    return Subset(root_dataset, indices)


class DatasetSplitter:
    def split_train_test(self, test_ratio: float, root_dataset: T_DATASET, train_transforms, test_transforms,
                         generate_train_subset_callback: Callable[[T_DATASET, np.ndarray], Dataset] = None,
                         generate_test_subset_callback: Callable[[T_DATASET, np.ndarray], Dataset] = None) \
            -> Tuple[TransformsWrapperDataset, TransformsWrapperDataset]:
        generate_train_subset_callback = generate_train_subset_callback or root_dataset_to_subset
        generate_test_subset_callback = generate_test_subset_callback or root_dataset_to_subset
        data_len = len(root_dataset)
        test_num = int(data_len * test_ratio)
        indices = np.random.permutation(data_len)
        test_indices = indices[:test_num]
        train_indices = indices[test_num:]
        return TransformsWrapperDataset(generate_train_subset_callback(root_dataset, train_indices), train_transforms), \
               TransformsWrapperDataset(generate_test_subset_callback(root_dataset, test_indices), test_transforms)

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
