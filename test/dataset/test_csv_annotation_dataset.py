import numpy as np
from collections import OrderedDict

from deepext_with_lightning.dataset import DatasetSplitter
from deepext_with_lightning.dataset.classification import CSVAnnotationDataset, CSVAnnotationDatasetWithOverSampling

n_classes = 3
dummy_dataset = CSVAnnotationDataset("hoge", OrderedDict({
    "test0-1": 0,
    "test1-1": 1,
    "test1-2": 1,
    "test1-3": 1,
    "test1-4": 1,
    "test1-5": 1,
    "test1-6": 1,
    "test1-7": 1,
    "test1-8": 1,
    "test2-1": 2,
    "test2-2": 2,
    "test2-3": 2,
    "test2-4": 2,
}), None)
origin_distribution = [1, 8, 4]


def test_CSVAnnotationDataset():
    assert len(dummy_dataset) == 13
    assert dummy_dataset.labels_distribution(n_classes) == origin_distribution


def test_split_dataset():
    indices = np.array([1, 2, 3, 9, 10])
    split_dataset = dummy_dataset.split_dataset(indices)
    assert len(split_dataset) == 5
    assert split_dataset.labels_distribution(n_classes) == [0, 3, 2]


def test_over_sampling():
    over_sampling_rate = [1, 1, 3]
    over_sampling_dataset = dummy_dataset.apply_over_sampling(over_sampling_rate)
    assert over_sampling_dataset.labels_distribution(n_classes) == [1, 8, 4 * 3]
    assert len(over_sampling_dataset) == 1 + 8 + 4 * 3


def test_split_with_over_sampling():
    over_sampling_rate = [1, 5, 10]
    test_ratio = 0.3

    def apply_over_sampling(dataset: CSVAnnotationDataset, indices):
        return dataset.split_dataset(indices).apply_over_sampling(over_sampling_rate)

    def apply_split(dataset: CSVAnnotationDataset, indices):
        return dataset.split_dataset(indices)

    root_dist = dummy_dataset.labels_distribution(n_classes)
    train_dataset, test_dataset = DatasetSplitter().split_train_test(test_ratio, root_dataset=dummy_dataset,
                                                                     train_transforms=None, test_transforms=None,
                                                                     generate_train_subset_callback=apply_over_sampling,
                                                                     generate_test_subset_callback=apply_split)
    total_dist = []
    train_dist = train_dataset.root_dataset.labels_distribution(n_classes)
    test_dist = test_dataset.root_dataset.labels_distribution(n_classes)
    for label in range(len(over_sampling_rate)):
        total_dist.append(int(train_dist[label] / over_sampling_rate[label]) + test_dist[label])
    assert root_dist == total_dist


def test_under_sampling():
    under_sampling_rate = [1, 0.5, 1.]
    expected_dist = [1, int(8 * 0.5), 4]
    under_sampling_dataset = dummy_dataset.apply_under_sampling(under_sampling_rate)
    assert under_sampling_dataset.labels_distribution() == expected_dist
    assert len(under_sampling_dataset) == sum(expected_dist)


def test_split_with_under_sampling():
    under_sampling_rate = [1, 0.5, 1.]
    test_ratio = 0.5

    def apply_under_sampling(dataset: CSVAnnotationDataset, indices):
        return dataset.split_dataset(indices).apply_under_sampling(under_sampling_rate)

    def apply_split(dataset: CSVAnnotationDataset, indices):
        return dataset.split_dataset(indices)

    root_dist = dummy_dataset.labels_distribution(n_classes)
    train_dataset, test_dataset = DatasetSplitter().split_train_test(test_ratio, root_dataset=dummy_dataset,
                                                                     train_transforms=None, test_transforms=None,
                                                                     generate_train_subset_callback=apply_under_sampling,
                                                                     generate_test_subset_callback=apply_split)
    total_dist = []
    train_dist = train_dataset.root_dataset.labels_distribution()
    test_dist = test_dataset.root_dataset.labels_distribution(n_classes)
    for label in range(len(under_sampling_rate)):
        total_dist.append(int(train_dist[label] / under_sampling_rate[label]) + test_dist[label])
    assert root_dist == total_dist
