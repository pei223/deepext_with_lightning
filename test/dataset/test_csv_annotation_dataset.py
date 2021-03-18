import numpy as np
from collections import OrderedDict

from deepext_with_lightning.dataset import DatasetSplitter
from deepext_with_lightning.dataset.classification import CSVAnnotationDataset

n_classes = 3
dummy_dataset = CSVAnnotationDataset("hoge", OrderedDict({
    "test0-1": 0,
    "test0-2": 0,
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
origin_distribution = [2, 8, 4]


def test_reading_from_file():
    annotation_path = "test/dataset/test_annotation_csv/test_annotation.csv"
    images_path = "test/dataset/test_images"
    dataset = CSVAnnotationDataset.create(images_path, annotation_path, transforms=None)
    data = dataset[1]
    assert len(dataset) == 2
    assert isinstance(data, tuple) and len(data) == 2
    assert isinstance(data[1], int)


def test_distribution():
    assert len(dummy_dataset) == 14
    assert dummy_dataset.labels_distribution(n_classes) == origin_distribution


def test_split_dataset():
    indices = np.array([1, 3, 4, 5, 11])
    split_dataset = dummy_dataset.split_dataset(indices)
    assert len(split_dataset) == 5
    assert split_dataset.labels_distribution(n_classes) == [1, 3, 1]


def test_over_sampling():
    over_sampling_rate = [1, 1, 3]
    over_sampling_dataset = dummy_dataset.apply_over_sampling(over_sampling_rate)
    assert over_sampling_dataset.labels_distribution(n_classes) == [2, 8, 4 * 3]
    assert len(over_sampling_dataset) == 2 + 8 + 4 * 3


def test_split_with_over_sampling():
    over_sampling_rate = [1, 5, 10]
    test_ratio = 0.5

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
        total_dist.append(round(train_dist[label] / over_sampling_rate[label]) + test_dist[label])
    assert root_dist[0] - 1 <= total_dist[0] <= root_dist[0] + 1
    assert root_dist[1] - 1 <= total_dist[1] <= root_dist[1] + 1
    assert root_dist[2] - 1 <= total_dist[2] <= root_dist[2] + 1


def test_under_sampling():
    under_sampling_rate = [1, 0.5, 1.]
    expected_dist = [2, int(8 * 0.5), 4]
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
        total_dist.append(round(train_dist[label] / under_sampling_rate[label]) + test_dist[label])
    assert root_dist[0] - 1 <= total_dist[0] <= root_dist[0] + 1
    assert root_dist[1] - 1 <= total_dist[1] <= root_dist[1] + 1
    assert root_dist[2] - 1 <= total_dist[2] <= root_dist[2] + 1
