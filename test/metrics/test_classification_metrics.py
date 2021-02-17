import warnings
import torch

warnings.simplefilter('ignore')

from deepext_with_lightning.metrics.classification import ClassificationAccuracy, ClassificationRecallPrecision

n_classes = 3

preds = torch.tensor([0, 1, 0, 2, 2, 2])
targets = torch.tensor([0, 2, 0, 2, 1, 1])


# Accuracy expected
# [0] = 2/2
# [1] = 0/2
# [2] = 1/2


def test_classification_accuracy():
    metric = ClassificationAccuracy(n_classes=n_classes)
    metric(preds, targets)
    metric_value = metric.compute()
    assert isinstance(metric_value.item(), float)
    assert 0.51 > metric_value.item() > 0.49


def test_classification_accuracy_class_average():
    metric = ClassificationAccuracy(n_classes=n_classes, average=True)
    metric(preds, targets)
    metric_value = metric.compute()
    expected_value = (1. + 0. + 1 / 2) / 3
    assert isinstance(metric_value.item(), float)
    assert expected_value + 0.01 > metric_value.item() > expected_value - 0.01


def test_classification_accuracy_by_classes():
    metric = ClassificationAccuracy(n_classes=n_classes, by_classes=True)
    metric(preds, targets)
    metric_value = metric.compute()
    assert metric_value.shape == (n_classes,)
    assert metric_value[0] == 1.
    assert metric_value[1] == 0
    assert metric_value[2] == 1 / 2


# Recall expected
# [0] = 2/2
# [1] = 0/2
# [2] = 1/2

# Precision expected
# [0] = 2/2
# [1] = 0/1
# [2] = 1/3

def test_recall_precision():
    metric = ClassificationRecallPrecision(n_classes)
    metric(preds, targets)
    recall, precision = metric.compute()
    # excepted
    assert isinstance(recall.item(), float)
    assert isinstance(precision.item(), float)
    expected_recall = 1 / 2
    expected_precision = 1 / 2
    assert expected_recall + 0.01 > recall.item() > expected_recall - 0.01
    assert expected_precision + 0.01 > precision.item() > expected_precision - 0.01


def test_recall_precision_average():
    metric = ClassificationRecallPrecision(n_classes, average=True)
    metric(preds, targets)
    recall, precision = metric.compute()
    expected_recall = (1. + 0. + 1 / 2) / 3
    expected_precision = (1. + 0 + 1 / 3) / 3
    assert isinstance(recall.item(), float)
    assert isinstance(precision.item(), float)
    assert expected_recall + 0.01 > recall.item() > expected_recall - 0.01
    assert expected_precision + 0.01 > precision.item() > expected_precision - 0.01


def test_recall_precision_by_classes():
    metric = ClassificationRecallPrecision(n_classes, by_classes=True)
    metric(preds, targets)
    recall, precision = metric.compute()
    assert recall.shape == (n_classes,)
    assert precision.shape == (n_classes,)
