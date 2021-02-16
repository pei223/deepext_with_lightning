import warnings

warnings.simplefilter('ignore')

import numpy as np

from deepext_with_lightning.metrics.object_detection import DetectionIoU, RecallPrecision, MeanAveragePrecision

n_classes = 3

preds = np.array([
    [
        [0, 0, 50, 50, 0, 0.9],
        [40, 40, 90, 90, 0, 0.8],
        [100, 0, 150, 50, 0, 0.7],
        [0, 250, 50, 300, 1, 0.8],
        [0, 400, 50, 450, 1, 0.6],
        [100, 300, 150, 350, 1, 0.6],
        [140, 40, 190, 90, 2, 0.8],
    ]
])

teachers = np.array([
    [
        [2, 2, 52, 52, 0],
        [100, 0, 150, 50, 0],
        [0, 400, 50, 450, 1],
        [200, 200, 250, 250, 2],
        [90, 290, 160, 360, 2],
    ]
])


def test_detection_recall_precision():
    metric = RecallPrecision(n_classes=n_classes)
    metric(preds, teachers)
    metric_value = metric.compute()
    recall, precision, f_score = metric_value
    assert isinstance(recall.item(), float) and 0.67 > recall.item() > 0.66, f"mRecall is {recall.item()}"
    assert isinstance(precision.item(), float) and 0.34 > precision.item() > 0.3, f"mPrecision is {precision.item()}"
    assert isinstance(f_score.item(), float) and 0.44 > f_score.item() > 0.43, f"mF Score is {f_score.item()}"


def test_detection_recall_precision_by_classes():
    metric = RecallPrecision(n_classes=n_classes, by_classes=True)
    metric(preds, teachers)
    metric_value = metric.compute()
    recall, precision, f_score = metric_value
    assert recall.shape == (n_classes,)
    assert precision.shape == (n_classes,)
    assert f_score.shape == (n_classes,)


def test_detection_iou():
    metric = DetectionIoU(n_classes=n_classes)
    metric(preds, teachers)
    metric_value = metric.compute()
    assert isinstance(metric_value.item(),
                      float) and 0.33 > metric_value.item() > 0.32, f"mIoU is {metric_value.item()}"


def test_detection_iou_by_classes():
    metric = DetectionIoU(n_classes=n_classes, by_classes=True)
    metric(preds, teachers)
    metric_value = metric.compute()
    assert metric_value.shape == (n_classes,)


def test_mAP():
    metric = MeanAveragePrecision(n_classes)
    metric(preds, teachers)
    metric_value = metric.compute()
    assert isinstance(metric_value.item(), float)
