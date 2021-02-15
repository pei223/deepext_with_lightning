import warnings

warnings.simplefilter('ignore')

import torch

from deepext_with_lightning.metrics.segmentation import SegmentationIoU

n_classes = 3
a = torch.tensor([[[1, 0, 1, 0], [2, 2, 0, 0], [0, 0, 1, 2]]])
b = torch.tensor([[[0, 0, 1, 1], [2, 2, 0, 0], [0, 0, 1, 1]]])


def test_segmentation_iou():
    metric = SegmentationIoU(n_classes=n_classes)
    metric(a, b)
    metric_value = metric.compute()
    assert isinstance(metric_value.item(), float)


def test_segmentation_iou_without_background():
    metric = SegmentationIoU(n_classes=n_classes)
    metric(a, b)
    metric_value = metric.compute()

    metric_without_background = SegmentationIoU(n_classes=n_classes, without_background_class=True)
    metric_without_background(a, b)
    metric_value_without_background = metric_without_background.compute()
    assert isinstance(metric_value_without_background.item(), float)
    assert metric_value > metric_value_without_background


def test_segmentation_iou_by_classes():
    metric = SegmentationIoU(n_classes=n_classes, by_classes=True)
    metric(a, b)
    metric_value = metric.compute()
    assert metric_value.shape == (n_classes,)
