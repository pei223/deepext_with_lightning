from typing import List, Tuple, Union

import numpy as np
import torch

import pytorch_lightning as pl


def calc_area(bbox: np.ndarray):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def calc_bbox_overlap_union_iou(pred: np.ndarray or None, teacher: np.ndarray) -> Tuple[float, float, float]:
    """
    :param pred: ndarray (4, )
    :param teacher: ndarray (4, )
    :return: overlap, union, iou
    """
    teacher_area = (teacher[2] - teacher[0]) * (teacher[3] - teacher[1])
    if pred is None:
        return 0.0, teacher_area, 0.0

    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])

    intersection_width = np.maximum(np.minimum(pred[2], teacher[2]) - np.maximum(pred[0], teacher[0]), 0)
    intersection_height = np.maximum(np.minimum(pred[3], teacher[3]) - np.maximum(pred[1], teacher[1]), 0)

    overlap = intersection_width * intersection_height
    union = teacher_area + pred_area - overlap
    iou = overlap / union
    return overlap, union, iou


class DetectionIoU(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes: bool = False):
        super().__init__()
        self._n_classes = n_classes
        self._by_classes = by_classes
        self.add_state("image_count_by_classes", default=torch.tensor([0. for _ in range(n_classes)]),
                       dist_reduce_fx="sum")
        self.add_state("total_iou_by_classes", default=torch.tensor([0. for _ in range(n_classes)]),
                       dist_reduce_fx="sum")

    def update(self, preds: List[np.ndarray], targets: Union[np.ndarray, torch.Tensor]) -> None:
        targets = targets.cpu().detach().numpy() if isinstance(targets, torch.Tensor) else targets
        # 全探索だと遅いのでクラスごとにまとめておく
        preds_by_class = []
        for pred_bboxes in preds:
            pred_by_class = [[] for _ in range(self._n_classes)]
            for pred_bbox in pred_bboxes:
                pred_by_class[int(pred_bbox[4])].append(pred_bbox)
            preds_by_class.append(pred_by_class)

        for i in range(targets.shape[0]):  # Explore every batch.
            bbox_annotations = targets[i, :, :]
            # Exclude invalid label annotation.
            bbox_annotations = bbox_annotations[bbox_annotations[:, 4] >= 0]

            pred_by_class = preds_by_class[i]

            """
            1画像でラベルごとに計算.
            ラベルごとの面積合計/overlapを計算
            1画像ごとにIoU算出、最終的に画像平均を算出
            """

            total_area_by_classes = [0 for _ in range(self._n_classes)]
            total_overlap_by_classes = [0 for _ in range(self._n_classes)]
            is_label_appeared = [False for _ in range(self._n_classes)]
            for bbox_annotation in bbox_annotations:

                label = int(bbox_annotation[4])
                total_area_by_classes[label] += calc_area(bbox_annotation)
                pred_bboxes = pred_by_class[label]

                if pred_bboxes is None or len(pred_bboxes) == 0:
                    continue

                # Calculate area and overlap by class.
                for pred_bbox in pred_bboxes:
                    overlap, _, _ = calc_bbox_overlap_union_iou(pred_bbox, bbox_annotation)
                    total_overlap_by_classes[label] += overlap
                    if is_label_appeared[label]:
                        continue
                    total_area_by_classes[label] += calc_area(pred_bbox)
                is_label_appeared[label] = True

            for label in range(self._n_classes):
                # Not exist label in this data.
                if total_area_by_classes[label] <= 0:
                    continue
                self.total_iou_by_classes[label] += total_overlap_by_classes[label] / (
                        total_area_by_classes[label] - total_overlap_by_classes[label])
                self.image_count_by_classes[label] += 1

    def compute(self):
        epsilon = 1e-8
        iou_by_classes = self.total_iou_by_classes / (self.image_count_by_classes + epsilon)
        if self._by_classes:
            return iou_by_classes
        return torch.mean(iou_by_classes)


class RecallPrecision(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes: bool = False):
        super().__init__()
        self._n_classes = n_classes
        self._by_classes = by_classes
        self.add_state("tp_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")
        self.add_state("fp_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")
        self.add_state("fn_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")

    def update(self, preds: List[np.ndarray], targets: Union[np.ndarray, torch.Tensor]) -> None:
        """
        :param pred: (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param teacher: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        targets = targets.cpu().detach().numpy() if isinstance(targets, torch.Tensor) else targets
        # 全探索だと遅いのでクラスごとにまとめておく
        preds_by_class = []
        for pred_bboxes in preds:
            pred_by_class = [[] for _ in range(self._n_classes)]
            for pred_bbox in pred_bboxes:
                pred_by_class[int(pred_bbox[4])].append(pred_bbox)
            preds_by_class.append(pred_by_class)

        for i in range(targets.shape[0]):
            bbox_annotations = targets[i, :, :]
            # Exclude invalid label annotation.
            bbox_annotations = bbox_annotations[bbox_annotations[:, 4] >= 0]

            pred_by_class = preds_by_class[i]

            applied_bbox_count_by_classes = [0 for _ in range(self._n_classes)]
            for bbox_annotation in bbox_annotations:
                label = int(bbox_annotation[4])
                pred_bboxes = pred_by_class[label]

                if pred_bboxes is None or len(pred_bboxes) == 0:
                    self.fn_by_classes[label] += 1
                    continue
                # Explore max iou of bbox_annotation
                is_matched = False
                for pred_bbox in pred_bboxes:
                    overlap, union, iou = calc_bbox_overlap_union_iou(pred_bbox, bbox_annotation)
                    if iou >= 0.5:
                        applied_bbox_count_by_classes[label] += 1
                        self.tp_by_classes[label] += 1
                        is_matched = True
                        break
                if not is_matched:
                    self.fn_by_classes[label] += 1

            for label in range(self._n_classes):
                self.fp_by_classes[label] += len(pred_by_class[label]) - applied_bbox_count_by_classes[label]

    def compute(self):
        epsilon = 1e-8
        recall = self.tp_by_classes / (self.tp_by_classes + self.fn_by_classes + epsilon)
        precision = self.tp_by_classes / (self.tp_by_classes + self.fp_by_classes + epsilon)
        f_score = 2. * recall * precision / (recall + precision + epsilon)
        if self._by_classes:
            return recall, precision, f_score
        return torch.mean(recall), torch.mean(precision), torch.mean(f_score)


class mAP(pl.metrics.Metric):
    def __init__(self, n_classes: int):
        self._n_classes = n_classes
        pass

#
#
# # TODO 実装途中
# class mAPByClasses:
#     def __init__(self, n_classes: int):
#         self._n_classes = n_classes
#
#     def __call__(self, results, teachers):
#         average_precisions = [_ for _ in range(self._n_classes)]
#         for label in range(self._n_classes):
#             false_positives = np.zeros((0,))
#             true_positives = np.zeros((0,))
#             scores = np.zeros((0,))
#             num_annotations = 0.0
#             for i in range(len(results)):
#                 detected_labels = []
#                 detections_by_class = results[i][label]
#                 annotations_by_class = teachers[i][label]
#                 num_annotations += annotations_by_class.shape[0]
#
#                 for detection in detections_by_class:
#                     scores = np.append(scores, detection[4])
#
#                     if annotations_by_class.shape[0] == 0:  # False detection
#                         false_positives = np.append(false_positives, 1)
#                         true_positives = np.append(true_positives, 0)
#                         continue
#
#                     overlaps = calc_bbox_overlap(np.expand_dims(detection, axis=0), annotations_by_class)
#                     assigned_annotation = np.argmax(overlaps, axis=1)
#                     max_overlaps = overlaps[0, assigned_annotation]
#
#                     if assigned_annotation not in detected_labels:
#                         false_positives = np.append(false_positives, 0)
#                         true_positives = np.append(true_positives, 1)
#                         detected_labels.append(assigned_annotation)
#                     else:
#                         false_positives = np.append(false_positives, 1)
#                         true_positives = np.append(true_positives, 0)
#             if num_annotations == 0:
#                 average_precisions[label] = 0, 0
#                 continue
#
#             # sort by score
#             indices = np.argsort(-scores)
#             # false_positives = false_positives[indices]
#             # true_positives = true_positives[indices]
#
#             # compute false positives and true positives
#             false_positives = np.cumsum(false_positives)
#             true_positives = np.cumsum(true_positives)
#
#             # compute recall and precision
#             recall = true_positives / num_annotations
#             precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
#
#             # compute average precision
#             average_precision = _compute_ap(recall, precision)
#             average_precisions[label] = average_precision, num_annotations
#
#         return average_precisions
