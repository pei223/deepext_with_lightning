# from collections import OrderedDict
# from typing import List, Dict, Tuple
#
# import numpy as np
# import torch
# from sklearn.metrics import accuracy_score
#
# from .base_metric import BaseMetric
# from .metric_keys import DetailMetricKey
#
#
# class ClassificationAccuracyByClasses(BaseMetric):
#     def __init__(self, label_names: List[str], val_key: DetailMetricKey = DetailMetricKey.KEY_TOTAL):
#         assert val_key in [DetailMetricKey.KEY_TOTAL, DetailMetricKey.KEY_AVERAGE]
#         self.label_names = label_names
#         self.correct_by_classes = [0 for _ in range(len(self.label_names))]
#         self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]
#         self._val_key = val_key
#
#     def clone_empty(self) -> 'ClassificationAccuracyByClasses':
#         return ClassificationAccuracyByClasses(self.label_names.copy(), self._val_key)
#
#     def clone(self) -> 'ClassificationAccuracyByClasses':
#         new_metric = self.clone_empty()
#         new_metric.correct_by_classes = self.correct_by_classes.copy()
#         new_metric.incorrect_by_classes = self.incorrect_by_classes.copy()
#         return new_metric
#
#     def calc_one_batch(self, pred: np.ndarray or torch.Tensor, teacher: np.ndarray or torch.Tensor):
#         assert pred.ndim in [1, 2]
#         assert teacher.ndim in [1, 2]
#
#         teacher = teacher.cpu().numpy() if isinstance(teacher, torch.Tensor) else teacher
#         pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
#         if pred.ndim == 2:
#             pred = pred.argmax(-1)
#         if teacher.ndim == 2:
#             teacher = teacher.argmax(-1)
#         result_flags: np.ndarray = (pred == teacher)
#         for label in range(len(self.label_names)):
#             label_index = (teacher == label)
#             class_result_flags = result_flags[label_index]
#             correct = np.count_nonzero(class_result_flags)
#             incorrect = class_result_flags.shape[0] - correct
#             self.correct_by_classes[label] += correct
#             self.incorrect_by_classes[label] += incorrect
#
#     def calc_summary(self) -> Tuple[float, Dict[str, float]]:
#         result = OrderedDict()
#         total_correct, total_incorrect = 0, 0
#         avg_acc = 0.0
#         for i, label_name in enumerate(self.label_names):
#             correct, incorrect = self.correct_by_classes[i], self.incorrect_by_classes[i]
#             result[label_name] = correct / (correct + incorrect) if correct + incorrect > 0 else 0
#             total_correct += correct
#             total_incorrect += incorrect
#             avg_acc += result[label_name]
#         result[DetailMetricKey.KEY_TOTAL.value] = total_correct / (total_correct + total_incorrect)
#         result[DetailMetricKey.KEY_AVERAGE.value] = avg_acc / len(self.label_names)
#         return result[self._val_key.value], result
#
#     def clear(self):
#         self.correct_by_classes = [0 for _ in range(len(self.label_names))]
#         self.incorrect_by_classes = [0 for _ in range(len(self.label_names))]
#
#     def __add__(self, other: 'ClassificationAccuracyByClasses') -> 'ClassificationAccuracyByClasses':
#         if not isinstance(other, ClassificationAccuracyByClasses):
#             raise RuntimeError(f"Bad class type. expected: {ClassificationAccuracyByClasses.__name__}")
#         if len(self.label_names) != len(other.label_names):
#             raise RuntimeError(
#                 f"Label count must be same. but self is {len(self.label_names)} and other is {len(other.label_names)}")
#         new_metric = self.clone_empty()
#         for i in range(len(self.correct_by_classes)):
#             new_metric.correct_by_classes[i] = self.correct_by_classes[i] + other.correct_by_classes[i]
#             new_metric.incorrect_by_classes[i] = self.incorrect_by_classes[i] + other.incorrect_by_classes[i]
#         return new_metric
#
#     def __truediv__(self, num: int) -> 'ClassificationAccuracyByClasses':
#         return self.clone()
# # TODO Recall/Precision/F Score
