from pathlib import Path
import os
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dotenv import load_dotenv

from deepext.layers.loss import ClassificationFocalLoss
from deepext.data.dataset import CSVAnnotationDataset, DatasetSplitter
from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import ClassificationModel, AttentionClassificationModel
from deepext.models.classification import EfficientNet, AttentionBranchNetwork
from deepext.trainer import Trainer, LearningCurveVisualizer, CosineDecayScheduler
from deepext.trainer.callbacks import ModelCheckout, GenerateAttentionMapCallback
from deepext.data.transforms import AlbumentationsClsWrapperTransform
from deepext.metrics.classification import *
from deepext.utils import *
from deepext.utils.dataset_util import create_label_list_and_dict

load_dotenv("envs/classification.env")

# File/Directory path
train_images_dir_path = os.environ.get("TRAIN_IMAGES_DIR_PATH")
train_annotation_file_path = os.environ.get("TRAIN_ANNOTATION_FILE_PATH")
test_images_dir_path = os.environ.get("TEST_IMAGES_DIR_PATH")
test_annotation_file_path = os.environ.get("TEST_ANNOTATION_FILE_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")

load_weight_path = os.environ.get("MODEL_WEIGHT_PATH")
saved_weights_dir_path = os.environ.get("SAVED_WEIGHTS_DIR_PATH")
progress_dir = os.environ.get("PROGRESS_DIR_PATH")
# Model params
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
n_classes = int(os.environ.get("N_CLASSES"))
# Learning params
batch_size = int(os.environ.get("BATCH_SIZE"))
lr = float(os.environ.get("LR"))
epoch = int(os.environ.get("EPOCH"))

label_names, label_dict = create_label_list_and_dict(label_file_path)

# TODO Learning detail params
lr_scheduler = CosineDecayScheduler(max_lr=lr, max_epochs=epoch, warmup_epochs=0)
ignore_indices = [255, ]

# TODO Data augmentation
train_transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.HorizontalFlip(),
    A.RandomResizedCrop(width=width, height=height, scale=(0.5, 2.0)),
    A.CoarseDropout(max_height=int(height / 5), max_width=int(width / 5)),
    A.RandomBrightnessContrast(),
    ToTensorV2(),
]))

test_transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]))

# dataset/dataloader
if test_images_dir_path == "":
    test_ratio = float(os.environ.get("TEST_RATIO"))
    root_dataset = CSVAnnotationDataset.create(train_images_dir_path,
                                               transforms=None,
                                               annotation_csv_filepath=train_annotation_file_path)
    train_dataset, test_dataset = DatasetSplitter().split_train_test(test_ratio, root_dataset,
                                                                     train_transforms, test_transforms)
else:
    train_dataset = CSVAnnotationDataset.create(image_dir=train_images_dir_path,
                                                annotation_csv_filepath=train_annotation_file_path,
                                                transforms=train_transforms)
    test_dataset = CSVAnnotationDataset.create(image_dir=test_images_dir_path,
                                               annotation_csv_filepath=test_annotation_file_path,
                                               transforms=test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# TODO Model detail params
voc_focal_loss = ClassificationFocalLoss(n_classes)
model: ClassificationModel = try_cuda(EfficientNet(num_classes=n_classes, network='efficientnet-b0'))
# model: ClassificationModel = try_cuda(AttentionBranchNetwork(n_classes=n_classes, backbone=BackBoneKey.RESNET_18))

if load_weight_path and load_weight_path != "":
    model.load_weight(load_weight_path)

# TODO Train detail params
# Metrics/Callbacks
callbacks = [ModelCheckout(per_epoch=int(epoch / 2), model=model, our_dir=saved_weights_dir_path)]
if isinstance(model, AttentionClassificationModel):
    callbacks.append(GenerateAttentionMapCallback(model=model, output_dir=progress_dir, per_epoch=5,
                                                  dataset=test_dataset, label_names=label_names))
metric_ls = [ClassificationAccuracyByClasses(label_names)]
metric_for_graph = ClassificationAccuracyByClasses(label_names, val_key=DetailMetricKey.KEY_AVERAGE)
learning_curve_visualizer = LearningCurveVisualizer(metric_name="Accuracy", ignore_epoch=0,
                                                    save_filepath="classification_learning_curve.png")

# Training.
Trainer(model, learning_curve_visualizer=learning_curve_visualizer).fit(train_data_loader=train_dataloader,
                                                                        test_data_loader=test_dataloader,
                                                                        epochs=epoch, callbacks=callbacks,
                                                                        epoch_lr_scheduler_func=lr_scheduler,
                                                                        metric_ls=metric_ls,
                                                                        metric_for_graph=metric_for_graph,
                                                                        calc_metrics_per_epoch=5)
