from pathlib import Path
import os
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dotenv import load_dotenv

from deepext.layers.loss import SegmentationFocalLoss
from deepext.data.dataset import IndexImageDataset, DatasetSplitter
from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import SegmentationModel
from deepext.models.segmentation import UNet, ResUNet, ShelfNet
from deepext.trainer import Trainer, LearningCurveVisualizer, CosineDecayScheduler
from deepext.trainer.callbacks import ModelCheckout, GenerateSegmentationImageCallback
from deepext.data.transforms import AlbumentationsSegmentationWrapperTransform
from deepext.metrics.segmentation import *
from deepext.utils import *

load_dotenv("envs/segmentation.env")

# File/Directory path
train_images_dir = os.environ.get("TRAIN_IMAGES_PATH")
train_annotations_dir = os.environ.get("TRAIN_ANNOTATIONS_PATH")
test_images_dir = os.environ.get("TEST_IMAGES_PATH")
test_annotations_dir = os.environ.get("TEST_ANNOTATIONS_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")

load_weight_path = os.environ.get("MODEL_WEIGHT_PATH")
saved_weights_dir = os.environ.get("SAVED_WEIGHTS_DIR_PATH")
progress_dir = os.environ.get("PROGRESS_DIR_PATH")
# Model params
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
n_classes = int(os.environ.get("N_CLASSES"))
# Learning params
batch_size = int(os.environ.get("BATCH_SIZE"))
lr = float(os.environ.get("LR"))
epoch = int(os.environ.get("EPOCH"))

label_names = []
with open(label_file_path, "r") as file:
    for line in file:
        label_names.append(line.replace("\n", ""))

# TODO Learning detail params
lr_scheduler = CosineDecayScheduler(max_lr=lr, max_epochs=epoch, warmup_epochs=0)
ignore_indices = [255, ]

# TODO Data augmentation
train_transforms = AlbumentationsSegmentationWrapperTransform(A.Compose([
    A.HorizontalFlip(),
    A.RandomResizedCrop(width=width, height=height, scale=(0.5, 2.0)),
    A.CoarseDropout(max_height=int(height / 5), max_width=int(width / 5)),
    A.RandomBrightnessContrast(),
    ToTensorV2(),
]), class_num=n_classes, ignore_indices=ignore_indices)

test_transforms = AlbumentationsSegmentationWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]), class_num=n_classes, ignore_indices=ignore_indices)

# dataset/dataloader
if test_images_dir == "":
    test_ratio = float(os.environ.get("TEST_RATIO"))
    root_dataset = IndexImageDataset.create(train_images_dir, train_annotations_dir,
                                            transforms=None)
    train_dataset, test_dataset = DatasetSplitter().split_train_test(test_ratio,
                                                                     root_dataset,
                                                                     train_transforms=train_transforms,
                                                                     test_transforms=test_transforms)
else:
    train_dataset = IndexImageDataset.create(image_dir_path=train_images_dir,
                                             index_image_dir_path=train_annotations_dir,
                                             transforms=train_transforms)
    test_dataset = IndexImageDataset.create(image_dir_path=test_images_dir, index_image_dir_path=test_annotations_dir,
                                            transforms=test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# TODO Model detail params
voc_focal_loss = SegmentationFocalLoss()
model: SegmentationModel = try_cuda(
    ShelfNet(n_classes=n_classes, out_size=(height, width), loss_func=voc_focal_loss))
if load_weight_path and load_weight_path != "":
    model.load_weight(load_weight_path)

# TODO Train detail params
# Metrics/Callbacks
callbacks = [ModelCheckout(per_epoch=int(epoch / 2), model=model, our_dir=saved_weights_dir),
             GenerateSegmentationImageCallback(output_dir=progress_dir, per_epoch=5, model=model,
                                               dataset=test_dataset)]
metric_ls = [SegmentationIoUByClasses(label_names), SegmentationRecallPrecision(label_names)]
metric_for_graph = SegmentationIoUByClasses(label_names, val_key=DetailMetricKey.KEY_AVERAGE)
learning_curve_visualizer = LearningCurveVisualizer(metric_name="mIoU", ignore_epoch=0,
                                                    save_filepath="segmentation_learning_curve.png")

# Training.
Trainer(model, learning_curve_visualizer=learning_curve_visualizer).fit(train_data_loader=train_dataloader,
                                                                        test_data_loader=test_dataloader,
                                                                        epochs=epoch, callbacks=callbacks,
                                                                        epoch_lr_scheduler_func=lr_scheduler,
                                                                        metric_for_graph=metric_for_graph,
                                                                        metric_ls=metric_ls,
                                                                        calc_metrics_per_epoch=5)
