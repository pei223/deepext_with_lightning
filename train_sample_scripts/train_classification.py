import os
from dotenv import load_dotenv

from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext_with_lightning.dataset import CSVAnnotationDataset, DatasetSplitter
from deepext_with_lightning.models.layers.backbone_key import BackBoneKey
from deepext_with_lightning.models.base import ClassificationModel, AttentionClassificationModel
from deepext_with_lightning.models.classification import EfficientNet, AttentionBranchNetwork, \
    CustomClassificationNetwork, MobileNetV3
from deepext_with_lightning.callbacks import GenerateAttentionMap, CSVClassificationResult
from deepext_with_lightning.transforms import AlbumentationsClsWrapperTransform
from deepext_with_lightning.dataset.functions import create_label_list_and_dict
from deepext_with_lightning.image_process.convert import try_cuda

load_dotenv("envs/classification.env")

# File/Directory path
train_images_dir_path = os.environ.get("TRAIN_IMAGES_DIR_PATH")
train_annotation_file_path = os.environ.get("TRAIN_ANNOTATION_FILE_PATH")
test_images_dir_path = os.environ.get("TEST_IMAGES_DIR_PATH")
test_annotation_file_path = os.environ.get("TEST_ANNOTATION_FILE_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")

load_checkpoint_path = os.environ.get("LOAD_CHECKPOINT_PATH")
save_checkpoint_dir_path = os.environ.get("SAVE_CHECKPOINT_DIR_PATH")
progress_dir = os.environ.get("PROGRESS_DIR_PATH")
# Model params
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
# Learning params
batch_size = int(os.environ.get("BATCH_SIZE"))
lr = float(os.environ.get("LR"))
epoch = int(os.environ.get("EPOCH"))

label_names, label_dict = create_label_list_and_dict(label_file_path)
n_classes = len(label_names)

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

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# TODO Model detail params
model: ClassificationModel = try_cuda(EfficientNet(num_classes=n_classes, network='efficientnet-b0', lr=lr))
# model: ClassificationModel = try_cuda(AttentionBranchNetwork(n_classes=n_classes, backbone=BackBoneKey.RESNET_18))

if load_checkpoint_path and load_checkpoint_path != "":
    model.load_from_checkpoint(load_checkpoint_path)

# TODO Train detail params
# Metrics/Callbacks
val_every_n_epoch = 5
callbacks = [ModelCheckpoint(period=val_every_n_epoch, filename=f"{model.generate_model_name()}",
                             dirpath=save_checkpoint_dir_path, monitor='val_acc', verbose=True, mode="max"),
             CSVClassificationResult(period=epoch, model=model, dataset=test_dataset,
                                     label_names=label_names, out_filepath=f"{progress_dir}/result.csv"), ]
if isinstance(model, AttentionClassificationModel):
    callbacks.append(GenerateAttentionMap(model=model, output_dir=progress_dir, period=2,
                                          dataset=test_dataset, label_names=label_names))
logger = MLFlowLogger(experiment_name=f"classification_{model.generate_model_name()}")

# Training.
Trainer(max_epochs=epoch, callbacks=callbacks, gpus=-1,
        check_val_every_n_epoch=val_every_n_epoch, logger=logger) \
    .fit(model, train_dataloader=train_data_loader, val_dataloaders=test_data_loader)
