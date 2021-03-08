import os
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext_with_lightning.dataset import VOCDataset, AdjustDetectionTensorCollator, DatasetSplitter
from deepext_with_lightning.models.base import DetectionModel
from deepext_with_lightning.models.object_detection import EfficientDetector
from deepext_with_lightning.callbacks import GenerateDetectionImageCallback
from deepext_with_lightning.transforms import AlbumentationsDetectionWrapperTransform
from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.dataset.functions import create_label_list_and_dict

load_dotenv("envs/detection.env")

# File/Directory path
train_images_dir = os.environ.get("TRAIN_IMAGES_PATH")
train_annotations_dir = os.environ.get("TRAIN_ANNOTATIONS_PATH")
test_images_dir = os.environ.get("TEST_IMAGES_PATH")
test_annotations_dir = os.environ.get("TEST_ANNOTATIONS_PATH")
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

label_names, class_index_dict = create_label_list_and_dict(label_file_path)
n_classes = len(label_names)

ignore_indices = [255, ]

# TODO Data augmentation
train_transforms = AlbumentationsDetectionWrapperTransform([
    A.HorizontalFlip(),
    A.RandomResizedCrop(width=width, height=height, scale=(0.8, 1.)),
    A.OneOf([
        A.Blur(blur_limit=5),
        A.RandomBrightnessContrast(),
        A.RandomGamma(), ]),
    ToTensorV2(),
])
test_transforms = AlbumentationsDetectionWrapperTransform([
    A.Resize(width=width, height=height),
    ToTensorV2(),
])

# dataset/dataloader
if test_images_dir == "":
    test_ratio = float(os.environ.get("TEST_RATIO"))
    root_dataset = VOCDataset.create(image_dir_path=train_images_dir, annotation_dir_path=train_annotations_dir,
                                     class_index_dict=class_index_dict, transforms=None)
    train_dataset, test_dataset = DatasetSplitter().split_train_test(test_ratio, root_dataset,
                                                                     train_transforms=train_transforms,
                                                                     test_transforms=test_transforms)
else:
    train_dataset = VOCDataset.create(image_dir_path=train_images_dir, annotation_dir_path=train_annotations_dir,
                                      transforms=train_transforms, class_index_dict=class_index_dict)
    test_dataset = VOCDataset.create(image_dir_path=test_images_dir, transforms=test_transforms,
                                     annotation_dir_path=test_annotations_dir, class_index_dict=class_index_dict)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=AdjustDetectionTensorCollator())
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=AdjustDetectionTensorCollator())

# TODO Model detail params
model: DetectionModel = try_cuda(EfficientDetector(n_classes=n_classes, lr=lr,
                                                   network=f"efficientdet-d0", score_threshold=0.5))

if load_checkpoint_path and load_checkpoint_path != "":
    model = model.load_from_checkpoint(load_checkpoint_path)

# TODO Train detail params
# Callbacks
val_every_n_epoch = 5
callbacks = [ModelCheckpoint(period=val_every_n_epoch, filename=f"{model.generate_model_name()}",
                             dirpath=save_checkpoint_dir_path, monitor='val_map', verbose=True, mode="max"),
             GenerateDetectionImageCallback(model, (height, width), test_dataset, per_epoch=5,
                                            out_dir=progress_dir, label_names=label_names)]

logger = MLFlowLogger(experiment_name=f"detection_{model.generate_model_name()}")

# Training.
Trainer(max_epochs=epoch, callbacks=callbacks, gpus=-1,
        check_val_every_n_epoch=val_every_n_epoch, logger=logger) \
    .fit(model, train_dataloader=train_data_loader, val_dataloaders=test_data_loader)
