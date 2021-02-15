from typing import Tuple, List

import torch
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader

from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.models.base import BaseDeepextModel

DETECTION_DATASET_INFO = {
    "voc2012": {
        "label_names": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "n_classes": 20,
    },
    "voc2007": {
        "label_names": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "n_classes": 20,
    },
}

CLASSIFICATION_DATASET_INFO = {
    "stl10": {
        "label_names": ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'],
        "n_classes": 10,
        "size": (96, 96),
    },
    "cifar10": {
        "label_names": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "n_classes": 10,
        "size": (32, 32),
    }
}

SEGMENTATION_DATASET_INFO = {
    "voc2012": {
        "label_names": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "n_classes": 20,
    },
    "voc2007": {
        "label_names": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "n_classes": 20,
    },
    "cityscape": {
        "label_names": ['ego vehicle', 'rectification', 'out of roi', 'static', 'dynamic',
                        'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building',
                        'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole',
                        'polegroup', 'traffic light', 'traffic sign', 'vegetation',
                        'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
                        'caravan', 'trailer', 'train', 'motorcycle', 'bicycle',
                        'license plate'],
        "n_classes": 34,
    }
}


def build_data_loader(args, train_dataset: Dataset, test_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4), \
           DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)


def get_logger(log_prefix: str, args, model: BaseDeepextModel):
    test_tensor = try_cuda(torch.randn(1, 3, args.image_size, args.image_size))
    if args.log_type == "mlflow":
        logger = MLFlowLogger(experiment_name=f"{log_prefix}_{args.dataset}_{model.generate_model_name()}")
        # Log the model
        # with mlflow.start_run():
        #     mlflow.pytorch.log_model(model, "model")
        #
        #     # convert to scripted model and log the model
        #     scripted_pytorch_model = torch.jit.script(model)
        #     mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")
        return logger
    elif args.log_type == "tensorboard":
        logger = TensorBoardLogger(save_dir="tensorboard_logs", version="v",
                                   name=f"segmentation_demo_{args.dataset}_{model.generate_model_name()}")
        logger.experiment.add_graph(model, test_tensor)
        return logger
    raise RuntimeError(f"Invalid log type: {args.log_type}")


def label_names_to_dict(label_names: List[str]):
    class_index_dict = {}
    for i, label_name in enumerate(label_names):
        class_index_dict[label_name] = i
    return class_index_dict
