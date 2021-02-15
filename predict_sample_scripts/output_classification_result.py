from pathlib import Path
import cv2
import tqdm
from dotenv import load_dotenv
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext.layers.backbone_key import BackBoneKey
from deepext.models.base import ClassificationModel
from deepext.models.classification import AttentionBranchNetwork, MobileNetV3
from deepext.utils import try_cuda
from deepext.data.dataset import ImageOnlyDataset
from deepext.data.transforms import AlbumentationsClsWrapperTransform
from deepext.utils.dataset_util import create_label_list_and_dict

load_dotenv("envs/classification.env")

images_dir_path = os.environ.get("IMAGES_DIR_PATH")
result_file_path = os.environ.get("RESULT_FILE_PATH")
weight_path = os.environ.get("MODEL_WEIGHT_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
n_classes = int(os.environ.get("N_CLASSES"))

label_names, label_dict = create_label_list_and_dict(label_file_path)

transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=images_dir_path, image_transform=transforms)

# TODO Choose model, parameters.
print("Loading model...")
model: ClassificationModel = try_cuda(AttentionBranchNetwork(n_classes=n_classes, backbone=BackBoneKey.RESNET_34))
# model: ClassificationModel = try_cuda(MobileNetV3(num_classes=n_classes))
model.load_weight(weight_path)
print("Model loaded")

# model.save_model_for_mobile(width=96, height=96, out_filepath="abn_model_android.pt", for_os="cpu")
print("Saved")
with open(result_file_path, "w") as file:
    file.write(f"filepath,result label\n")
    for i, image in enumerate(tqdm.tqdm(dataset)):
        label = model.predict_label(image)
        file.write(f"{dataset.current_file_path()},{label_names[label]}\n")
