from pathlib import Path
import cv2
import os, tqdm
from dotenv import load_dotenv
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from deepext_with_lightning.models.base import DetectionModel
from deepext_with_lightning.models import model_service
from deepext_with_lightning.dataset import ImageOnlyDataset
from deepext_with_lightning.transforms import AlbumentationsClsWrapperTransform
from deepext_with_lightning.image_process.convert import try_cuda, normalize255, tensor_to_cv, to_4dim
from deepext_with_lightning.dataset.functions import create_label_list_and_dict

load_dotenv("envs/detection.env")

model_name = os.environ.get("MODEL_NAME")
images_dir_path = os.environ.get("IMAGES_DIR_PATH")
result_dir_path = os.environ.get("RESULT_DIR_PATH")
checkpoint_path = os.environ.get("CHECKPOINT_PATH")
label_file_path = os.environ.get("LABEL_FILE_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))

if not Path(result_dir_path).exists():
    Path(result_dir_path).mkdir()

label_names, label_dict = create_label_list_and_dict(label_file_path)
n_classes = len(label_names)

transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=images_dir_path, image_transform=transforms)

print("Loading model...")
model_class = model_service.resolve_detection_model(model_name)
model: DetectionModel = try_cuda(model_class.load_from_checkpoint(checkpoint_path))
print("Model loaded")

for i, img_tensor in enumerate(tqdm.tqdm(dataset)):
    origin_image = normalize255(tensor_to_cv(img_tensor))
    result_bboxes = model.predict_bboxes(to_4dim(img_tensor))[0]
    result_img = model.generate_bbox_draw_image(origin_image, bboxes=result_bboxes,
                                                model_img_size=(width, height),
                                                label_names=label_names)
    result_img = cv2.resize(result_img, dataset.current_image_size())
    cv2.imwrite(f"{result_dir_path}/result_{i}.jpg", result_img)
