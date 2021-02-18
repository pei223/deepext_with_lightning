from pathlib import Path
import cv2
import tqdm
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dotenv import load_dotenv

from deepext_with_lightning.models import model_service
from deepext_with_lightning.transforms import AlbumentationsClsWrapperTransform
from deepext_with_lightning.models.base import SegmentationModel
from deepext_with_lightning.image_process.convert import try_cuda, normalize255, tensor_to_cv, to_4dim
from deepext_with_lightning.dataset import ImageOnlyDataset

load_dotenv("envs/segmentation.env")

model_name = os.environ.get("MODEL_NAME")
images_dir_path = os.environ.get("IMAGES_DIR_PATH")
result_dir_path = os.environ.get("RESULT_DIR_PATH")
checkpoint_path = os.environ.get("CHECKPOINT_PATH")
width, height = int(os.environ.get("IMAGE_WIDTH")), int(os.environ.get("IMAGE_HEIGHT"))
n_classes = int(os.environ.get("N_CLASSES"))

if not Path(result_dir_path).exists():
    Path(result_dir_path).mkdir()

transforms = AlbumentationsClsWrapperTransform(A.Compose([
    A.Resize(width=width, height=height),
    ToTensorV2(),
]))

dataset = ImageOnlyDataset(image_dir=images_dir_path, image_transform=transforms)

# TODO Choose model, parameters.
print("Loading model...")
model_class = model_service.resolve_segmentation_model(model_name)
model: SegmentationModel = try_cuda(model_class.load_from_checkpoint(checkpoint_path))
print("Model loaded")

for i, img_tensor in enumerate(tqdm.tqdm(dataset)):
    origin_image = normalize255(tensor_to_cv(img_tensor))
    pred_label, prob = model.predict_index_image(to_4dim(img_tensor))
    index_image = tensor_to_cv(pred_label[0])
    result_img = model.generate_mixed_segment_image(origin_image, index_image)
    result_img = cv2.resize(result_img, dataset.current_image_size())
    cv2.imwrite(f"{result_dir_path}/result_{i}.jpg", result_img)
