import os
from dotenv import load_dotenv

from deepext_with_lightning.models import model_service

load_dotenv("envs/mobile_compile.env")
checkpoint_path = os.environ.get("CHECKPOINT_PATH")
model_name = os.environ.get("MODEL_NAME")
out_path = os.environ.get("OUT_PATH")

img_width = os.environ.get("IMAGE_WIDTH")
img_height = os.environ.get("IMAGE_HEIGHT")

model_class = model_service.resolve_model_class_from_name(model_name)
model = model_class.load_from_checkpoint(checkpoint_path)
model.save_model_for_mobile(img_width, img_height, out_filepath=out_path, for_os="android")
