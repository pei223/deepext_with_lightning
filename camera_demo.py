import argparse

from pytorch_lightning import LightningModule

from deepext_with_lightning.dataset.functions import create_label_list_and_dict
from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.models import model_service
from deepext_with_lightning.models.base import ClassificationModel, AttentionClassificationModel, SegmentationModel, \
    DetectionModel
from deepext_with_lightning.camera import RealtimeClassification, RealtimeAttentionClassification, RealtimeSegmentation, \
    RealtimeDetection

parser = argparse.ArgumentParser(description='Pytorch Image detection training.')

parser.add_argument('--model', type=str, required=True, help=f"Model type in {model_service.valid_model_names()}")
parser.add_argument('--load_checkpoint_path', type=str, help="Saved weight path", required=True)
parser.add_argument('--image_size', type=int, default=256, help="Image size(default is 256)")
parser.add_argument('--label_names_path', type=str, default="voc_label_names.txt",
                    help="File path of label names (Classification and Detection only)")

if __name__ == "__main__":
    args = parser.parse_args()

    # Fetch model and load weight.
    model_class = model_service.resolve_model_class_from_name(args.model)

    model: LightningModule = try_cuda(model_class.load_from_checkpoint(args.load_checkpoint_path))

    if isinstance(model, SegmentationModel):
        RealtimeSegmentation(model=model, img_size_for_model=(args.image_size, args.image_size)).realtime_predict(
            video_output_path="output.mp4")
    elif isinstance(model, DetectionModel):
        label_names, label_dict = create_label_list_and_dict(args.label_names_path)
        RealtimeDetection(model=model, img_size_for_model=(args.image_size, args.image_size),
                          label_names=label_names).realtime_predict(video_output_path="output.mp4")
    if isinstance(model, AttentionClassificationModel):
        label_names, label_dict = create_label_list_and_dict(args.label_names_path)
        RealtimeAttentionClassification(model=model, img_size_for_model=(args.image_size, args.image_size),
                                        label_names=label_names).realtime_predict(video_output_path="output.mp4")
    elif isinstance(model, ClassificationModel):
        label_names, label_dict = create_label_list_and_dict(args.label_names_path)
        RealtimeClassification(model=model, img_size_for_model=(args.image_size, args.image_size),
                               label_names=label_names).realtime_predict(video_output_path="output.mp4")
