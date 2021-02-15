import warnings

warnings.simplefilter('ignore')

from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.models.base import SegmentationModel
from deepext_with_lightning.models.layers.backbone_key import BackBoneKey
from deepext_with_lightning.models.segmentation import ShelfNet, UNet, ResUNet
from test.utils import gen_random_tensor, assert_tensor_shape

n_classes = 11
image_size = (256, 256)


def test_shelfnet_with_efficientnet():
    model = ShelfNet(n_classes, out_size=image_size, backbone_pretrained=False,
                     backbone=BackBoneKey.EFFICIENTNET_B0)
    _test_segmentation(model, 3, n_classes, image_size)
    _test_segmentation(model, 1, n_classes, image_size)


def test_shelfnet():
    model = ShelfNet(n_classes, out_size=image_size, backbone_pretrained=False,
                     backbone=BackBoneKey.RESNET_18)
    _test_segmentation(model, 3, n_classes, image_size)
    _test_segmentation(model, 1, n_classes, image_size)


def test_unet():
    model = UNet(n_classes)
    _test_segmentation(model, 3, n_classes, image_size)
    _test_segmentation(model, 1, n_classes, image_size)


def test_resunet():
    model = ResUNet(n_classes)
    _test_segmentation(model, 3, n_classes, image_size)
    _test_segmentation(model, 1, n_classes, image_size)


def _test_segmentation(model: SegmentationModel, batch_size: int, n_classes=11, image_size=(256, 256)):
    model = try_cuda(model)
    expected_prob_tensor_shape = (batch_size, n_classes) + image_size
    expected_label_tensor_shape = (batch_size,) + image_size

    test_tensor = try_cuda(gen_random_tensor(batch_size, image_size))

    labels, probs = model.predict_index_image(test_tensor)
    assert_tensor_shape(labels, expected_label_tensor_shape, "output label shape")
    assert_tensor_shape(probs, expected_prob_tensor_shape, "output prob shape")
