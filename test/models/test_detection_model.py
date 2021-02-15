import warnings

warnings.simplefilter('ignore')

from deepext_with_lightning.image_process.convert import try_cuda
from deepext_with_lightning.models.object_detection import EfficientDetector
from test.utils import gen_random_tensor, assert_tensor_shape

n_classes = 10
image_size = (512, 512)


def test_efficientdet():
    model: EfficientDetector = try_cuda(EfficientDetector(n_classes=n_classes, score_threshold=1e-2))
    test_image = gen_random_tensor(1, image_size)
    result = model.predict_bboxes(test_image)

    assert result[0].ndim == 2 and result[0].shape[
        -1] == 6, "Detection model result must contain (xmin, ymin, xmax, ymax, class, score)"


def test_efficientdet_with_multi_batch():
    model: EfficientDetector = try_cuda(EfficientDetector(n_classes=n_classes, score_threshold=1e-2))
    test_image = gen_random_tensor(2, image_size)
    result = model.predict_bboxes(test_image)

    assert result[0].ndim == 2 and result[0].shape[
        -1] == 6, "Detection model result must contain (xmin, ymin, xmax, ymax, class, score)"
    assert result[1].ndim == 2 and result[1].shape[
        -1] == 6, "Detection model result must contain (xmin, ymin, xmax, ymax, class, score)"


def test_efficientdet_with_2scale():
    model: EfficientDetector = try_cuda(
        EfficientDetector(n_classes=n_classes, score_threshold=1e-2, network="efficientdet-d2"))
    test_image = gen_random_tensor(1, image_size)
    result = model.predict_bboxes(test_image)

    assert result[0].ndim == 2 and result[0].shape[
        -1] == 6, "Detection model result must contain (xmin, ymin, xmax, ymax, class, score)"
