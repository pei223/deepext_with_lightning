from typing import Iterable, Tuple

import cv2
import numpy as np

INDEXED_COLOR_PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
                         64, 0, 0,
                         192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0,
                         64, 0, 128, 64,
                         0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0,
                         192, 64, 0, 64,
                         192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0,
                         64, 0, 128,
                         64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64,
                         64, 128, 64,
                         192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0,
                         192, 64, 128,
                         192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64,
                         192, 64, 192,
                         192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128,
                         0, 160, 128,
                         0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224,
                         128, 0, 96,
                         0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192,
                         0, 32, 64,
                         128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0,
                         96, 64, 128,
                         224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64,
                         32, 0, 192,
                         160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96,
                         0, 192, 224,
                         0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32,
                         64, 192, 160,
                         64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96,
                         64, 192, 224,
                         64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128,
                         128, 32,
                         128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128,
                         192, 32, 128,
                         64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96,
                         128, 0, 224,
                         128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128,
                         64, 224, 128,
                         192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0,
                         160, 192, 128,
                         160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64,
                         160, 192, 192,
                         160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224,
                         192, 128, 224,
                         192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224,
                         192, 192, 224,
                         192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128,
                         160, 160, 128,
                         96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224,
                         160, 128, 32,
                         96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224,
                         128, 96, 96,
                         0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128,
                         32, 32, 64,
                         160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192,
                         96, 32, 64,
                         224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192,
                         32, 96, 64,
                         160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192,
                         96, 96, 64,
                         224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]
INDEXED_COLOR_PALETTE = np.array(INDEXED_COLOR_PALETTE).reshape((-1, 3))


def draw_bounding_box_with_name_tag(origin_image: np.ndarray, bounding_box: Iterable[float],
                                    text: str = None,
                                    is_bbox_norm=False, thickness=1, color=(0, 0, 255), text_color=(0, 0, 0)):
    image = origin_image.copy()
    height, width = image.shape[:2]
    if len(bounding_box) < 4:
        return image
    x_min, y_min, x_max, y_max = bounding_box[:4]
    if is_bbox_norm:
        x_min *= width
        x_max *= width
        y_min *= height
        y_max *= height
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    if text:
        image = draw_text_with_background(image, text=text, offsets=(x_min, y_min), background_color=color,
                                          text_color=text_color)
    return image


def draw_text_with_background(origin_img: np.ndarray, text, offsets: Tuple[int, int], background_color=(255, 255, 255),
                              text_color=(0, 0, 0), margin_px=8, font_scale=0.7, font=cv2.FONT_HERSHEY_DUPLEX):
    img = origin_img.copy()
    text_width, text_height = cv2.getTextSize(text, font, font_scale, 1)[0]
    name_tag_height = text_height + margin_px * 2
    background_coors = (offsets,
                        (int(offsets[0] + text_width + margin_px * 2), int(offsets[1] + name_tag_height)))
    img = cv2.rectangle(img, background_coors[0], background_coors[1], background_color, cv2.FILLED)
    img = cv2.putText(img, text, (offsets[0] + margin_px, offsets[1] + margin_px + text_height), font, font_scale,
                      text_color, 1, cv2.LINE_AA)
    return img


def draw_text(origin_img: np.ndarray, text, offsets: Tuple[int, int],
              text_color=(0, 0, 0), font_scale=0.7, font=cv2.FONT_HERSHEY_DUPLEX):
    text_width, text_height = cv2.getTextSize(text, font, font_scale, 1)[0]
    return cv2.putText(origin_img, text, (offsets[0], offsets[1] + text_height), font, font_scale,
                       text_color, 1, cv2.LINE_AA)


def combine_heatmap(origin_image: np.ndarray, heatmap_img: np.ndarray, origin_alpha=0.5):
    assert origin_image.ndim == 3
    assert heatmap_img.ndim == 2
    heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    blended_img = cv2.addWeighted(origin_image, origin_alpha, heatmap_img, 1. - origin_alpha, 0)
    return blended_img


def indexed_image_to_rgb(indexed_img: np.ndarray) -> np.ndarray:
    """
    :param indexed_img: (height, width)
    :return: (height, width, 3
    """
    assert indexed_img.ndim == 2
    return cv2.cvtColor(INDEXED_COLOR_PALETTE[indexed_img].astype('uint8'), cv2.COLOR_BGR2RGB)
