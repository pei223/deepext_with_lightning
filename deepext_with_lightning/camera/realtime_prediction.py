from abc import abstractmethod
from typing import Tuple

import cv2
import numpy as np
import time
from ..models.base import BaseDeepextModel
from ..image_process.drawer import draw_text_with_background


class RealtimePrediction:
    def __init__(self, model: BaseDeepextModel, img_size_for_model: Tuple[int, int]):
        """
        :param model:
        :param img_size_for_model: (width, height)
        """
        self.model = model
        self.is_running = False
        self.img_size_for_model = img_size_for_model

    def stop(self):
        self.is_running = False

    def realtime_predict(self, frame_size=(1080, 720), fps=5, device_id=0, video_output_path: str = None, verbose=True):
        capture = cv2.VideoCapture(device_id)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size) if video_output_path else None

        self.is_running = True
        while capture.isOpened() and self.is_running:
            ret, frame = capture.read()
            if not ret:
                print("Failed to read frame.")
                break

            start = time.time()
            result_img = self.calc_result(frame)
            infer_speed = time.time() - start
            result_img = self._arrange_image_for_video_writing(result_img, frame_size)
            result_img = self._write_inference_speed(result_img, infer_speed)
            if video_writer:
                video_writer.write(result_img)
            cv2.imshow('frame', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Keyboard q pushed.")
                break
        if video_writer:
            video_writer.release()
        capture.release()
        cv2.destroyAllWindows()

    def _write_inference_speed(self, frame: np.ndarray, infer_time: float):
        text = f"Inference speed:   {infer_time} s / frame"
        offsets = (0, 25)
        background_color = (255, 255, 255)
        text_color = (0, 0, 255)
        return draw_text_with_background(frame, background_color=background_color, text_color=text_color, text=text,
                                         offsets=offsets, font_scale=0.5)

    @abstractmethod
    def calc_result(self, frame: np.ndarray) -> np.ndarray:
        pass

    def _arrange_image_for_video_writing(self, img, img_size):
        img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[-1] == 4 else img
