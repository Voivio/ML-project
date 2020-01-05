import numpy as np
import cv2
import dlib
from typing import Optional


class OpenCVDetector:
    def __init__(self, detector_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(detector_path + "shape_predictor_68_face_landmarks.dat")

    def detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        :param img: np.ndarray, shape is (h, w, 3), BGR image
        :return: (68, 2) points, or None
        """
        #         raise NotImplementedError
        faces_in_image = self.detector(img, 0)
        try:
            face = faces_in_image[0]
        except IndexError:
            return None

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = self.predictor(img_gray, face)

        landmarks_list = []
        for i in range(0, landmarks.num_parts):
            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

        return np.array(landmarks_list, dtype=np.float)


def get_detector(detector_path='./'):
    #     raise NotImplementedError
    return OpenCVDetector(detector_path)
