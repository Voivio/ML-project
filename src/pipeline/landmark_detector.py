import numpy as np
from typing import Optional


class OpenCVDetector:
    def detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        :param img: np.ndarray, shape is (h, w, 3), BGR image
        :return: (68, 2) points, or None
        """
        raise NotImplementedError


def get_detector():
    raise NotImplementedError
