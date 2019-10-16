import cv2
import numpy as np

__all__ = ['inputResize']

class inputResize():
    def __init__(self, inputSize=None):
        self.inputSize = inputSize

    def __call__(self, data):
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return cv2.resize(data, (self.inputSize[1], self.inputSize[0]))