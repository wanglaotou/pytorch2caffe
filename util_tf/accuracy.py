

__all__ = ['landmarkAccuracy']

class landmarkAccuracy():
    def __init__(self, leftEyeIndex, rightEyeIndex):
        
        self.leftEyeIndex = leftEyeIndex
        self.rightEyeIndex = rightEyeIndex

    def __call__(self, out, target):
        if isinstance(out, (list, tuple)):
            leftEyePosX = out[self.leftEyeIndex*2]
            rightEyePosX = out[self.rightEyeIndex*2]

            for i in range():