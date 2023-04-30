class Detector():
    def __init__(self):
        self.detector = None

    def load(self):
        raise NotImplementedError("Detector:load")

    def predict(self, image):
        raise NotImplementedError("Detector:predict")
 