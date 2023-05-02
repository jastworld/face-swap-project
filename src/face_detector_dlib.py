import detector
import dlib

class DlibDetector(detector.Detector):
    def __init__(self):
        super(DlibDetector, self).__init__() 

    def load (self):
        self.detector = dlib.get_frontal_face_detector()

    # detects face from input image and returns a detected region in a rectangle format [left, top, right, bottom]
    def predict(self, image):
        faces_rect = self.detector(image)

        return faces_rect[0]
    
