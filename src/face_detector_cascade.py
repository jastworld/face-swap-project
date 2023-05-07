import detector
import cv2, dlib

class CascadeDetector(detector.Detector):
    def __init__(self):
        super(CascadeDetector, self).__init__() 

    def load (self):
        self.detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    # detects face from input image and returns a detected region in a rectangle format [left, top, right, bottom]
    def predict(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(grayscale, 1.1, 9)
        rect = faces[0]
        faces_rect = dlib.rectangle(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])

        return faces_rect
    
