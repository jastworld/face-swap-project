import detector
import dlib
import numpy as np
import cv2

class LandmarkDetector(detector.Detector):
    def __init__(self):
        super(LandmarkDetector, self).__init__() 

    def load (self):
        self.detector = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

    # detects landmarks from face 
    def predict(self, face_img, original_img):
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        landmarks = self.detector(img_gray, face_img)
        landmarks_points = []
        
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

        points = np.array(landmarks_points, np.int32)
        return points
    
