import argparse
import cv2
from landmark_detector import *
from face_swapper import *
from poisson_blending import *
from simple_merging import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=int, required=True, help='detector mode 0-dlib 1-cascade')
    parser.add_argument('--image1', type=str, required=True, help='path to first image[required]')
    parser.add_argument('--image2', type=str, required=True, help='path to second image[required]')
    parser.add_argument('--output', type=str, required=True, help='path to output image[required]')
    args = parser.parse_args()

    return args

def get_detector_backend(detector):
    if detector == 0:
        from face_detector_dlib import DlibDetector
        detector = DlibDetector()
    elif detector == 1:
        from face_detector_cascade import CascadeDetector
        detector = CascadeDetector()
    else:
        raise ValueError("unknown detector value: " + detector)
    return detector


def main():
    args = get_args()
    detectorMode = args.detector
    image1 = args.image1
    image2 = args.image2
    output = args.output

    detectorBackend = get_detector_backend(detectorMode)
    detectorBackend.load()
    predictorBackend = LandmarkDetector()
    predictorBackend.load()

    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    
    img1 = cv2.imread(image1)
    if img1 is None:
        raise ValueError("Cannot open image 1 :", image1)
    img2 = cv2.imread(image2)
    if img2 is None:
        raise ValueError("Cannot open image 2 :", image2)

    detected_face1 = detectorBackend.predict(img1)
    detected_face2 = detectorBackend.predict(img2)

    # cv2.rectangle(img1, (detected_face1[0], detected_face1[1]), (detected_face1[2], detected_face1[3]), (0, 255, 0), 2)
    # cv2.rectangle(img2, (detected_face2[0], detected_face2[1]), (detected_face2[2], detected_face2[3]), (0, 255, 0), 2)

    # cv2.imshow("Detected face", img1)
    # cv2.waitKey(0)
    # cv2.imshow("Detected face", img2)
    # cv2.waitKey(0)

    landmarks1 = predictorBackend.predict(detected_face1, img1)
    landmarks2 = predictorBackend.predict(detected_face2, img2)

    face_swapper = FaceSwapper(detectorBackend, predictorBackend)
    tgt, new_face = face_swapper.swap(img1, img2)

    poisson_blend = PoissonBlend()
    poisson_blend_image = poisson_blend.blend(new_face, tgt)
    simple_merge = SimpleMerge()
    simple_merge_image = simple_merge.merge(new_face, tgt)

    im_blend_out = np.uint8(poisson_blend_image * 255)
    im_out = cv2.cvtColor(im_blend_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite('Poisson_blend.jpg', im_out)

    im_merge_out = np.uint8(simple_merge_image * 255)
    im_out2 = cv2.cvtColor(im_merge_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite('Simple_Merge.jpg', im_out2)


if __name__ == "__main__":
    main()