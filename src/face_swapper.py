import cv2
import numpy as np
import dlib
import sys


class FaceSwapper:
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

    def __init__(self, detector, predictor):
        self.detector = detector
        self.predictor = predictor

    def __get_landmark_pts(self, img):
        detected_face = self.detector.predict(img)
        landmarks = self.predictor.predict(detected_face, img)
        return landmarks

    def __get_center(self, landmark_pts):
        return np.mean(landmark_pts, axis=0).astype(int)

    def __extract_eye(self, landmarks, eye_indices):
        points = np.zeros((len(eye_indices), 2))
        j = 0
        for i in eye_indices:
            points[j, :] = landmarks[i, :]
            j += 1
        return points

    def __get_angle(self, p1, p2):
        distance = p2 - p1
        angle = np.rad2deg(np.arctan2(distance[1], distance[0]))
        return angle

    def __get_avg_dist(self, center, landmarks):
        sum = 0
        for i in range(68):
            du = landmarks[i, 0] - center[0]
            dv = landmarks[i, 1] - center[1]
            dist = np.sqrt(du**2 + dv**2)
            sum += dist
        return dist / 68

    def __get_scaling_factor(self, src_landmarks, tgt_landmarks):
        src_center_face = self.__get_center(src_landmarks)
        tgt_center_face = self.__get_center(tgt_landmarks)

        src_avg_dist = self.__get_avg_dist(src_center_face, src_landmarks)
        tgt_avg_dist = self.__get_avg_dist(tgt_center_face, tgt_landmarks)

        scale_factor = tgt_avg_dist / src_avg_dist
        return scale_factor

    def __get_rotation_matrix(self, src_landmarks, tgt_landmarks):
        src_left_eye = self.__get_center(
            self.__extract_eye(src_landmarks, self.LEFT_EYE_INDICES)
        )
        src_right_eye = self.__get_center(
            self.__extract_eye(src_landmarks, self.RIGHT_EYE_INDICES)
        )
        src_angle = self.__get_angle(src_left_eye, src_right_eye)  # angle to horiz

        tgt_left_eye = self.__get_center(
            self.__extract_eye(tgt_landmarks, self.LEFT_EYE_INDICES)
        )
        tgt_right_eye = self.__get_center(
            self.__extract_eye(tgt_landmarks, self.RIGHT_EYE_INDICES)
        )
        tgt_angle = self.__get_angle(tgt_left_eye, tgt_right_eye)  # angle to horiz

        angle_to_tgt = src_angle - tgt_angle
        src_center_face = self.__get_center(src_landmarks)

        M = cv2.getRotationMatrix2D(
            ((int(src_center_face[0]), int(src_center_face[1]))), angle_to_tgt, 1
        )
        return M

    # Scale and rotate face to match target
    def __align_src(self, src, src_landmarks, tgt_landmarks):
        scaling_factor = self.__get_scaling_factor(src_landmarks, tgt_landmarks)
        src_scaled = cv2.resize(src, None, fx=scaling_factor, fy=scaling_factor)

        M = self.__get_rotation_matrix(src_landmarks, tgt_landmarks)
        src_scaled_rotated = cv2.warpAffine(
            src_scaled, M, (src_scaled.shape[1], src_scaled.shape[0])
        )
        return src_scaled_rotated

    # Get the triangles of the landmarks
    def _getTriangles(self, landmarks):
        convexhull = cv2.convexHull(landmarks)
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks.astype(np.float32).tolist())
        triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)
        triangle_indexes = []
        for t in triangles:
            pt1 = [t[0], t[1]]
            pt2 = [t[2], t[3]]
            pt3 = [t[4], t[5]]
            index_pt1 = np.where((landmarks == pt1).all(axis=1))[0][0]
            index_pt2 = np.where((landmarks == pt2).all(axis=1))[0][0]
            index_pt3 = np.where((landmarks == pt3).all(axis=1))[0][0]
            triangle = [index_pt1, index_pt2, index_pt3]
            triangle_indexes.append(triangle)
        return triangle_indexes

    def __get_new_face(self, src, tgt, src_landmarks, tgt_landmarks):
        src_triangle_indexes = self._getTriangles(src_landmarks)
        new_face = np.zeros(tgt.shape, np.uint8)

        for triangle in src_triangle_indexes:
            src_triangle = np.array(
                [
                    src_landmarks[triangle[0]],
                    src_landmarks[triangle[1]],
                    src_landmarks[triangle[2]],
                ],
                np.int32,
            )
            src_rect = cv2.boundingRect(src_triangle)
            (x, y, w, h) = src_rect
            cropped_triangle = src[y : y + h, x : x + w]
            cropped_src_mask = np.zeros((h, w), np.uint8)
            src_points = src_triangle - np.array([x, y])
            cv2.fillConvexPoly(cropped_src_mask, src_points, 255)

            tgt_triangle = np.array(
                [
                    tgt_landmarks[triangle[0]],
                    tgt_landmarks[triangle[1]],
                    tgt_landmarks[triangle[2]],
                ],
                np.int32,
            )
            tgt_rect = cv2.boundingRect(tgt_triangle)
            (x, y, w, h) = tgt_rect
            cropped_tgt_mask = np.zeros((h, w), np.uint8)
            tgt_points = tgt_triangle - np.array([x, y])
            cv2.fillConvexPoly(cropped_tgt_mask, tgt_points, 255)

            # Warp triangles of src to the same shapes as tgt
            M = cv2.getAffineTransform(np.float32(src_points), np.float32(tgt_points))
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=cropped_tgt_mask
            )

            # Put triangles back together
            src_new_face_rect_area = new_face[y : y + h, x : x + w]
            src_new_face_rect_area_gray = cv2.cvtColor(
                src_new_face_rect_area, cv2.COLOR_BGR2GRAY
            )
            _, mask_triangles_designed = cv2.threshold(
                src_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV
            )
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=mask_triangles_designed
            )

            src_new_face_rect_area = cv2.add(src_new_face_rect_area, warped_triangle)
            new_face[y : y + h, x : x + w] = src_new_face_rect_area

        return new_face

    def swap(self, src, tgt):
        tgt_landmarks = self.__get_landmark_pts(tgt)
        src_landmarks = self.__get_landmark_pts(src)

        src = self.__align_src(src, src_landmarks, tgt_landmarks)
        src_landmarks = self.__get_landmark_pts(src)

        new_face = self.__get_new_face(src, tgt, src_landmarks, tgt_landmarks)
        return tgt, new_face
