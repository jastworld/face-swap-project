import cv2
import numpy as np
import dlib
import sys


class FaceSwapper:
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    FACE_OUTLINE_IDICES = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 18, 24, 25, 16]

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

            src_new_face_section = new_face[y : y + h, x : x + w]
            src_new_face_section_gray = cv2.cvtColor(
                src_new_face_section, cv2.COLOR_BGR2GRAY
            )
            _, triangle_mask = cv2.threshold(
                src_new_face_section_gray, 1, 255, cv2.THRESH_BINARY_INV
            )
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=triangle_mask
            )

            src_new_face_section = cv2.add(src_new_face_section, warped_triangle)
            new_face[y : y + h, x : x + w] = src_new_face_section

        return new_face

    def __src_warp_swap(self, src, tgt):
        tgt_landmarks = self.__get_landmark_pts(tgt)
        src_landmarks = self.__get_landmark_pts(src)

        src = self.__align_src(src, src_landmarks, tgt_landmarks)
        src_landmarks = self.__get_landmark_pts(src)

        new_face = self.__get_new_face(src, tgt, src_landmarks, tgt_landmarks)
        return tgt, new_face
    
    def __get_face_cutout(self, landmarks, src, tgt, tgt_center):
      cutout_route = []
      for i in range(0, len(self.FACE_OUTLINE_IDICES) - 1):
        curr_index =  self.FACE_OUTLINE_IDICES[i]
        next_index = self.FACE_OUTLINE_IDICES[i+1]

        curr_pt = landmarks[curr_index]
        end_pt = landmarks[next_index]
        cutout_route.append(curr_pt)
        cv2.line(src, tuple(curr_pt), tuple(end_pt), (255,255,255),2)

      # append final point to the end so that loop completes
      cutout_route.append(cutout_route[0])

      mask = np.zeros(src.shape[:2])
      mask = cv2.fillConvexPoly(mask, np.array(cutout_route), 1)
      mask = mask.astype(bool)
      cutout = np.zeros_like(src) 
      cutout[mask] = src[mask]
      for i in range(0, len(self.FACE_OUTLINE_IDICES) - 1):
        curr_index =  self.FACE_OUTLINE_IDICES[i]
        next_index = self.FACE_OUTLINE_IDICES[i+1]

        curr_pt = landmarks[curr_index]
        end_pt = landmarks[next_index]
        cutout_route.append(curr_pt)
        cv2.line(cutout, tuple(curr_pt), tuple(end_pt), (0,0,0),2)
      src_center = self.__get_center(landmarks)
      tx = tgt_center[0] - src_center[0]
      ty = tgt_center[1] - src_center[1]
      M = np.float32([[1, 0, tx], [0, 1, ty]])
      out = cv2.warpAffine(cutout, M, (tgt.shape[1], tgt.shape[0]))
      return out
        
    def __target_warp_swap(self, src, tgt):
      tgt_landmarks = self.__get_landmark_pts(tgt)
      src_landmarks = self.__get_landmark_pts(src)

      src = self.__align_src(src, src_landmarks, tgt_landmarks)
      src_landmarks = self.__get_landmark_pts(src)

      src_center = self.__get_center(src_landmarks)
      tgt_center = self.__get_center(tgt_landmarks)

      important_points = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,33]

      triangle_indexes = [
          [0,1,2],[2,3,4],[4,5,6],[6,7,8],[0,2,4],[4,6,8],[0,4,8],[0,8,17],[0,18,20],[0,1,18],
          [1,2,18],[2,3,18],[3,4,18],[4,5,18],[5,6,18],[6,7,18],[7,8,18],[8,18,19],[8,9,10],[10,11,12],
          [12,13,14],[14,15,16],[8,10,12],[12,14,16],[8,12,16],[8,16,17],[8,9,19],[9,10,19],[10,11,19],[11,12,19],
          [12,13,19],[13,14,19],[14,15,19],[15,16,19],[16,19,21]
      ]

      tgt_points = []
      new_locations = []
      index = []
      for i, landmark_pt in enumerate(tgt_landmarks):
        if i in important_points:
          index.append(i)
          tgt_points.append((int(landmark_pt[0]),int(landmark_pt[1])))
          if i in [1,17,34]:  
            new_locations.append(landmark_pt)
          else:
            src_landmark = src_landmarks[i]
            new_point = tgt_center + (src_landmark - src_center)
            new_locations.append((int(new_point[0]),int(new_point[1])))

      tgt_h,tgt_w,_ = tgt.shape
      tgt_points.append((0,tgt_h-1))
      tgt_points.append((tgt_w-1,tgt_h-1))
      tgt_points.append((0,0))
      tgt_points.append((tgt_w-1,0))

      new_locations.append((0,tgt_h-1))
      new_locations.append((tgt_w-1,tgt_h-1))
      new_locations.append((0,0))
      new_locations.append((tgt_w-1,0))
      new_locations = np.array(new_locations, np.int32)

      tgt_points = np.array(tgt_points, np.float32).reshape(-1, 2)
      warped_tgt = np.zeros(tgt.shape, np.uint8)

      for triangle_index in triangle_indexes:
        tgt_pt1 = tuple(tgt_points[triangle_index[0]].astype(np.int32))
        tgt_pt2 = tuple(tgt_points[triangle_index[1]].astype(np.int32))
        tgt_pt3 = tuple(tgt_points[triangle_index[2]].astype(np.int32))
        tgt_triangle = np.array([tgt_pt1, tgt_pt2, tgt_pt3], np.int32)

        (x, y, w, h) = cv2.boundingRect(tgt_triangle)
        cropped_triangle = tgt[y: y + h, x: x + w]
        cropped_tgt_mask = np.zeros((h, w), np.uint8)


        tgt_shifted_points = np.array([[tgt_pt1[0] - x, tgt_pt1[1] - y],
                            [tgt_pt2[0] - x, tgt_pt2[1] - y],
                            [tgt_pt3[0] - x, tgt_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tgt_mask, tgt_shifted_points, 255)

        new_pt1 = new_locations[triangle_index[0]]
        new_pt2 = new_locations[triangle_index[1]]
        new_pt3 = new_locations[triangle_index[2]]
        new_location_triangle = np.array([new_pt1, new_pt2, new_pt3], np.int32)

        (x, y, w, h) = cv2.boundingRect(new_location_triangle)
        cropped_new_mask = np.zeros((h, w), np.uint8)

        new_shifted_points = np.array([[new_pt1[0] - x, new_pt1[1] - y],
                            [new_pt2[0] - x, new_pt2[1] - y],
                            [new_pt3[0] - x, new_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_new_mask, new_shifted_points, 255)

        # Warp triangles of tgt to new locations
        M = cv2.getAffineTransform(tgt_shifted_points.astype(np.float32), new_shifted_points.astype(np.float32))
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_new_mask)

        tgt_new_face_section = warped_tgt[y: y + h, x: x + w]

        tgt_new_face_section_gray = cv2.cvtColor(tgt_new_face_section, cv2.COLOR_BGR2GRAY)
        _, triangle_mask = cv2.threshold(tgt_new_face_section_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=triangle_mask)

        tgt_new_face_section = cv2.add(tgt_new_face_section, warped_triangle)
        warped_tgt[y: y + h, x: x + w] = tgt_new_face_section

      mask = cv2.inRange(warped_tgt, np.array([0,0,0]), np.array([0,0,0]))
      tgt_masked = cv2.bitwise_and(tgt, tgt, mask=mask)
      warped_tgt += tgt_masked
      warped_tgt = cv2.convertScaleAbs(warped_tgt)

      src_face = self.__get_face_cutout(src_landmarks, src, tgt, tgt_center)
      return warped_tgt, src_face

    def swap(self, src, tgt, mode):
        if mode == 1:
          return self.__src_warp_swap(src,tgt)
        else:
          return self.__target_warp_swap(src,tgt)