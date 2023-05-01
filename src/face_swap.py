import cv2
import numpy as np
import dlib
import sys

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]


def get_landmark_pts(img, detector, predictor):
  face = detector(img)[0]
  landmarks = predictor(img, face)
  landmarks_pts = np.zeros((68,2),dtype=np.int32)
  for i in range(68):
    landmarks_pts[i,0] = landmarks.part(i).x
    landmarks_pts[i,1] = landmarks.part(i).y
  return face, landmarks_pts

def get_center(landmark_pts):
  return np.mean(landmark_pts,axis=0).astype(int)

def extract_eye(landmarks, eye_indices):
  points = np.zeros((len(eye_indices),2))
  j = 0
  for i in eye_indices:
    points[j,:] = landmarks[i,:]
    j += 1
  return points

def get_angle(p1, p2):
  distance = p2 - p1
  angle = np.rad2deg(np.arctan2(distance[1], distance[0]))
  return angle

def get_avg_dist(center, landmarks):
  sum = 0
  for i in range(68):
    du = landmarks[i,0] - center[0]
    dv = landmarks[i,1] - center[1]
    dist = np.sqrt(du**2 + dv**2)
    sum += dist
  return dist/68

def get_scaling_factor(src_landmarks, tgt_landmarks):
  src_center_face = get_center(src_landmarks)
  tgt_center_face = get_center(tgt_landmarks)

  src_avg_dist = get_avg_dist(src_center_face, src_landmarks)
  tgt_avg_dist = get_avg_dist(tgt_center_face, tgt_landmarks)

  scale_factor = tgt_avg_dist/src_avg_dist
  return scale_factor

def get_rotation_matrix(src_landmarks,tgt_landmarks):
  src_left_eye = get_center(extract_eye(src_landmarks, LEFT_EYE_INDICES))
  src_right_eye = get_center(extract_eye(src_landmarks, RIGHT_EYE_INDICES))
  src_angle = get_angle(src_left_eye, src_right_eye) # angle to horiz

  tgt_left_eye = get_center(extract_eye(tgt_landmarks, LEFT_EYE_INDICES))
  tgt_right_eye = get_center(extract_eye(tgt_landmarks, RIGHT_EYE_INDICES))
  tgt_angle = get_angle(tgt_left_eye, tgt_right_eye) # angle to horiz

  angle_to_tgt = src_angle - tgt_angle
  src_center_face = get_center(src_landmarks)

  M = cv2.getRotationMatrix2D(((int(src_center_face[0]), int(src_center_face[1]))), angle_to_tgt, 1)
  return M

# Align src face with target
def align_src(src, src_landmarks, tgt_landmarks):
  scaling_factor = get_scaling_factor(src_landmarks,tgt_landmarks)
  src_scaled = cv2.resize(src, None, fx=scaling_factor, fy=scaling_factor)

  M = get_rotation_matrix(src_landmarks,tgt_landmarks)
  src_scaled_rotated = cv2.warpAffine(src_scaled, M, (src_scaled.shape[1], src_scaled.shape[0]))
  return src_scaled_rotated

def getTriangles(img, detector, predictor):
  face = detector(img)[0]
  landmarks = predictor(img, face)
  landmarks_points = []
  for n in range(0, 68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    landmarks_points.append((x, y))

  # Delaunay triangulation
  points = np.array(landmarks_points, np.int32)
  convexhull = cv2.convexHull(points)
  rect = cv2.boundingRect(convexhull)
  subdiv = cv2.Subdiv2D(rect)
  subdiv.insert(landmarks_points)
  triangles = subdiv.getTriangleList()
  triangles = np.array(triangles, dtype=np.int32)

  triangle_indexes = []
  for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    index_pt1 = np.where((points == pt1).all(axis=1))[0][0]
    index_pt2 = np.where((points == pt2).all(axis=1))[0][0]
    index_pt3 = np.where((points == pt3).all(axis=1))[0][0]

    triangle = [index_pt1, index_pt2, index_pt3]
    triangle_indexes.append(triangle)
  return points, triangle_indexes


def swap_faces(src, tgt, detector, predictor):
  tgt_gray = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
  src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

  _, tgt_landmarks = get_landmark_pts(tgt_gray, detector, predictor)
  _, src_landmarks = get_landmark_pts(src_gray, detector, predictor)

  new_src = align_src(src, src_landmarks, tgt_landmarks)
  new_src_gray = cv2.cvtColor(new_src, cv2.COLOR_BGR2GRAY)
  new_src_face, new_src_landmarks = get_landmark_pts(new_src_gray, detector, predictor)


  # TODO: CLEAN THIS UP
  new_face = np.zeros(tgt.shape, np.uint8)

  # Delaunay triangulation
  new_src_landmark_pts, triangle_indexes = getTriangles(new_src_gray, detector, predictor)

  # Face TGT
  faces2 = detector(tgt_gray)
  for face in faces2:
      landmarks = predictor(tgt_gray, face)
      landmarks_points2 = []
      for n in range(0, 68):
          x = landmarks.part(n).x
          y = landmarks.part(n).y
          landmarks_points2.append((x, y))


      points2 = np.array(landmarks_points2, np.int32)
      convexhull2 = cv2.convexHull(points2)

  lines_space_mask = np.zeros_like(new_src_gray)
  new_face = np.zeros(tgt.shape, np.uint8)
  # Triangulation of both faces
  for triangle_index in triangle_indexes:
      # Triangulation of the first face
      tr1_pt1 = tuple(new_src_landmark_pts[triangle_index[0]])
      tr1_pt2 = tuple(new_src_landmark_pts[triangle_index[1]])
      tr1_pt3 = tuple(new_src_landmark_pts[triangle_index[2]])
      triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


      rect1 = cv2.boundingRect(triangle1)
      (x, y, w, h) = rect1
      cropped_triangle = new_src[y: y + h, x: x + w]
      cropped_tr1_mask = np.zeros((h, w), np.uint8)


      points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

      cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

      # Lines space
      cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
      cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
      cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
      lines_space = cv2.bitwise_and(new_src, new_src, mask=lines_space_mask)

      # Triangulation of second face
      tr2_pt1 = landmarks_points2[triangle_index[0]]
      tr2_pt2 = landmarks_points2[triangle_index[1]]
      tr2_pt3 = landmarks_points2[triangle_index[2]]
      triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


      rect2 = cv2.boundingRect(triangle2)
      (x, y, w, h) = rect2

      cropped_tr2_mask = np.zeros((h, w), np.uint8)

      points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                          [tr2_pt2[0] - x, tr2_pt2[1] - y],
                          [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

      cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

      # Warp triangles
      points = np.float32(points)
      points2 = np.float32(points2)
      M = cv2.getAffineTransform(points, points2)
      warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
      warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

      # Reconstructing destination face
      img2_new_face_rect_area = new_face[y: y + h, x: x + w]
      img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
      _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
      warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

      img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
      new_face[y: y + h, x: x + w] = img2_new_face_rect_area



  # Face swapped (putting 1st face into 2nd face)
  img2_face_mask = np.zeros_like(tgt_gray)
  img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
  img2_face_mask = cv2.bitwise_not(img2_head_mask)

  tgt_no_face = cv2.bitwise_and(tgt, tgt, mask=img2_face_mask)

  result = cv2.add(tgt_no_face, new_face)

  (x, y, w, h) = cv2.boundingRect(convexhull2)
  center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

  seamlessclone = cv2.seamlessClone(result, tgt, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
  return seamlessclone, tgt, tgt_no_face, new_face

def main():
  src_file = '../image/bella-ramsey.jpg'
  tgt_file = '../image/natalie-portman.jpg'

  src = cv2.imread(src_file)
  tgt = cv2.imread(tgt_file)

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")
  seamlessclone, tgt, tgt_no_face, new_face = swap_faces(src, tgt, detector, predictor)
  cv2.imshow("Swap", seamlessclone)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  

   

if __name__ == '__main__':
  main()

