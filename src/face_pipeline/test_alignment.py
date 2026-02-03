import cv2
import mediapipe as mp
import numpy as np

image_path = "data/test/face.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Image non trouvée")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

results = face_mesh.process(image_rgb)

if not results.multi_face_landmarks:
    print("Aucun visage")
    exit()

landmarks = results.multi_face_landmarks[0].landmark
h, w, _ = image.shape

# Points des yeux (MediaPipe)
left_eye = np.array([
    int(landmarks[33].x * w),
    int(landmarks[33].y * h)
])
right_eye = np.array([
    int(landmarks[263].x * w),
    int(landmarks[263].y * h)
])

# Calcul de l'angle
dx = right_eye[0] - left_eye[0]
dy = right_eye[1] - left_eye[1]
angle = np.degrees(np.arctan2(dy, dx))

# Centre entre les yeux
eyes_center = (
    int((left_eye[0] + right_eye[0]) / 2),
    int((left_eye[1] + right_eye[1]) / 2)
)

# Matrice de rotation
M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
aligned = cv2.warpAffine(image, M, (w, h))

print("Visage aligné")

cv2.imshow("Original", image)
cv2.imshow("Aligned", aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()
