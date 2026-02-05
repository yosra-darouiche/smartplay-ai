import cv2
import mediapipe as mp
import numpy as np

TARGET_SIZE = (160, 160)

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

# Bounding box à partir des landmarks
xs = [int(lm.x * w) for lm in landmarks]
ys = [int(lm.y * h) for lm in landmarks]

x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)

face_crop = image[y_min:y_max, x_min:x_max]

# Resize
face_resized = cv2.resize(face_crop, TARGET_SIZE)

# Normalisation (histogram equalization sur Y)
ycrcb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(ycrcb)
y_eq = cv2.equalizeHist(y)
face_normalized = cv2.merge((y_eq, cr, cb))
face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_YCrCb2BGR)

print("Préprocessing terminé")

cv2.imshow("Face crop", face_crop)
cv2.imshow("Preprocessed face", face_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
