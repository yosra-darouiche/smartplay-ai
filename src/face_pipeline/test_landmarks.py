import cv2
import mediapipe as mp

image_path = "data/test/face.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Image non trouvée")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            h, w, _ = image.shape
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    print("Landmarks détectés")
else:
    print("Aucun landmark")

cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
