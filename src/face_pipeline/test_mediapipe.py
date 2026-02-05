import cv2
import mediapipe as mp

# Initialisation MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# Charger l'image
image = cv2.imread("data/test/face.jpg")
if image is None:
    raise FileNotFoundError("Image non trouvée : data/test/face.jpg")

# Convertir BGR -> RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Détection
results = face_detection.process(image_rgb)

# Dessiner les résultats
if results.detections:
    for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    print(f"{len(results.detections)} visage(s) détecté(s)")
else:
    print("Aucun visage détecté")

# Afficher l'image
cv2.imshow("Face Detection - MediaPipe", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
