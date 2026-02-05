import cv2
from insightface.app import FaceAnalysis
import glob

print("SCRIPT LANCE - TEST MULTIPLE IMAGES")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

image_paths = glob.glob("data/test/*.jpg")  # toutes les images JPG

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Image introuvable : {path}")
        continue

    faces = app.get(img)
    print(f"{path} - Nombre de visages détectés : {len(faces)}")

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face Detection Demo", img)
    cv2.waitKey(1000)  # 1 seconde par image

cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)  # webcam
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

print("LIVE DEMO - Appuyez sur 'q' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face Detection Live Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Nombre de visages détectés : {len(faces)}")
