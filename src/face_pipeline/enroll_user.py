import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
NB_IMAGES = 10
SAVE_DIR = "data/enrolled_users"
# ----------------------------------------

def main():
    user_id = input("Entrer l'identifiant utilisateur : ").strip()
    user_path = os.path.join(SAVE_DIR, user_id)
    os.makedirs(user_path, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    count = 0

    print("📸 Capture en cours... Regardez la caméra")

    while count < NB_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        if len(faces) == 1:
            face = faces[0]
            emb = face.embedding

            np.save(
                os.path.join(user_path, f"emb_{count}.npy"),
                emb
            )

            count += 1
            print(f"✅ Image {count}/{NB_IMAGES} capturée")

            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Enrôlement terminé avec succès !")

if __name__ == "__main__":
    main()
