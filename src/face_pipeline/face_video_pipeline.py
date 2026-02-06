import cv2
import os
import csv
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
VIDEO_PATH = "data/raw/sample1.mp4"  # chemin de ta vidéo
OUTPUT_DIR = "aligned_faces"              # dossier pour sauvegarder les visages
CSV_PATH = "faces_tracking.csv"           # fichier CSV
DET_SIZE = (640, 640)                     # taille pour la détection

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- INITIALISATION ----------------
app = FaceAnalysis()
app.prepare(ctx_id=-1, det_size=DET_SIZE)  # CPU
print("FaceAnalysis prêt !")

# Ouvrir la vidéo
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
face_id_global = 0

# Ouvrir CSV
with open(CSV_PATH, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "face_id", "x", "y", "width", "height"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces = app.get(frame)

        for face in faces:
            # Aligner si possible
            aligned_face = getattr(face, "normed_face", None)

            if aligned_face is None:
                # Si alignement impossible, recadrer le visage détecté
                x1, y1, x2, y2 = face.bbox.astype(int)
                aligned_face = frame[y1:y2, x1:x2]

            # Sauvegarder le visage
            face_filename = os.path.join(OUTPUT_DIR, f"frame{frame_count}_face{face_id_global}.jpg")
            cv2.imwrite(face_filename, aligned_face)

            # Écrire dans le CSV
            x1, y1, x2, y2 = face.bbox.astype(int)
            csv_writer.writerow([frame_count, face_id_global, x1, y1, x2-x1, y2-y1])

            face_id_global += 1

        print(f"Frame {frame_count} traitée, {len(faces)} visage(s) détecté(s)")

cap.release()
print("Traitement terminé ! Tous les visages sont sauvegardés et CSV généré.")
