# src/face_pipeline/recognize_user.py

import os
import numpy as np
import cv2
import sys

# 🔹 Ajouter le dossier racine au PYTHONPATH pour que 'embeddings' soit reconnu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from embeddings.extractor import FaceEmbeddingExtractor  # maintenant Python le trouve
from embeddings.create_faiss_index import load_faiss_index

# 🔹 Chemin vers la vidéo ou webcam
VIDEO_PATH = "data/test_video.mp4"  # ou 0 pour la webcam

# 🔹 Charger le modèle d'embeddings
extractor = FaceEmbeddingExtractor(model_name="buffalo_l")
faiss_index, user_labels = load_faiss_index("data/faiss_index.idx")

# 🔹 Ouvrir la vidéo / webcam
cap = cv2.VideoCapture(VIDEO_PATH if os.path.exists(VIDEO_PATH) else 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 Détection + extraction embeddings
    faces = extractor.detect_and_extract(frame)  # retourne une liste de dicts {'bbox':.., 'embedding':..}

    for face in faces:
        emb = face['embedding']
        # 🔹 Comparer à la base FAISS
        D, I = faiss_index.search(np.array([emb]), k=1)
        label = user_labels[I[0][0]] if D[0][0] < 0.6 else "Unknown"

        # 🔹 Dessiner sur la frame
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
