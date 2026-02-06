import cv2
import numpy as np
import faiss
import json
from embeddings.extractor import FaceEmbeddingExtractor

# Charger FAISS index
index = faiss.read_index("embeddings/faiss_index.index")

# Charger users metadata
with open("embeddings/users.json", "r") as f:
    users = json.load(f)

# Capture vidéo
cap = cv2.VideoCapture(0)
extractor = FaceEmbeddingExtractor()

while True:
    ret, frame = cap.read()
    cv2.imshow("Reconnaissance", frame)
    key = cv2.waitKey(1)
    
    if key == ord("r"):  # appuie 'r' pour reconnaître
        embedding = extractor.get_embedding(frame)
        D, I = index.search(np.array([embedding]), k=1)  # k=1 pour 1:1
        nearest = I[0][0]
        if D[0][0] < 0.8:  # seuil distance L2
            print(f"Joueur identifié : {users[nearest]['name']}")
        else:
            print("Inconnu. Veuillez enrôler.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
