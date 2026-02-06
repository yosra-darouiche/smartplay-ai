import os
import numpy as np
import faiss

EMB_DIR = "data/enrolled_users"
INDEX_PATH = "embeddings/faiss.index"
LABELS_PATH = "embeddings/labels.npy"

embeddings = []
labels = []

for user in os.listdir(EMB_DIR):
    user_dir = os.path.join(EMB_DIR, user)
    if not os.path.isdir(user_dir):
        continue

    for file in os.listdir(user_dir):
        if file.endswith(".npy"):
            emb = np.load(os.path.join(user_dir, file))
            embeddings.append(emb)
            labels.append(user)

embeddings = np.array(embeddings).astype("float32")

print(f"[INFO] {len(embeddings)} embeddings chargés")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, INDEX_PATH)
np.save(LABELS_PATH, np.array(labels))

print("✅ Index FAISS créé avec succès")
