import faiss
import numpy as np
import os

EMBEDDING_DIM = 512
INDEX_PATH = "data/enrolled_users/faiss.index"

def create_index():
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    os.makedirs("data/enrolled_users", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print("Index FAISS créé")

if __name__ == "__main__":
    create_index()
