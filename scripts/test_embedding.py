import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from embeddings.extractor import FaceEmbeddingExtractor

extractor = FaceEmbeddingExtractor(device="cpu")

result = extractor.extract("data/test/face.jpg")

if result is None:
    print("❌ Aucun visage détecté")
else:
    print("✅ Embedding extrait avec succès")
    print("Shape :", result["embedding"].shape)
    print("Confidence :", result["confidence"])
