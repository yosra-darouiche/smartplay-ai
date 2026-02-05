import cv2
from insightface.app import FaceAnalysis

class FaceEmbeddingExtractor:
    def __init__(self, device="cpu"):
        providers = (
            ["CUDAExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0 if device == "cuda" else -1)

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image non trouvée")

        faces = self.app.get(img)

        if len(faces) == 0:
            return None

        face = faces[0]

        return {
            "embedding": face.normed_embedding,
            "confidence": float(face.det_score),
            "bbox": face.bbox.astype(int).tolist()
        }
