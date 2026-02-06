import numpy as np
from insightface.app import FaceAnalysis

class FaceEmbeddingExtractor:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def get_embedding(self, image):
        faces = self.app.get(image)
        if len(faces) == 0:
            return None
        return faces[0].embedding.astype("float32")
