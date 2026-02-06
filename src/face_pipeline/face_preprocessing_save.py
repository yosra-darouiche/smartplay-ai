from insightface.app import FaceAnalysis
import cv2
import os

# Initialisation de l'application
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU utilisé

# Dossiers d'entrée et sortie
input_dir = "data/test/"
output_dir = "data/aligned_faces"
os.makedirs(output_dir, exist_ok=True)

# Parcours des images
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Impossible de lire {img_name}")
        continue

    faces = app.get(img)

    if not faces:
        print(f"Aucun visage détecté dans {img_name}")
        continue

    # Sauvegarde des visages alignés
    for i, face in enumerate(faces):
        # Vérifie que l'objet face contient la face alignée
        if hasattr(face, "normed_face") and face.normed_face is not None:
            aligned_face = face.normed_face
            save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_{i}.jpg")
            cv2.imwrite(save_path, aligned_face)
            print(f" Visage aligné sauvegardé : {save_path}")
        else:
            print(f" Impossible d'aligner le visage {i} dans {img_name}")
