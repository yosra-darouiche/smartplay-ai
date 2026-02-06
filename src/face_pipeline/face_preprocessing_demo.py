import cv2
import glob
from insightface.app import FaceAnalysis
from insightface.utils import face_align

print("SPRINT 2 - Face Preprocessing Pipeline")

# 1️⃣ Initialisation du modèle InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
print("Modèle prêt")

# 2️⃣ Charger toutes les images de test
image_paths = glob.glob("data/test/*.jpg")

if len(image_paths) == 0:
    print("Aucune image trouvée dans data/test_images/")
    exit()

print(f"{len(image_paths)} images trouvées")

# 3️⃣ Boucle sur les images
for path in image_paths:
    print("\nImage :", path)
    img = cv2.imread(path)

    if img is None:
        print("Impossible de charger l'image")
        continue

    faces = app.get(img)
    print("Visages détectés :", len(faces))

    # Copier image pour affichage
    display_img = img.copy()

    for i, face in enumerate(faces):
        # 🔲 Bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 🔴 Landmarks
        for (x, y) in face.kps:
            cv2.circle(display_img, (int(x), int(y)), 2, (0, 0, 255), -1)

        # ✂️ Alignement du visage
        aligned_face = face_align.norm_crop(img, face.kps)

        print(f"  - Visage {i} aligné")

        # Afficher le visage aligné
        cv2.imshow(f"Aligned Face {i}", aligned_face)
        cv2.waitKey(500)

    # Afficher image originale avec bbox + landmarks
    cv2.imshow("Detection + Landmarks", display_img)
    cv2.waitKey(1000)

cv2.destroyAllWindows()
print("\nSprint 2 terminé avec succès")
