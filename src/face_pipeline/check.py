import os

aligned_dir = "data/aligned_faces/"

if not os.path.exists(aligned_dir):
    print(" Le dossier aligned_faces n'existe pas !")
    exit()

files = [f for f in os.listdir(aligned_dir) if f.endswith(".jpg")]

if not files:
    print("Aucun visage aligné n'a été trouvé !")
else:
    print(f" {len(files)} visages alignés trouvés :")
    for f in files:
        print("   -", f)

print("Vérification terminée !")
