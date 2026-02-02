import cv2
import csv
import os

VIDEO_PATH = "data/raw/sample.mp4"
OUTPUT_CSV = "data/outputs/player_positions.csv"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Impossible d'ouvrir la vidéo")
        return

    os.makedirs("data/outputs", exist_ok=True)

    with open(OUTPUT_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "player_id", "x", "y"])

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape

            # Fake player detection (Sprint 1 = pipeline, pas précision)
            player_x = width // 2
            player_y = height // 2

            writer.writerow([frame_id, 1, player_x, player_y])
            frame_id += 1

    cap.release()
    print("CSV généré avec succès :", OUTPUT_CSV)

if __name__ == "__main__":
    main()
