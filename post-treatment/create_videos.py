import cv2
import os
import re

def sort_key(filename):
    # Extrait le numéro de l'image en utilisant une expression régulière
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else float('inf')

def create_video_from_images(folder_path, output_name="output.mp4", fps=10):
    # Liste des fichiers d'image triés par numéro
    images = sorted([img for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))], key=sort_key)
    if not images:
        print("Aucune image trouvée dans le dossier.")
        return

    # Lecture de la première image pour obtenir les dimensions
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialisation du writer vidéo avec codec MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour fichier .mp4
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    # Ajout de chaque image au flux vidéo avec le nom affiché
    for image in images:
        img_path = os.path.join(folder_path, image)
        frame = cv2.imread(img_path)

        # Ajout du texte avec le nom de l'image sans extension
        image_name = os.path.splitext(image)[0]
        cv2.putText(frame, image_name, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Ajout de l'image avec texte au flux vidéo
        video.write(frame)

    # Libération des ressources
    video.release()
    print(f"Vidéo créée et sauvegardée sous {output_name}")

# Utilisation de la fonction
create_video_from_images("../results_MAM/MAM_projNorm0", "results_MAM/MAM_projNorm0RGB.mp4", fps=2)
create_video_from_images("../results_MAM/MAM_projNorm0GAN", "results_MAM/MAM_projNorm0RGB_GAN.mp4", fps=2)
