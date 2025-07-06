import numpy as np
import matplotlib.pyplot as plt
from MAM_proj_Bx import *


# Créer des images de couleurs RGB simples
def create_color_image(color, size=(10, 10)):
    """Crée une image remplie d'une couleur spécifique."""
    im = np.tile(np.array(color).reshape(1, 1, 3), (size[0], size[1], 1))
    return im.reshape(-1,3)


# Définir trois images de couleurs : rouge, vert, bleu avec des valeurs RGB entre 0 et 255
image1 = create_color_image([225, 0, 0])  # Image rouge
image2 = create_color_image([0, 225, 0])  # Image verte
image3 = create_color_image([0, 0, 225])  # Image bleue

# Liste des images (chaque image est une mesure de probabilité)
images = [image1, image2, image3]

res = MAM_projX(images, M_dist=[], rho=1000, ksparse=False)



# Afficher les images et le barycentre avant projection
plt.figure(figsize=(10, 3))
plt.subplot(1, 4, 1)
plt.imshow(image1.astype(np.uint8))
plt.title("Image Rouge")

plt.subplot(1, 4, 2)
plt.imshow(image2.astype(np.uint8))
plt.title("Image Verte")

plt.subplot(1, 4, 3)
plt.imshow(image3.astype(np.uint8))
plt.title("Image Bleue")

plt.subplot(1, 4, 4)
# plt.imshow(barycenter.astype(np.uint8))
plt.title("Barycentre (Non Projeté)")
plt.show()


# Définir l'ensemble X pour ne conserver que les teintes de bleu
def MAM_projX(barycenter, X=None):
    """
    Project the computed barycenter onto the convex set X.
    X est un ensemble qui impose une contrainte sur les teintes (couleurs).
    Dans cet exemple, nous ne conservons que les pixels dominés par le bleu.
    """
    projection = np.copy(barycenter)

    # Contrainte : Conserver uniquement les pixels dominés par la composante bleue
    blue_threshold = 100  # On garde les pixels où la composante bleue est > 100
    mask = projection[:, :, 2] > blue_threshold

    # On garde seulement les pixels bleus, les autres deviennent noirs
    projection[~mask] = [0, 0, 0]

    # Renormalisation (ici ce n'est pas nécessaire car ce ne sont pas des probabilités)
    return projection



