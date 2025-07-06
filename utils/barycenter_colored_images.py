# Basics
import pickle

import matplotlib.pyplot as plt
import numpy as np

# My codes:
# MAM parallel with the unbalanced configuration if needed
# from MAM_non_convex import *
from MAM_proj_Bx import *
from PIL import Image

# functions
def normalize(vec):
    # print(np.sum(vec))
    return vec/np.sum(vec)

def central_square(img, square_size):
    hauteur, largeur = img.size
    left = (largeur - square_size) // 2
    top = (hauteur - square_size) // 2
    right = left + square_size
    bottom = top + square_size

    # Extraire le carré central
    central_square =  img.crop((left, top, right, bottom))
    return central_square

# List of measures
# liste_images = [18,19,22,24,35]
liste_images = [72, 68] #[18, 35]
bR, bG, bB = [], [], []
r_mean, g_mean, b_mean = [], [], []
for i in liste_images:
    img = Image.open(f'dataset/img_align_celeba/0000{i}.jpg')
    # img.show()
    # Extraire le carré central square_size x square_size
    square_size = 80
    im = central_square(img, square_size)

    hauteur, largeur = np.array(im)[:,:,0].shape
    # je conserve les valeurs moyennes
    r_mean.append( np.sum( np.reshape( np.array(im)[:,:,0], (hauteur*largeur)) ) )
    g_mean.append( np.sum( np.reshape( np.array(im)[:,:,1], (hauteur*largeur)) ) )
    b_mean.append( np.sum( np.reshape( np.array(im)[:,:,2], (hauteur*largeur)) ) )
    # je normalise
    bR.append( normalize(np.reshape( np.array(im)[:,:,0], (hauteur*largeur))) )
    bG.append( normalize(np.reshape( np.array(im)[:,:,1], (hauteur*largeur))) )
    bB.append( normalize(np.reshape( np.array(im)[:,:,2], (hauteur*largeur))) )


# Distance matrix:
def compute_M_dist(hauteur, largeur):
    # Taille des matrices
    R = hauteur * largeur  # Nombre de pixels dans l'image de barycentre
    S = R  # Nombre de pixels dans l'image de base

    # eps_R et eps_S
    eps_R = 1 / (R ** .5)
    eps_S = 1 / (S ** .5)

    # Création des coordonnées
    la = np.linspace(0, R - 1, R)
    lb = np.linspace(0, S - 1, S)

    # Coordonnées de l'image barycentrique
    x_R = la % largeur * eps_R + eps_R / 2
    y_R = la // largeur * eps_R + eps_R / 2

    # Coordonnées des images échantillons
    x_S = lb % largeur * eps_S + eps_S / 2
    y_S = lb // largeur * eps_S + eps_S / 2

    # Création des matrices de coordonnées
    XA = np.array([x_R, y_R]).T
    XB = np.array([x_S, y_S]).T

    # Calcul de la matrice de distance
    M_dist = spatial.distance.cdist(XA, XB, metric='euclidean')

    return M_dist**2

M_dist = compute_M_dist(hauteur, largeur)
# plt.imshow(M_dist)
print("Compute barycenter...")
tps = 10
res = MAM_projX(bR, M_dist=M_dist, rho=1000, ksparse=False, computation_time=tps)
baryR = res[0]

res = MAM_projX(bR, M_dist=M_dist, rho=1000, ksparse=False, computation_time=tps)
baryG = res[0]

res = MAM_projX(bR, M_dist=M_dist, rho=1000, ksparse=False, computation_time=tps)
baryB = res[0]

def bary_to_image(R, G, B, r_mean, g_mean, b_mean, hauteur=80, largeur=80):
    # Créer une image à partir des listes normalisées R, G, B
    R1 = np.array(R).reshape((hauteur, largeur)) *  (r_mean[0] + r_mean[1])/2  # Multiplier par 255
    G1 = np.array(G).reshape((hauteur, largeur))  *  (g_mean[0] + g_mean[1])/2
    B1 = np.array(B).reshape((hauteur, largeur))  *  (b_mean[0] + b_mean[1])/2

    # Combiner les canaux en une seule image
    image_array = np.stack((R1, G1, B1), axis=-1)

    # Convertir les valeurs d'intensité en format uint8
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)  # Assurez-vous que les valeurs sont dans la plage [0, 255]

    # Créer l'image avec PIL
    image = Image.fromarray(image_array)
    return image

# visualize barycenter:
# plt.figure()
image = bary_to_image(baryR, baryG, baryB,  r_mean, g_mean, b_mean, hauteur=80, largeur=80)
image.show()
# plt.imshow(np.reshape(baryR, (80,80)))
# plt.imshow(np.reshape(baryG, (80,80)))
# plt.imshow(np.reshape(baryB, (80,80)))