## Imports
# Basics
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from os.path import basename

# My codes
# GAN
from .generators.pix2pix.models import networks


def display(p, height, width, title=None):
    plt.close()
    plt.figure()
    plt.imshow(np.reshape(p, (height, width)))
    plt.title(title)
    plt.colorbar()
    plt.pause(.1)

def display_grid(bary_projected, baryprojGAN, l_nb_pixels):
    fig, axes = plt.subplots(len(l_nb_pixels), 2, figsize=(10, 5 * len(l_nb_pixels)))
    for i, (img_proj, img_gan) in enumerate(zip(bary_projected, baryprojGAN)):
        # Afficher bary_projected
        axes[i, 0].imshow(img_proj)
        axes[i, 0].axis('off')  # Désactiver les axes pour une meilleure visualisation
        axes[i, 0].set_title(f'Projection onto Norm0, nb_pixels={l_nb_pixels[i]}')

        # Afficher baryprojGAN
        axes[i, 1].imshow(img_gan)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Then projection onto GAN, nb_pixels={l_nb_pixels[i]}')

    # Ajuster l'espacement pour éviter les chevauchements
    plt.tight_layout()
    plt.show()


def load_images(paths: List, im_size: int = 128) -> Tuple[np.ndarray, np.ndarray, str]:
    """Loads images

    Arguments:
        path1 {str} -- path to the first image
        path2 {str} -- path to the second image

    Keyword Arguments:
        im_size {int} -- the desired image size (default: {128})

    Returns:
        Tuple[np.ndarray, np.ndarray, str] -- returns the two images and a the files' names (used later to save to results)
    """
    if im_size==0:
        im1 = np.array(Image.open(paths[0]).convert("RGB"), dtype=float) / 255
        return im1
    images = [np.array(Image.open(path).convert("RGB").resize((im_size, im_size), Image.LANCZOS), dtype=float) / 255
                for path in paths]  # np.float
    files_basename = '_'.join([basename(path).split('.')[0] for path in paths])
    return images, files_basename


def load_pix2pix(model_path: str = 'networks/zap50k_pix2pix') -> networks.UnetGenerator:
    """Loads the Pix2Pix model

    Keyword Arguments:
        model_path {str} -- Path to the trained Pix2Pix model (default: {'networks/zap50k_pix2pix'})

    Returns:
        networks.UnetGenerator -- An object that holds the Pix2Pix conditional generator model
    """
    netG = networks.define_G(3, 3, 64, 'unet_128', 'batch', True, 'normal', 0.02, [0]).module
    netG.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    netG.eval()
    for param in netG.parameters():
        param.requires_grad = False
    return netG


def load_results(image_path):
    """
    This function load a previous result
    :param image_path: {str} -- path to the first image
    :return:
    im1, im2, im3, barycenter
    """

    # Charger l'image en tant que tenseur
    loaded_image = load_images([image_path], im_size=0)

    # Séparer les canaux pour obtenir im1, im2, et out_OT
    loaded_image = loaded_image[2:-2, 2:-2, :]  # Enlever 2 pixels en haut et 2 en bas
    WIDTH = loaded_image.shape[1]
    # Dimensions de chaque sous-image sans bandes noires
    img_height, img_width = 128, 128

    # Extraire im1, im2, et im3 en sautant les bandes de séparation
    i = 0
    last_pixel = 0
    images = []
    while last_pixel < WIDTH:
        im = loaded_image[:, img_width*i + 2*i: (i+1) * img_width + 2*i, :]
        images.append( im )
        i += 1
        last_pixel = (i + 1) * img_width + 2 * i
    return(images)
