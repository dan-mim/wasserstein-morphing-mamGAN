## Imports
# Basics
from typing import Tuple
import numpy as np
import torch
from os.path import join
import time

# My codes
from .io import load_pix2pix
# GAN
from .generators.dcgan_models import Generator, Encoder
from .generators.pix2pix.models import networks
from .generators.utils import load_generator, load_encoder, denorm


def load_models(args) -> Tuple[Generator, Encoder, networks.UnetGenerator]:
    """ Load the generative models

    Returns:
        Tuple[Generator, Encoder, networks.UnetGenerator] -- the DCGAN generator, its respective encoder and the Pix2Pix model
    """
    generator = load_generator(args.dcgan_latent_size, args.dcgan_num_filters,
                                     join(args.models_save_dir, 'generator'))
    encoder = load_encoder(args.dcgan_latent_size, args.dcgan_num_filters, join(args.models_save_dir, 'encoder'))
    pix2pix = load_pix2pix(join(args.models_save_dir, args.dataset_name + "_pix2pix"))
    return generator, encoder, pix2pix


def project_ontoGAN(im, dcgan_size, pix2pix_size, Generator, Encoder, pix2pix):
    record_time = time.time()
    im = torch.tensor(im).permute(2, 0, 1).numpy()
    img_size = im.shape[1:]
    GAN_projections = project_on_generator(Generator, pix2pix, im, Encoder, dcgan_img_size=dcgan_size, pix2pix_img_size=pix2pix_size)
    print('projection onto latent space and back took ', time.time() - record_time)
    GAN_projections_images, GAN_projections_noises = GAN_projections
    out_ours = torch.Tensor(GAN_projections_images).reshape(3,*img_size).permute(1, 2, 0).numpy()
    return(out_ours)

def project_on_generator(G: Generator, pix2pix: networks.UnetGenerator,
                         target_image: np.ndarray, E: Encoder, dcgan_img_size: int = 64,
                         pix2pix_img_size: int = 128) -> Tuple[np.ndarray, torch.Tensor]:
    """Projects the input image onto the manifold span by the GAN. It operates as follows:
    1. reshape and normalize the image
    2. run the encoder to obtain a latent vector
    3. run the DCGAN generator to obtain a low resolution image
    4. run the Pix2Pix model to obtain a high resulution image

    Arguments:
        G {Generator} -- DCGAN generator
        pix2pix {networks.UnetGenerator} -- Low resolution to high resolution Pix2Pix model
        target_image {np.ndarray} -- The image to project
        E {Encoder} -- The DCGAN encoder

    Keyword Arguments:
        dcgan_img_size {int} -- Low resolution image size (default: {64})
        pix2pix_img_size {int} -- High resolution image size (default: {128})

    Returns:
        Tuple[np.ndarray, torch.Tensor] -- The projected high resolution image and the latent vector that was used to generate it.
    """
    # reshape and normalize image
    target_image = torch.Tensor(target_image).cuda().reshape(1, 3, pix2pix_img_size, pix2pix_img_size)
    target_image = G.interpolate(target_image, scale_factor=dcgan_img_size / pix2pix_img_size, mode='bilinear')
    target_image = target_image.clamp(min=0)
    target_image = target_image / target_image.max()
    target_image = (target_image - 0.5) / 0.5

    # Run dcgan
    z = E(target_image)
    dcgan_image = G(z)

    # run pix2pix
    pix_input = G.interpolate(dcgan_image, scale_factor=pix2pix_img_size / dcgan_img_size, mode='bilinear')
    pix_outputs = pix2pix(pix_input)
    out_image = denorm(pix_outputs.detach()).clamp(0, 1).cpu().numpy().reshape(3, -1, 1)
    return out_image, z

