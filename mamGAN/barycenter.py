
## Imports
# Basics
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from os.path import join
from torchvision.utils import save_image

# My codes
from .mam import MAM_colored_images
from .project import project_ontoNorm0
from .generator import project_ontoGAN
from .io import display_grid, load_results
from .metrics import generate_metric
from .utils import preprocess_Q
# GAN
from .generators.dcgan_models import Generator, Encoder
from .generators.pix2pix.models import networks


def compute_bary(args):
    dim, Q, C, img_size, rho, max_val, Q_counts = args
    print(f"Color space {dim + 1}/3")
    bary =  MAM_colored_images([Q[dim, :, 0], Q[dim, :, 1]], C, img_size[0], img_size[1], rho=rho,
                              computation_time=60, iterations_min=10)
    bary = max_val - bary * (Q_counts[dim, 0, 0] + Q_counts[dim, 0, 1]) / 2
    return bary




def bary_mamGAN(images: List,
               Generator: Generator, Encoder: Encoder, pix2pix: networks.UnetGenerator,
               rho: float=1, L: int=9,
               dcgan_size: int=64,
               pix2pix_size: int=128,
               iterationsMAM: int=10,
               simulation_name: str="image_interpolation",
               results_path: str="results", ksparse=False, image_path=False
                ):

    if image_path :
        print( 'THE BARYCENTER IS NOT COMPUTED BUT LOADED FROM PREVIOUS RESULTS !!')
        bary = load_results(image_path)[-1]
        plt.imshow(bary)
        plt.show()
        l_nb_pixels = [130, 1000, 2000, 3000, 4000, 5000, 128**2]
        bary_projected, baryprojGAN = [], []
        for nb_sparse in l_nb_pixels:
            projection = project_ontoNorm0(bary, nb_sparse)
            bary_projected.append(projection) # Projection of each canal onto the Norm0 for nb_sparse per canal
            baryprojGAN.append( project_ontoGAN(projection, dcgan_size, pix2pix_size, Generator, Encoder, pix2pix) )
        display_grid(bary_projected, baryprojGAN, l_nb_pixels)
        baryprojGAN = project_ontoGAN(bary, dcgan_size, pix2pix_size, Generator, Encoder, pix2pix)
        plt.imshow(baryprojGAN)
        plt.title("Proj onto the GAN of the original picture")
        plt.show()
        return

    img_size = images[0].shape[:2]
    dimRGB_is_3 = len(images[0].shape) == 3
    if dimRGB_is_3 :
        dimRGB = 3
        images = [I.transpose(2, 0, 1).reshape(3, -1, 1) for I in images]


    print("Preparing transportation cost matrix...")
    C = generate_metric(img_size)
    Q = np.concatenate(images, axis=-1)
    Q, max_val, Q_counts = preprocess_Q(Q)
    out_ours = []
    out_GAN = []
    out_OT = []

    # print("Computing transportation plan...")
    # # Préparation des arguments
    rho = rho * np.mean(Q_counts)/10 #5098.065359476964
    # args_list = [(dim, Q, C, img_size, rho, max_val, Q_counts) for dim in range(3)]
    # with Pool(processes=3) as pool:
    #     barycenter = pool.map(compute_bary, args_list)
    print("Computing transportation plan...")
    barycenter = []
    barycenter_sinkhorn = []
    for dim in range(dimRGB):
        print(f"Color space {dim + 1}/3")
        images_measures = [Q[dim, :, i] for i in range(Q.shape[-1])]
        bary = MAM_colored_images(images_measures, C, img_size[0], img_size[1], rho=rho,
                                  computation_time=10, iterations_min=iterationsMAM, ksparse=ksparse, display_p=True)
        bary = max_val - bary * (Q_counts[dim, 0, 0] + Q_counts[dim, 0, 1])/2
        # plt.imshow(bary)
        # plt.show()
        barycenter.append( bary )
        # # methode sinkhorn
        # P = sinkhorn(Q[dim,:,0], Q[dim,:,1], C, img_size[0], img_size[1], 20)
        # t = .5
        # bary_sinkhorn = max_val - generate_interpolation(img_size[0],img_size[1],P,t)*((1-t)*Q_counts[dim,0,0] + t*Q_counts[dim,0,1])
        # # plt.imshow(bary_sinkhorn)
        # # plt.show()
        # barycenter_sinkhorn.append(bary_sinkhorn)
        # # barycenter_sinkhorn = np.stack(barycenter_sinkhorn, axis=0)

    if dimRGB_is_3:
        barycenter = np.stack(barycenter, axis=0)
        # barycenter_sinkhorn = np.stack(barycenter_sinkhorn, axis=0)

    # Save results:
    print("Saving results...")
    initial_images = [im.reshape(dimRGB, *img_size) for im in images]
    images = [torch.Tensor(im).reshape(dimRGB, *img_size) for im in initial_images]
    out_OT = torch.Tensor(barycenter).reshape(dimRGB,*img_size)
    # out_OT_sinkhorn = torch.Tensor(barycenter_sinkhorn).reshape(dimRGB,*img_size)
    # images.append(out_OT_sinkhorn)
    images.append(out_OT)
    out_OT = torch.stack(images)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    output_path = join(results_path, simulation_name+f'_{rho}.png')
    save_image(torch.cat([out_OT], dim=0), output_path, nrow=L, normalize=False, scale_each=False) #, range=(0,1))
    print(f"Image saved in {output_path}")