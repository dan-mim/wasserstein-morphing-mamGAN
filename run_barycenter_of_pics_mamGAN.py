# -*- coding: utf-8 -*-
"""
(c) Daniel Mimouni
"""

## Imports
# Basics
import matplotlib.pyplot as plt
import sys
import argparse
from typing import List
# my codes
from mamGAN import load_images, bary_mamGAN, load_models, get_jpg_images

parser = argparse.ArgumentParser()
parser.add_argument("--dcgan_latent_size", type=int, dest="dcgan_latent_size", help="The size of the latent vectors used in the DCGAN model", default=100)
parser.add_argument("--dcgan_num_filters", type=List[int], dest="dcgan_num_filters", help="Num of filters in each of the DCGAN's layers", default=[1024, 512, 256, 128])
parser.add_argument("--dcgan_image_size", type=int, dest="dcgan_image_size", help="input/output size for the DCGAN model", default=64)
parser.add_argument("--models_save_dir", type=str, dest="models_save_dir", help="The path to load the trained models from", default="outputs/networks")
parser.add_argument("--dataset_name", type=str, dest="dataset_name", help="Name of the used dataset", default="zap50k")
parser.add_argument("--dataset_base_dir", type=str, dest="dataset_base_dir", help="Path to the base dir of the dataset's jpg images", default="dataset/ut-zap50k-images-square")
parser.add_argument("--pix2pix_image_size", type=int, dest="pix2pix_image_size", help="input/output size for the pix2pix model. 128|256", default=128)
parser.add_argument("--results_folder", type=str, dest="results_folder", help="Path to store the interpolation results", default="results_MAM")
parser.add_argument("--entropy_regularization", type=float, dest="entropy_regularization", help="Value for the Wasserstein distance's entropy regularization", default=20.0)
parser.add_argument("--interpolation_steps", type=float, dest="interpolation_steps", help="Number of interpolation steps in the trainsformation", default=9)
parser.add_argument("--simulation_name", type=str, dest="simulation_name", help="A name for the simulation out file. If not set, simulation will be named by the chosen file names", default=None)
parser.add_argument("--image_1_path", type=str, dest="image_1_path", help="A path to the first image to interpolate from. If none, an image is selected randomly from the dataset.", default=None)
parser.add_argument("--image_2_path", type=str, dest="image_2_path", help="A path to the second image to interpolate to. If none, an image is selected randomly from the dataset.", default=None)
args = parser.parse_args()


# Load images
print("Loading images...")
if args.image_1_path is None or args.image_2_path is None:
    image_list = get_jpg_images(args.dataset_base_dir)
    images_paths = [a for a in image_list if "8082348.247448" in a or "7871491.184651" in a or '7750312.89' in  a]# [a for a in image_list if "8069538.387944" in a or "7393864.19085" in a or '7970329.248956' in  a] #
    # images_paths = np.random.choice(image_list, size=3, replace=False)
images, files_basename = load_images(images_paths, im_size=args.pix2pix_image_size)
print(images_paths)
for im in images:
    plt.close()
    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.pause(2)
plt.close()

# Load models
print("Loading models...")
generator, encoder, pix2pix = load_models(args)
simulation_name = args.simulation_name if args.simulation_name is not None else files_basename

print("Interpolating images:")
image_path = False
# image_path = 'results_MAM/10_iterations_ProjNorm0_3000_7393864_8069538_5098.065359476964.png'
# image_path = 'results_MAM/5000s_7393864_8069538_5098.065359476964.png'
# image_path = 'results_MAM/20_iterations_3pics_ProjNorm0_3000_7970329_7393864_8069538_5856.542919389917.png'  # Charger l'image et la convertir en tenseur
ksparse = False # 3000
iterationsMAM = 10
res = bary_mamGAN(images[:2],
                  generator, encoder, pix2pix,
                  args.entropy_regularization, args.interpolation_steps,
                  args.dcgan_image_size, args.pix2pix_image_size,
                  iterationsMAM,
                  f'{iterationsMAM}_iterations_3pics_ProjNorm0_{ksparse}_'+simulation_name, args.results_folder,
                  ksparse=ksparse,
                  image_path=image_path)

