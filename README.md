
<h1 align="center">🌀 Wasserstein Morphing with mamGAN</h1>

<p align="center">
  <em>Wasserstein barycenter-based image morphing using the Method of Averaged Marginals (MAM), projected into the latent space of pretrained GANs.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-green"/>
  <img src="https://img.shields.io/badge/build-passing-brightgreen"/>
  <img src="https://img.shields.io/badge/platform-MPI%20%7C%20CPU-lightgrey"/>
  <img src="https://img.shields.io/github/last-commit/dan-mim/wasserstein-morphing-mamGAN"/>
  <img src="https://img.shields.io/github/repo-size/dan-mim/wasserstein-morphing-mamGAN"/>
</p>

---

## 🚀 Overview

This repository explores **image morphing** through **Wasserstein barycenters** computed in the **latent space of pretrained GANs**. It supports both:

- **Sinkhorn regularized OT**, and
- **MAM (Method of Averaged Marginals)**: an exact, operator-splitting-based OT solver see [our article for more](https://arxiv.org/pdf/2309.05315).

The morphing effect is achieved by computing Wasserstein barycenters between multiple images **after mapping them to a latent space**, then **re-generating images** using GANs.
This project aims to present an efficient and interpretable method to improve the realistic appearance and structural quality of Wasserstein barycenters, especially in the challenging case where the barycenter involves more than two input images.
By leveraging the Method of Averaged Marginals (MAM) and projecting barycenters into a generative latent space, the resulting morphings preserve both content fidelity and visual coherence.

---

## 📦 Package Structure

```
mam_bary/
├── __init__.py
├── barycenter.py        # Entry point for barycenter(im1, im2, ...) or barycenter(images)
├── mam_algorithm.py     # Core MAM loop and DR-based projection
├── projections.py       # Projection onto simplex, ℓ0, GAN, convex constraints
├── generators.py        # DCGAN / Pix2Pix model loading and inference
├── metrics.py           # Cost matrices, Wasserstein distances
├── io.py                # Image loading, saving, result parsing
├── utils.py             # Helper functions: display, reshaping, task division
```

---

## 📊 Visualization

You can use `visualization.py` to load results stored in the `outputs/` folder and recreate visuals from the paper.

For example:
```bash
python visualization.py --input outputs/res_mamgan.pkl
```

---

## 📁 Other Folders

- **`post-treatment/`**: Contains utilities to convert generated images into **GIFs** or **videos**.
- **`utils/`**: Includes lightweight function sketches to help understand how MAM and latent-space projections operate.
- **`run_barycenter_of_pics_mamGAN.py`**: Uses 3 input images from a dataset and computes their barycenter via GAN projection.

---
## 🧩 Scientific Context and Extensions

This project builds upon and extends two major lines of research:

- 🖼️ **Image Barycenters with GANs**  
  The idea of computing barycenters in the latent space of generative models was first explored in  
  📄 *Simon et al., “Barycenters of Images with GANs”* [(arXiv:1912.11545)](https://arxiv.org/pdf/1912.11545),  
  and implemented in [this repository](https://github.com/drorsimon/image_barycenters).  
  Their work proposed using **Sinkhorn barycenters** in the latent space of **pretrained DCGANs**, allowing image interpolation with learned priors.  
  Our project revisits this idea with a stronger mathematical backbone, using **exact solvers** like **MAM** instead of entropic methods, and extending the pipeline to multiple GAN types (DCGAN, Pix2Pix).

- 🧠 **Constrained Optimal Transport**  
  This project also relies on recent advances in constrained OT formulations described in:  
  📘 *Mimouni et al., “Constrained Wasserstein Barycenters”* [(PDF)](https://dan-mim.github.io/files/constrained_Wasserstein.pdf)  
  → GitHub: [Constrained-Optimal-Transport](https://github.com/dan-mim/Constrained-Optimal-Transport)  
  The paper provides a **general operator-splitting framework** to incorporate constraints in Wasserstein barycenter computation, such as projections onto GAN manifolds or sparse supports.  
  The associated implementation has been tested and optimized for **industrial use** and is currently deployed at **IFPEN** for **energy system optimization**.

Together, these two sources shape the foundation of **mamGAN**, bridging theoretical precision, practical scalability, and visual quality for high-fidelity barycentric morphing.

---

## 🧠 Highlight: Method of Averaged Marginals (MAM)

- ✅ Exact solver for linear OT barycenter (no entropic smoothing)
- 🔄 Douglas–Rachford projection scheme
- 🔍 Scalable to large datasets, and adaptable to **constrained** and **projected** barycenters
- 🧩 Can be integrated with **GAN priors** and sparse regularizers

For more details, refer to:
- 📄 [Computing Wasserstein Barycenters via Operator Splitting (SIAM, 2024)](https://dan-mim.github.io/files/Computing_Wasserstein_Barycenters_via_operator_splitting.pdf)
- GitHub: https://github.com/dan-mim/computing-wasserstein-barycenters-MAM

---

## 🛠 Installation

```bash
git clone https://github.com/dan-mim/wasserstein-morphing-mamGAN.git
cd wasserstein-morphing-mamGAN
pip install -r requirements.txt
```

Make sure to install PyTorch and any GPU drivers as required for DCGAN or Pix2Pix inference.

---

## 🔧 Usage

```bash
python run_barycenter_of_pics_mamGAN.py \
  --image_1_path img1.jpg \
  --image_2_path img2.jpg \
  --image_3_path img3.jpg \
  --model pix2pix --projection gan --solver mam
```

---

## 📜 License

For academic use, please cite the original references listed above.

---
