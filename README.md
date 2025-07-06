# wasserstein-morphing-mamGAN
Wasserstein barycenter-based image morphing using both Sinkhorn and the Method of Averaged Marginals (MAM), applied in the latent space of a pretrained generative model (GAN).

visualization.py permets de visualiser les résultats de mamGAN saved in outputs folder et retrouver les visuels des figures.

le package est structuré comme suit:
mam_bary/
├── __init__.py
├── barycenter.py        # Point d’entrée public : barycenter(im1, im2, ...) ou barycenter(images)
├── mam_algorithm.py     # Implémentation de l'algorithme MAM (initialisation, itérations)
├── projections.py       # Fonctions de projection : simplexe, ℓ0, GAN, contraintes convexes
├── generators.py        # Chargement des modèles GAN (DCGAN, pix2pix, etc.)
├── metrics.py           # Fonctions de coût, distances, etc.
├── io.py                # Chargement/sauvegarde d’images, parsing de résultats
├── utils.py             # Helpers génériques (division de tâches, reshape, display)

In the folder post-treatment you can find some documents to make your photos into gifs or videos

[text](run_barycenter_of_pics_mamGAN.py) takes 3 pictures from the dataset and compute the projected barycenter thanks to the pretrained GAN.

In utils I put some sketchs of basic functions to make you understand how the algorithms work