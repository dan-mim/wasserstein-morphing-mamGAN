"""
MAM algorithm
"""
## Imports
# Basics
from numba import njit
import numpy as np
import time
# My codes
from .project import projection_simplex_numba, proj_norm0
from .io import display


@njit
def initialization_MAM(b, M_dist, height, width, rho):
    # Initialisations
    M = len(b)           # Nombre de distributions de probabilité
    R = height * width    # Dimension des images
    sum_theta_mean = np.zeros(R)
    inv_sum_S = 0.0

    # Conversion des tableaux
    S = np.zeros(M)
    theta = [np.zeros((R, np.sum(b[m] > 0))) for m in range(M)]  # stockage des transport plans

    # Boucle de calcul
    for m in range(M):
        I = b[m] > 0
        S[m] = np.sum(I)
        inv_sum_S += 1 / S[m]

        # Calcul de theta[m]
        theta[m] = -1 / rho * M_dist[:, I]

        # Calcul de la moyenne manuellement
        for r in range(R):
            sum_theta_mean[r] += np.sum(theta[m][r]) / S[m]

    # Calcul de p
    p = sum_theta_mean / inv_sum_S
    return p, M, S, R, theta, inv_sum_S

@njit
def _update_theta_MAM(b, M_dist, p, theta, rho, t, S, m):
    I = b[m] > 0  # Indices où b[m] est supérieur à zéro

    # Calculer deltaU
    deltaU = (p - np.sum(theta[m], axis=1)) / S[m]
    deltaU = deltaU[:, np.newaxis]  # Augmenter la dimension de deltaU

    # Mettre à jour theta[m]
    theta[m] -= (1 / rho) * M_dist[:, I]  # Mise à jour avec M_dist
    theta[m] += 2 * t * deltaU  # Ajouter 2 * t * deltaU
    theta[m] /= b[m][I]  # Normaliser theta[m]

    return theta, deltaU, I

def MAM_colored_images(b, M_dist, height, width, rho=1000, ksparse=False, display_p=False,
                       computation_time=10, iterations_min=3, iterations_max=1000, precision=10 ** -100):

    st00 = time.time()

    # Initialization
    p, M, S, R, theta, inv_sum_S = initialization_MAM(b, M_dist, height, width, rho)


    # Algorithm iterations:
    spent_time = 0
    iterations_k = 0
    print('time for initialization : ',time.time() - st00)
    while (iterations_k < iterations_min) or (spent_time < computation_time and iterations_k < iterations_max and evol_p > precision):  # and count_stop<10):  #  # and evol_p>10**-6) : # and evol_p>10**-16
        iterations_k = iterations_k + 1
        start = time.time()

        # Initialize for inner loops
        sum_theta_mean = np.zeros(R)

        t = 1  # if balanced Wasserstein barycenter

        # PARALLELIZATION
        # iterate over probabilities
        for m in range(M):
            t2 = time.time()

            # Update theta before the projection
            theta, deltaU, I = _update_theta_MAM(b, M_dist, p, theta, rho, t, S, m)

            # the transport plan is un-normalized after the projection onto the simplex
            print(f"iteration={iterations_k}, for m={m}", 'time before the projection ',time.time() - t2)
            t1 = time.time()
            theta[m] = projection_simplex_numba(theta[m], z=1, axis=0) * b[m][I] #projection_simplex
            print('time for the projection ',time.time() - t1)

            theta[m] -= t * deltaU

            # mean of theta:
            sum_theta_mean = sum_theta_mean + np.mean(theta[m], axis=1)  # equivalent to: np.sum(theta[m], axis=1) /S[m]

        # Stopping criterion: using the norm max
        evol_p = np.max(np.abs(p - sum_theta_mean / inv_sum_S))

        # Compute the approximated barycenter:
        p = sum_theta_mean / inv_sum_S
        if display_p:
            display(p, height, width, title='p')
        if ksparse:
            tic = time.time()
            p = proj_norm0(p,ksparse)
            print(f'projection onto normO time {time.time() - tic}')
            display(p, height, width, title=f'after ProjNorm0 {ksparse} pixels')

        # Time management: Here is the end of one iteration
        end = time.time()
        iteration_time = np.round((end - start), 2)
        spent_time = end - st00 # manage time at a global scale

    # Output
    p = p.reshape(height,width)
    p = p/p.sum()
    # np.imshow
    print(f"iterations = {iterations_k}, computation time = {spent_time}, approximated precision = {evol_p}")
    return p

