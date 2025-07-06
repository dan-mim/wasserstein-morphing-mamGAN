from numba import njit
import numpy as np

from .utils import preprocess_Q

# Projection onto the simplex with NUMBA
@njit
def _proj(V, z):
    n_features = V.shape[1]
    result = np.zeros_like(V)
    for i in range(V.shape[0]):
        U = np.sort(V[i])[::-1]  # Tri décroissant
        cssv = 0.0  # Initialisation de la somme cumulative
        rho = -1
        # Calcul de la somme cumulative manuellement
        for j in range(n_features):
            if U[j] - (cssv + U[j] - z) / (j + 1) > 0:  # Utilisation de z dans la condition
                rho = j
            else:
                break
            cssv += U[j]
        # Éviter une division par zéro en s'assurant que rho >= 0
        if rho >= 0:
            theta = (cssv- z) / (rho + 1)  # Intégration de z dans le calcul de theta
        else:
            theta = 0
        result[i] = np.maximum(V[i] - theta, 0)

    return result

@njit
def projection_simplex_numba(V, z=1, axis=None):
    if axis == 1:
        result = _proj(V,z)
        return result

    elif axis == 0:
        return projection_simplex_numba(V.T, z, axis=1).T

    else:
        # En cas d'absence de spécification d'axe, on traite V comme un vecteur plat
        V_flat = V.ravel()
        n = V_flat.size
        U = np.sort(V_flat)[::-1]
        cssv = np.cumsum(U) - z
        rho = 0
        for j in range(n):
            if U[j] - cssv[j] / (j + 1) > 0:
                rho = j + 1
            else:
                break
        theta = cssv[rho - 1] / rho
        return np.maximum(V_flat - theta, 0).reshape(V.shape)

# Vectorize function that project vectors onto a simplex, inspired by https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()

def proj_norm0(x,k):
    """
    The function is the projection of the vector x into the ensemble E={x: ||x||0=kk}
    """
    reshape = False
    if len(x.shape)>1:
        height, width = x.shape
        x = x.reshape(height*width)
        reshape = True

    n = len(x)

    # Vérifications des conditions
    if k < 0:
        raise ValueError("The number k of nonzero components must be nonnegative")
    elif k > n:
        raise ValueError("The number k of nonzero components must be less or equal to n")

    # Calcul du nombre d'éléments à mettre à zéro
    kk = n - k

    # On trie les indices des éléments de x
    I = np.argsort(x)

    # Copie de x pour la projection
    p = np.array(x, copy=True)

    # Remplacement des kk plus petits éléments par 0
    p[I[:kk]] = 0

    if reshape:
        return p.reshape((height, width))
    return p

def project_ontoNorm0(im0, ksparse):
    im, max_val, im_counts = preprocess_Q(im0.transpose(2, 0, 1), one_pic=True)
    R, G, B = im[0], im[1], im[2]
    R, G, B = proj_norm0(R,ksparse), proj_norm0(G,ksparse), proj_norm0(B,ksparse)
    R, G, B = max_val - R*im_counts[0, 0, 0], max_val - G*im_counts[1, 0, 0], max_val - B*im_counts[2, 0, 0]
    im = np.stack([R, G, B], axis=2)
    # plt.imshow(im)
    # plt.title(f'{ksparse} pixels')
    # plt.show()
    return(im)
