import numpy as np
from typing import Tuple
from scipy import spatial

# Compute the L2 metric for the transportation cost. Can probably be vectorized to run faster.
def _generate_metric(height, width, grid):
    S = height * width
    R = S
    # eps_R and eps_S are the discretization size of the barycenter image (for R) and the sample images (for S)
    eps_R_height, eps_R_width = 1 / height, 1 / width
    eps_S_height, eps_S_width = 1 / height, 1 / width

    la = np.arange(R)
    lb = np.arange(S)

    x_R, y_R = la % width * eps_R_width + eps_R_width / 2, la // width * eps_R_height + eps_R_height / 2
    x_S, y_S = lb % width * eps_S_width + eps_S_width / 2, lb // width * eps_S_height + eps_S_height / 2

    XA = np.column_stack((x_R, y_R)) #np.array([x_R, y_R]).T
    XB = np.column_stack((x_S, y_S)) #np.array([x_S, y_S]).T
    M_dist = spatial.distance.cdist(XA, XB, metric='euclidean')
    return M_dist**2

def generate_metric(im_size: Tuple[int]) -> np.ndarray:
    """
    Computes the Euclidean distances matrix

    Arguments:
        im_size {Tuple[int]} -- Size of the input image (height, width)

    Returns:
        np.ndarray -- distances matrix
    """
    grid = np.meshgrid(*[range(x) for x in im_size])
    grid = np.stack(grid, -1).astype(np.int64)
    return _generate_metric(im_size[0], im_size[1], grid)