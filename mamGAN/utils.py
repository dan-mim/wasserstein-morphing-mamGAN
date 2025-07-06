## Imports
# Basics
from typing import List, Tuple
import numpy as np



def preprocess_Q(Q: np.ndarray, max_val: float = None, Q_counts: np.ndarray = None, one_pic=False) -> Tuple[
    np.ndarray, float, np.ndarray]:
    """ Preprocess (normalize) input images before computing their barycenters

    Arguments:
        Q {np.ndarray} -- Input images. Every image should reshaped to a column in Q.

    Keyword Arguments:
        max_val {float} -- The maximum value. Should be changed from None when using the iterative algorithm (more than 1 iteration in the Algorithm) (default: {None})
        Q_counts {np.ndarray} -- The sum of all the pixel values in each image. Should be changed from None when using the iterative algorithm (more than 1 iteration in the Algorithm) (default: {None})

    Returns:
        Tuple[np.ndarray, float, np.ndarray] -- The normalized images the total maximum value and sum of pixels in each image
    """
    if max_val is None:
        max_val = Q.max()
    Q = max_val - Q
    if one_pic:
        # Q_counts = np.sum(Q, axis=(1, 2))
        # for i in range(3):
        #     Q[i,:,:] = Q[i,:,:] / Q_counts[i]
        # return Q, max_val, Q_counts
        Q_counts = np.sum(Q, axis=(1, 2)).reshape(3, 1, 1)
        Q = Q / Q_counts
        return Q, max_val, Q_counts
    if Q_counts is None:
        Q_counts = np.sum(Q, axis=1, keepdims=True)
    Q = Q / Q_counts
    return Q, max_val, Q_counts



def division_tasks(nb_tasks, pool_size):
    """
    Inputs: (int)
    *nb_tasks
    *pool_size : number of CPU/GPU to divide the tasks between

    Outputs:
    rearranged: numpy list of lists so that rearranged[i] should be treated by CPU[i] (rank=i)
    """
    # The tasks can be equaly divided for each CPUs
    if nb_tasks % pool_size == 0:
        rearranged = np.array([i for i in range(nb_tasks)])
        rearranged = np.split(rearranged, pool_size)

    # Some CPUs will receive more tasks
    else:
        div = nb_tasks // pool_size
        congru = nb_tasks % pool_size
        rearranged1 = np.array([i for i in range(div * congru + congru)])
        rearranged1 = np.split(rearranged1, congru)
        rearranged2 = np.array([i for i in range(div * congru + congru, nb_tasks)])
        rearranged2 = np.split(rearranged2, pool_size - congru)
        rearranged = rearranged1 + rearranged2

    # Output:
    return (rearranged)