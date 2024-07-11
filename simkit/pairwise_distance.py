import numpy as np

from .pairwise_displacement import pairwise_displacement

def pairwise_distance(X : np.ndarray, Y : np.ndarray, return_displacement=False):
    """
    Returns the pairwise displacement between each row of X to each row of Y, in a 3-tensor D

    D[i, j] = ||X[i, :] - Y[j, :]||
    """
    D = pairwise_displacement(X, Y)

    R = np.linalg.norm(D, axis=2)
    if return_displacement:
        return R, D
    else:
        return R