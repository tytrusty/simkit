import numpy as np

from .pairwise_distance import pairwise_distance


# pairwose distance
def gaussian_rbf(X, p):
    """
    A gaussian radial basis function as defined in  https://en.wikipedia.org/wiki/Radial_basis_function

    """
    dx = X.shape[1]
    px = p[:, 0:dx]
    pg = p[:, dx]

    r = pairwise_distance(X, px)

    phi = np.exp(-0.5 * (pg**2) * (r**2))

    return phi
