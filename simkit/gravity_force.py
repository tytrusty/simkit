import numpy as np

from .massmatrix import massmatrix


def gravity_force(X, T, a=-9.8, rho=1):
    """
    Compute gravity force
    """

    dim = X.shape[1]
    # Compute the volume of each triangle
    M = massmatrix(X, T, rho=rho)
    
    g = np.zeros((X.shape[0], dim))
    g[:, 1] = a
    g =  (M @ g)
    return g