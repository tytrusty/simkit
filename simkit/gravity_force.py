import numpy as np

from .massmatrix import massmatrix


def gravity_force(X, T, a=-9.8, rho=1):
    """
    Compute gravity force
    """
    # Compute the volume of each triangle
    M = massmatrix(X, T, rho=rho)
    
    g = np.zeros((X.shape[0], 2))
    g[:, 1] = a
    g =  (M @ g)
    return g