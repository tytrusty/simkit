import numpy as np



def kinetic_energy(x : np.ndarray, y : np.ndarray, M, h : float):
    """
    Computes the kinetic energy of the elastic system

    Parameters
    ----------
    x : (n*d, 1) numpy array
        positions of the elastic system

    y : (n*d, 1) numpy array
        inertial target positions (2 x_curr -  x_prev or equivalently x_curr + h * x_dot_curr)

    Returns
    -------
    k : float
        Kinetic energy of the elastic system
    """
    d = x - y
    g = 0.5 * d.T @ M @ d * (1/ (h**2))
    return g
