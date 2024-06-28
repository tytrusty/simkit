import numpy as np

def kinetic_energy(x : np.ndarray, y : np.ndarray, M, h : float):
    d = x - y
    g = 0.5 * d.T @ M @ d * (1/ (h**2))
    return g
