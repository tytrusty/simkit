import numpy as np

def kinetic_gradient(x : np.ndarray, y : np.ndarray, M, h : float):
    d = x - y
    g = M @ d * (1/ (h**2))
    return g
