import numpy as np

from .arap_hessian import arap_hessian

def elastic_hessian(x: np.ndarray, V: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, material):
    if material == 'arap':
        return arap_hessian(x, V, T, mu)
    else:
        raise ValueError("Unknown material type: " + material)
    return

