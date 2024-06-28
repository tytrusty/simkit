import numpy as np

from .arap_hessian import arap_hessian_x

def elastic_hessian(x: np.ndarray, V: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, material, J=None, vol=None):
        
    if material == 'arap':
        return arap_hessian_x(x, V, T, mu, J=J, vol=vol)
    else:
        raise ValueError("Unknown material type: " + material)
    return

