
import numpy as np
from .arap_gradient import arap_gradient

def elastic_gradient(x: np.ndarray, V: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, material, J=None, vol=None):

    if material == 'arap':
        return arap_gradient(x, V, T, mu, J=J, vol=vol)
    else:
        raise ValueError("Unknown material type: "  + material)
    return

