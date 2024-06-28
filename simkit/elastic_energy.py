import numpy as np


from .arap_energy import arap_energy


def elastic_energy(x: np.ndarray, V: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, material, J=None, vol=None):

  

    if material == 'linear_elasticity':
        raise NotImplemented
    elif material == 'arap':
        e = arap_energy(x, V, T, mu, J=J, vol=vol)
    elif material == 'corot':
        raise NotImplemented
    elif material == 'neo_hookean':
        raise NotImplemented
    else:
        raise ValueError("Unknown material type: " + material)
    return e
