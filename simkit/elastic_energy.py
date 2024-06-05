import numpy as np




def elastic_energy(x: np.ndarray, V: np.ndarray, T: np.ndarray, my: np.ndarray, lam: np.ndarray, material):

    if material == 'linear_elasticity':
        raise NotImplemented
    elif material == 'arap':
        arap_energy(x, V, T, mu)
    elif material == 'corot':
        raise NotImplemented
    elif material == 'neo_hookean':
        raise NotImplemented
    return
