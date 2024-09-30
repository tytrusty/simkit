import numpy as np

from .average_onto_simplex import average_onto_simplex
from .skinning_eigenmodes import skinning_eigenmodes
from .spectral_clustering import spectral_clustering
from .pairwise_distance import pairwise_distance
from .biharmonic_coordinates import biharmonic_coordinates
from .harmonic_coordinates import harmonic_coordinates
def spectral_basis_localization(X, T, m,  W=None, order=2, return_clustering_info=False, threshold=0):
    if W is None:
        [W, _E, _B] = skinning_eigenmodes(X, T, m)


    [l, c] = spectral_clustering(W, m)

    
    D = pairwise_distance(c, W)
    cI = D.argmin(axis=1)

    Wh = None
    if order == 1:
        Wh = harmonic_coordinates(X, T, cI)
    elif order == 2:
        Wh = biharmonic_coordinates(X, T, cI)  

    if threshold > 0:
        Wh[np.abs(Wh) < threshold] = 0

    out = (Wh, cI)
    if return_clustering_info:
        out = out + (l, c)
    return out
