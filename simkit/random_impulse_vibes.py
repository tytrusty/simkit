import numpy as np
import scipy as sp


from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix
from .farthest_point_sampling import farthest_point_sampling
from .skinning_eigenmodes import skinning_eigenmodes
from .spectral_clustering import spectral_clustering
from .pairwise_distance import pairwise_distance

def random_impulse_vibes(X, T, m, h=1e-2, ord=1):
    L = dirichlet_laplacian(X, T)
    M = massmatrix(X,  T)
    H = L + M / h**2

    Mi = sp.sparse.diags(1 / M.diagonal()).tocsc().toarray()
    
    [W, _E, _B] = skinning_eigenmodes(X, T, m)
    [l, c] = spectral_clustering(W, m)
    D = pairwise_distance(c, W)
    cI = D.argmin(axis=1)


    
    F = Mi[:, cI]
    G_full= sp.sparse.linalg.spsolve(H, F ).reshape(F.shape)
    G_full = G_full / G_full.max(axis=1)[:, None]
    G = G_full.copy()

    G[G < 1e-8] = 1e-8
    G = G**ord / (G**ord).sum(axis=1)[:, None]

    

    return G, cI, G_full