import scipy as sp
import numpy as np

from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix


def biharmonic_coordinates(X, T, bI):

    L = dirichlet_laplacian(X, T)
    M = massmatrix(X, T)
    Mi = sp.sparse.diags(1/M.diagonal())
    
    bI = np.unique(bI)
    aI = np.setdiff1d(np.arange(X.shape[0]), bI)

    Q = L.T @ Mi @ L

    bc = np.identity(bI.shape[0])

    Qii = Q[aI, :][:, aI]
    Qbi = Q[aI, :][:, bI]

    xii = sp.sparse.linalg.spsolve(Qii, -Qbi @ bc)
    if xii.ndim == 1:
        xii = xii.reshape(-1, 1)
    x = np.zeros((X.shape[0], bI.shape[0]))

    x[aI, :] = xii
    x[bI, :] = bc

    return x