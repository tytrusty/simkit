import scipy as sp
import numpy as np

from .lbs_jacobian import lbs_jacobian
from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix



def skinning_eigenmodes(X, T, k, mu=1, bI=None):

    M = massmatrix(X, T)
    L = dirichlet_laplacian(X, T, mu=mu)

    if bI is not None:
        assert(isinstance(bI, np.ndarray))
        Ii = np.setdiff1d(np.arange(X.shape[0]), bI)
        L = L[Ii, :][:, Ii]
        M = sp.sparse.diags(M.diagonal()[Ii,], 0)
        
        [E, Wi] = sp.sparse.linalg.eigs(L, k=k, M=M, which='LM', sigma=0)
        Wi = Wi.real
        E = E.real
        W = np.zeros((X.shape[0], k))
        W[Ii, :] = Wi
        
    else:
        [E, W] = sp.sparse.linalg.eigs(L, k=k, M=M, which='LM', sigma=0)
        

        E = E.real
        W = W.real
    B = lbs_jacobian(X, W)

    return W, E, B