import scipy as sp
import numpy as np

from .deformation_jacobian import deformation_jacobian
from .volume import volume

def dirichlet_laplacian(X, T, mu=1, vector=False):
    vol = volume(X, T)

    a = vol * mu

    dim = X.shape[1]
    n = X.shape[0]
    ae = np.repeat(a, dim*dim )
    A = sp.sparse.diags(ae)
    J = deformation_jacobian(X, T)

    H = J.T @ A @ J
    L = H 

    if not vector:
        L = sp.sparse.csc_matrix((n, n))
        for i in range(dim):
            Ii = np.arange(n)*dim + i
            L = L + H[Ii, :][:, Ii]
        L = L / dim
    return L
