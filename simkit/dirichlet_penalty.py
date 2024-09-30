import scipy as sp
import numpy as np

def dirichlet_penalty(bI, y, nv,  gamma, only_b=False, SGamma=None, return_SGamma=False):
    """
    Determines a quadratic pinning penalty objective to hold vertex indices bI of mesh V, T
    fixed in place.

    The pinning penalty is given by:
    E = 1/2 * || Sx - y||_Gamma^2
    E = 1/2 x^T S^T Gamma S x - x^T S^T Gamma S y + const

    Parameters
    ----------
    bI : (cn, 1) numpy array
        Indices of vertices to pin
    bc : (cn, d) numpy array
        Positions of vertices to pin
    nv : int
        Number of vertices in the mesh

    gamma : float or (cn, 1) numpy array
        Attractive force constant

    Returns
    -------
    Q : (n*d, n*d) sparse matrix
        Quadratic term matrix
    b : (n*d, 1) numpy array
        Linear term vector
    """

    assert(y.ndim == 2)
    if SGamma is None:
        nc = bI.shape[0]
        S = sp.sparse.csc_matrix((  np.ones(nc), (bI, np.arange(nc))) , (nv, nc) )
        d = y.shape[1]

        S = sp.sparse.kron(S, sp.sparse.eye(d))
        cn = bI.shape[0]

        if isinstance(gamma, float) or isinstance(gamma, int):
            gamma = np.ones(cn) * gamma


        Gamma = sp.sparse.diags(gamma).tocsc()
        Gamma = sp.sparse.kron(Gamma, sp.sparse.identity(d))

        SGamma = S @ Gamma



    b = -SGamma @ y.reshape(-1, 1)
   
    if only_b:
        out = (b, )
    else:
        Q = SGamma @ S.T
        out = (Q, b)
    
    if return_SGamma:
        out = out + (SGamma,)
    return out 





