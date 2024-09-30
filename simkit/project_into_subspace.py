import scipy as sp
import numpy as np 



def project_into_subspace(y, B, M=None, BMB=None, BMy=None):
    """
    0.5 || Bz - y ||^2_M

    """
    if M is None:
        M = sp.sparse.identity(y.shape[0])

    if BMy is None:
        BMy = B.T @ M @ y

    if BMB is None:
        BMB = B.T @ M @ B


    if sp.sparse.issparse(BMB):
        z = sp.sparse.linalg.spsolve(BMB, BMy).reshape(-1, 1)
    else:
        z = np.linalg.solve(BMB, BMy).reshape(-1, 1)

    return z