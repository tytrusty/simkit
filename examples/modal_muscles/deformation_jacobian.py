import igl
import scipy as sp
import numpy as np


def deformation_jacobian(V, T):
    '''
    Linear mapping between positions and deformation gradients, assuming x has been flattened with order="F":

    Examples:
        x = X.reshape(-1, 1, order="F")
        f = J @ x
        F = f.reshape(-1, 3, 3)

    In the above, F will have the form:
    F = [[Fxx Fxy Fxz]]
        [[Fyx Fyy Fyz]]
        [[Fzx Fzy Fzz]]

    '''

    t = T.shape[0]
    n = V.shape[0]
    d = V.shape[1]

    if d == 2:
        V2 = np.hstack((V, np.zeros((V.shape[0], 1))))
        G = igl.grad(V2, T)
        G = G[0:-t, :]
        Ge = sp.sparse.block_diag((G, G))
        imat = np.tile(np.arange(0, 4), (t, 1)) + np.arange(0, t)[:, np.newaxis] * 4;
        jmat = np.tile(np.array([0, 1 * t, 2 * t, 3 * t]), (t, 1)) + np.arange(0, t)[                                                                                  :, np.newaxis]
        vals = np.ones(imat.shape)
    elif d == 3:
        G = igl.grad(V, T)
        Ge = sp.sparse.block_diag((G, G, G))
        imat = np.tile(np.arange(0, 9), (t, 1)) + np.arange(0, t)[:, np.newaxis] * 9;
        jmat = np.tile(np.array([0,  1 * t, 2 * t, 3 * t, 4 * t, 5 * t, 6 * t,  7 * t,  8 * t]), (t, 1)) + np.arange(0, t)[:, np.newaxis]
        vals = np.ones(imat.shape)
    else:
        raise ValueError("Only 2D and 3D supported")

    P = sp.sparse.coo_matrix((vals.flatten(), (imat.flatten(), jmat.flatten())), shape=(d*d* t, d*d * t)).tocsc()
    J = P @ Ge

    # f = J@V.flatten(order="F")
    # F = f.reshape((t, 9))

    return J


