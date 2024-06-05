import igl
import scipy as sp
import numpy as np

def deformation_jacobian(V, T):
    '''
    Linear mapping between positions and deformation gradients, assuming x has been flattened with default order="C":

    Parameters
    ----------
    V : (n, dim) array
        The vertices of the mesh

    T : (t, 3|4) array
        Simplex indices

    Returns
    -------
    J : (d*d*t, d*n) sparse matrix
        The deformation Jacobian matrix

    Example
    -------
    ```python
    x = X.reshape(-1, 1)
    f = J @ x
    F = f.reshape(-1, 3, 3)
    ```

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
        G = G[0:-t, : ]
    elif d == 3:
        G = igl.grad(V, T)

    Ge = sp.sparse.kron(sp.sparse.identity(d), G).tocsc()

    # shuffle rows with permutation matrix
    imat = np.tile(np.arange(0, d*d), (t, 1)) + np.arange(0, t)[:, np.newaxis] * d*d;
    oo = np.arange(d*d) * t
    jmat = np.tile(oo, (t, 1)) + np.arange(0, t)[:, np.newaxis]
    vals = np.ones(imat.shape)
    Pr = sp.sparse.coo_matrix((vals.flatten(), (imat.flatten(), jmat.flatten())), shape=(d*d* t, d*d* t)).tocsc()

    # shuffle columns with permutation matrix
    oo = np.arange(d)
    jmat = np.tile(np.arange(n)[:, None], ( 1, d)) * d + oo
    imat = np.tile(np.arange(n)[:, None], ( 1, d)) + np.tile(oo, (n, 1))*n
    vals = np.ones(imat.shape)
    Pc = sp.sparse.coo_matrix((vals.flatten(), (imat.flatten(), jmat.flatten())), shape=(d * n, d*n)).tocsc()

    J = Pr @ Ge @ Pc

    return J



# Otman : I think the deformation jacobian function should be written explicitly for each element,
# for a better learning opportunity, and for less dependence on libigl.

# '''
# Gradient of triangle or tet mesh . For now, U is scalar
# '''
# def gradient(X, F, U, HXHi=None):
#
#     if (len(U.shape) == 2):
#         U = U[None, :, :]
#     t = F.shape[1]  # simplex size
#     dx = X.shape[1]
#     du = U.shape[2]
#
#     TU =(U[:, F]).transpose([0, 1, 3, 2])
#     assert (U.dtype == X.dtype)
#     if (HXHi is None):
#         if t == 3:
#             # triangle mesh!
#             H = np.array([[-1.0, -1],
#                           [1., 0],
#                           [0, 1.0]], dtype=U.dtype)
#
#         # for each triangle, get the three vertex positions dealing with it
#         elif F.shape[1] == 4:
#             # triangle mesh!
#             H = np.array([[-1, -1, -1],
#                           [1.0, 0, 0],
#                           [0, 1.0, 0],
#                           [0, 0, 1.0]], dtype=U.dtype)
#         else:
#             raise ValueError("Only triangle and tet meshes are supported")
#         Tx = X[F].transpose([0, 2, 1])
#         XH = Tx @ H
#         XHi = np.linalg.pinv(XH)
#         HXHi = H @ XHi
#
#         grad = TU @ HXHi
#         # grad = torch.squeeze(grad, dim=1)
#
#         # squeze first two dims
#         grad = np.squeeze(grad)
#         return grad, HXHi
#     else:
#         grad = TU @ HXHi
#         grad = np.squeeze(grad)
#         return grad, HXHi