import numpy as np
import scipy as sp
def grad(X, F, U):


    t = F.shape[0]  # simplex size
    n = X.shape[0]
    dt = F.shape[1]

    TU =(U[F]).transpose([0, 2, 1])
    if t == 3:
        # triangle mesh!
        H = np.array([[-1.0, -1],
                      [1., 0],
                      [0, 1.0]], dtype=U.dtype)

    # for each triangle, get the three vertex positions dealing with it
    elif F.shape[1] == 4:
        # tet mesh!
        H = np.array([[-1, -1, -1],
                      [1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]], dtype=U.dtype)

    Tx = X[F].transpose([0, 2, 1])
    XH = Tx @ H
    XHi = np.linalg.pinv(XH)
    HXHi = H @ XHi


    tu = TU.reshape(-1, 3)

    grad = TU @ HXHi


    d= X.shape[1]

    # J = np.repeat(np.F[0, :], d)

    Fe = np.repeat(F[:, None, :], 2, axis=1)
    Fe = Fe*2
    Fe[:, 1, :] += 1
    J = Fe.reshape( -1)
    I = np.arange(J.shape[0])
    vals = np.ones(I.shape)
    Pr = sp.sparse.csc_matrix((vals, (I, J)), shape=(d* dt * t, d*n))


    u = HXHi[:, :, 0]
    # add 0 between every column of u
    u = np.repeat(u, 2, axis=1)
    # only keep

    v = HXHi[:, :, 1]


    # grad = torch.squeeze(grad, dim=1)

    # squeze first two dims
    grad = np.squeeze(grad)
    return grad, HXHi

# def grad(V, T):
#     '''
#     Computes the gradient of the energy function with respect to the vertex positions V
#
#     Parameters
#     ----------
#     V : (n, d) numpy array
#         Vertex positions
#     T : (t, 3|4) numpy array
#         Simplex indices
#
#     Returns
#     -------
#     g : (n*d, 1) numpy array
#         Gradient vector
#     '''
#
#     dim = V.shape[1]
