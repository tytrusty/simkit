import igl
import scipy as sp
import numpy as np

from .deformation_jacobian import deformation_jacobian
from .rotation_gradient import rotation_gradient_F
from .volume import volume

def arap_hessian_F(F, mu=1, vol=1):

    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)

    n = F.shape[0]
    I = np.tile(np.identity(dim * dim), (n, 1, 1))

    H = 2 * I - 2 *  rotation_gradient_F(F)
    d = mu * vol
    H *= d.reshape(-1, 1, 1)
    return H

def arap_hessian(x, V, T, mu=1, pre=None):
    dim = V.shape[1]
    J = deformation_jacobian(V, T)
    f = J @ x

    vol = volume(V, T)
    F = np.reshape(f, (-1, dim, dim))
    d2psidF2 = arap_hessian_F(F, mu=mu, vol=vol)
    H = sp.sparse.block_diag(d2psidF2)  # block diagonal hessian matrix

    Q = J.transpose() @ H @ J

    return Q
# def arap_hessian(V, T, mu=None, U=None):
#     """
#
#         Computes the ARAP Hessian matrix at a given displacement U
#
#     """
#     dim = V.shape[1]
#     if (mu is None):
#         mu = np.ones((T.shape[0]))
#     elif (np.isscalar(mu)):
#         mu = mu* np.ones((T.shape[0]))
#     else:
#         assert(mu.shape[0] == T.shape[0])
#
#     if U is None:
#         U = V.copy() # assume at rest
#
#     if dim == 2:
#         vol = igl.doublearea(V, T).reshape(-1)/2
#     elif dim == 3:
#         vol = igl.volume(V, T).reshape(-1)
#     else:
#         ValueError("Only V.shape[1] == 2 or 3 are supported")
#
#     J = deformation_jacobian(V, T)
#     f = J @ U.flatten()
#
#     F = np.reshape(f, (-1, dim, dim))
#     d2psidF2 = d2arap_dF2(F, mu=mu, a=vol)
#     H = sp.sparse.block_diag(d2psidF2) # block diagonal hessian matrix
#
#
#     Q = J.transpose() @ H @ J
#     return Q
#
#
