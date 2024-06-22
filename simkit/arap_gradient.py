# import igl
import scipy as sp
import numpy as np

from .volume import volume
from .deformation_jacobian import deformation_jacobian
from .polar_svd import polar_svd




def arap_gradient_F(F, mu=1, vol=1):
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    [R, S] = polar_svd(F)
    PK1 = F - R
    d = mu * vol
    PK1 *= d.reshape(-1, 1, 1)
    return PK1
def arap_gradient(x, V, T, mu, pre=None):

    dim = V.shape[1]
    vol = volume(V, T)
    J = deformation_jacobian(V, T)
    f = J @ x
    F = np.reshape(f, (-1, dim, dim))
    dpsi_dF = arap_gradient_F(F, mu=mu, vol=vol)
    pk1 = dpsi_dF.reshape(-1, 1)
    g = J.transpose() @ pk1
    return g
#
# def arap_gradient(V, T, mu=None, U=None):
#     """
#
#         Computes the ARAP Gradient Vector at a given displacement U
#
#     """
#     dim = V.shape[1]
#     # B = deformation_jacobian(V, F);
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
#     F = np.reshape(f, (-1, dim, dim))
#     dpsi_dF = darap_dF(F, mu=mu, a=vol)
#     pk1 = dpsi_dF.reshape(-1, 1)
#     g = J.transpose() @ pk1
#     return g
#
#
