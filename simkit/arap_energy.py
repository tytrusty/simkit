import numpy as np

from .volume import volume
from .deformation_jacobian import deformation_jacobian
from .polar_svd import polar_svd

def arap_energy(x, V, F, mu, pre=None):

    dim = V.shape[1]
    vol = volume(V, F)

    J = deformation_jacobian(V, F)
    f = J @ x

    F = np.reshape(f, (-1, dim, dim))
    [R, S] = polar_svd(F)

    E = 0.5 * np.sum((mu * vol) * np.sum((F - R)**2))

    return E
#
# def arap_energy(V, F, mu=None, U=None):
#     """
#         Computes the ARAP energy evaluated at a given displacement U
#
#         Parameters
#         ----------
#         V : (n, 3) array
#             The input mesh vertices
#         F : (m, 4) array
#             Tet mesh indices
#         mu : float or (m, 1) array
#             first lame parameter, defaults to 1.0
#         U : (n, 3) array
#             The deformed  mesh vertices
#
#         Returns
#         -------
#         E : float
#             The ARAP energy
#     """
#     dim = V.shape[1]
#
#     sim = F.shape[1]
#     if (mu is None):
#         mu = np.ones((F.shape[0]))
#     elif (np.isscalar(mu)):
#         mu = mu* np.ones((F.shape[0]))
#     else:
#         assert(mu.shape[0] == F.shape[0])
#
#     if U is None:
#         U = V.copy() # assume at rest
#
#     assert(V.shape == U.shape)
#     assert(V.shape[0] >= F.max())
#     assert((sim == 4 or sim == 3) and "Only tet or triangle meshes supported")
#
#     if dim == 2:
#         assert(sim == 3)
#     if dim == 3 :
#         assert(sim == 4)
#
#     vol = volume(V, F)
#
#     J = deformation_jacobian(V, F)
#     f = J @ U.reshape(-1, 1)
#
#     F = np.reshape(f, (-1, dim, dim))
#     [R, S] = polar_svd(F)
#
#     E = 0.5 * np.sum((mu * vol) * np.sum((F - R)**2))
#
#     return E
#
#
