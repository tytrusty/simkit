

import numpy as np
import scipy as sp

from .polar_svd import polar_svd
from .rotation_gradient import rotation_gradient_F
from .deformation_jacobian import deformation_jacobian

def stretch(F):
    [R, S] = polar_svd(F)
    s = S.reshape(-1, 1)
    return s


# def stretch_from_x(x, J, dim):
#     F = (J @ x).reshape(-1, dim, dim)
#     s = stretch_F(F)
#     return s


# def stretch(X, T, U, J=None):
#     if J is None:
#         J = deformation_jacobian(X, T)

#     x = U.reshape(-1, 1)
#     dim = X.shape[1]

#     S = stretch_from_x(x, J, dim).reshape(-1, dim, dim)
#     return S