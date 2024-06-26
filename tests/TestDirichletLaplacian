
import igl
import numpy as np

from simkit import arap_hessian
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix


# test 2D spectrum

# [X, _, _, T, _, _] = igl.read_obj("./data/2d/beam/beam.obj")
[X, _, _, T, _, _] = igl.read_obj("./data/2d/bingby/bingby.obj")
# [X, T, F] = igl.read_mesh("./data/3d/treefrog/treefrog.mesh")
X = X[:, :2]

n = X.shape[0]

# C = igl.cotmatrix(X, T)
L = dirichlet_laplacian(X, T, mu=1)

C = igl.cotmatrix(X, T)

# np.linalg.norm((L/3 - C).todense())
M = massmatrix(X, T)
import scipy as sp

# H = arap_hessian(X.reshape(-1, 1), X, T)
# [E, W] = sp.sparse.linalg.eigs(H, 10, M=sp.sparse.block_diag([M, M]), sigma=0, which='LM')
# Wc = W.real


# L = H[:n, :][:, :n]
[E, W] = sp.sparse.linalg.eigs(L, 10, M=M, sigma=0, which='LM')
Wl = W.real



import polyscope as ps

ps.init()
ps.remove_all_structures()

mesh = ps.register_surface_mesh("mesh", X, T)
for i in range(10):
    mesh.add_scalar_quantity(f"eigenvector_{i}", Wl[:, i])
    # mesh.add_vector_quantity("hessian_eigenvector_" + str(i), Wc[:, i].reshape(-1, 2))
ps.show()