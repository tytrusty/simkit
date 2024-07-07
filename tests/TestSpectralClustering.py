
import igl
import numpy as np
import scipy as sp
from simkit import arap_hessian
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix
from simkit.average_onto_simplex import average_onto_simplex
from simkit.spectral_clustering import spectral_clustering

from simkit.polyscope import view_displacement_modes


########## test 2D spectrum ##################
# [X, _, _, T, _, _] = igl.read_obj("./data/2d/beam/beam.obj")
[X, _, _, T, _, _] = igl.read_obj("../data/2d/bingby/bingby.obj")
# [X, T, F] = igl.read_mesh("./data/3d/treefrog/treefrog.mesh")
X = X[:, :2]

n = X.shape[0]


# np.linalg.norm((L/3 - C).todense())
M = massmatrix(X, T)

H = dirichlet_laplacian(X=X, T=T)
[E, W] = sp.sparse.linalg.eigs(H, 20, M=M, sigma=0, which='LM')

# Wc = W.real

Wl = W.real

Wt = average_onto_simplex(Wl, T)

import polyscope as ps
ps.init()
ps.set_ground_plane_mode("none")
mesh = ps.register_surface_mesh("mesh", X, T, edge_width=1)

l, c = spectral_clustering(Wt, 10)
mesh.add_scalar_quantity("W", l, defined_on='faces', cmap='rainbow', enabled=True)

ps.show()


