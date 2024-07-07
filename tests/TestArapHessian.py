
import igl
import numpy as np
import scipy as sp
from simkit import arap_hessian
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix
from simkit.polyscope import view_displacement_modes


########## test 2D spectrum ##################
# [X, _, _, T, _, _] = igl.read_obj("./data/2d/beam/beam.obj")
[X, _, _, T, _, _] = igl.read_obj("../data/2d/bingby/bingby.obj")
# [X, T, F] = igl.read_mesh("./data/3d/treefrog/treefrog.mesh")
X = X[:, :2]

n = X.shape[0]


# np.linalg.norm((L/3 - C).todense())
M = massmatrix(X, T)
Mv = sp.sparse.kron(M, sp.sparse.identity(2))

H = arap_hessian(X=X, T=T)
[E, W] = sp.sparse.linalg.eigs(H, 10, M=Mv, sigma=0, which='LM')
# Wc = W.real

Wl = W.real
# view_displacement_modes(X, T, Wl, a=1e5)



################# NOW STretch Tests ####################

from simkit.symmetric_stretch_map import symmetric_stretch_map
from simkit.arap_hessian import arap_hessian_d2S
num_trials = 100

dim = 2
num_t = 1

C, Ci = symmetric_stretch_map(num_t, dim)

# test the stretch part
for i in range(num_trials):
    mu = np.ones((num_t, 1)) # np.abs(np.random.rand(num_t, 1) * 100)
    vol = np.ones((num_t, 1)) # np.abs(np.random.rand(num_t, 1) * 100)

    
    S = np.random.rand(num_t, dim, dim) * 0 + np.identity(dim)
    S = (S  + S.transpose(0, 2, 1 ))/2
    s = (Ci @ S.reshape(-1, 1)).reshape(-1, int(dim * (dim + 1)/ 2))

    S_t = (C @ s.reshape(-1, 1)).reshape(-1, dim, dim)
    H_S = arap_hessian_d2S(S, mu, vol)
    H_s = arap_hessian_d2S(s, mu, vol)
    H = Ci.T @ H_s @ C.T
    err = np.linalg.norm((H_S - H).toarray())
    print(err)
