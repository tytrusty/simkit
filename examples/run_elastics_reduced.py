
from turtle import st
import numpy as np
import igl
import polyscope as ps
import scipy as sp
from simkit import deformation_jacobian, massmatrix, volume
from simkit.pairwise_displacement import pairwise_displacement
from simkit.polyscope import view_displacement_modes
from simkit.sims.elastic import *
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.grad import grad
from simkit.gravity_force import gravity_force
from simkit.sims.elastic import ElasticMFEMSim, ElasticMFEMState, ElasticROMFEMSim, ElasticROMFEMState, ElasticROMFEMSimParams
from simkit.stretch import stretch
from simkit.symmetric_stretch_map import symmetric_stretch_map
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.spectral_cubature import spectral_cubature
from simkit.average_onto_simplex import average_onto_simplex
from simkit.polyscope.view_clusters import view_clusters
from simkit.polyscope.view_cubature import view_cubature
from simkit.pairwise_distance import pairwise_distance
from simkit.volume import volume
from simkit.project_into_subspace import project_into_subspace


[X, _, _, T, _, _] = igl.read_obj("../data/2d/cthulu/cthulu.obj")

X = X[:, 0:2]
X = X / max(X.max(axis=0) - X.min(axis=0))

dim = X.shape[1]

m = 20
k = 100
[W, E,  B] = skinning_eigenmodes(X, T, m)
[cI, cW, labels] = spectral_cubature(X, T, W, k, return_labels=True)


# view_displacement_modes(X, T, B, a=1)
# view_cubature(X, T, cI, cW, labels)

A = np.zeros(((W.shape[1], dim, dim+1)))
A[:, :dim, :dim] = np.identity(dim)
z = project_into_subspace(X.reshape(-1, 1), B)


# s = np.ones((k, 3))
# s[:, -1] = 0
# l = np.zeros(s.shape)
z_dot = np.zeros(z.shape)


rho = 1e3
bg =  - gravity_force(X, T, rho=rho).reshape(-1, 1) # negative sign of force we desire bc the b's are gonna be in rhs of newton's method
bI =  np.where(X[:, 0] < 0.001 + X[:, 0].min())[0]
bc0 = (X[bI, :])
# [Qp, bp] = dirichlet_penalty(bI, bc, X.shape[0],  1e7)

sim_params = ElasticROMFEMSimParams()
# sim_params = ElasticMFEMSimParams()

# sim_params.b = B.T @ ( bp + bg)
# sim_params.Q = B.T @ Qp @ B


sim_params.ym = 5e5
sim_params.h = 1e-2
sim_params.rho = rho
sim_params.solver_p.max_iter= 1
sim_params.solver_p.do_line_search = True #True

sim = ElasticROMFEMSim(X, T,B, cI=cI, cW=cW, p=sim_params)
# sim = ElasticMFEMSim(X, T, sim_params)

period = 100

ps.init()
ps.set_ground_plane_mode("none")
mesh = ps.register_surface_mesh("mesh", X, T, edge_width=1)
for i in range(1000):
    
    bc = bc0 + np.sin( 2 * np.pi * i / (period)) * np.array([[1, 0]])
    [Q_ext, b_ext] = dirichlet_penalty(bI, bc, X.shape[0],  1e7)

    BQB_ext = sim.B.T @ Q_ext @ sim.B
    Bb_ext = sim.B.T @ b_ext
    z_next = sim.step(z,  z_dot, BQB_ext, Bb_ext)
    z_dot = (z_next - z) / sim_params.h    
    z = z_next.copy()

    # x_next, s_next, l_next = sim.step(x, s, l,  x_dot)
    # x_dot = (x_next - x) / sim_params.h   
    # x = x_next.copy()
    # s = s_next.copy()
    # l = l_next.copy()
    x = B @ z
    mesh.update_vertex_positions(x.reshape(-1, 2))
    ps.frame_tick()




# What Ty needs :
# Energy per-stencil
