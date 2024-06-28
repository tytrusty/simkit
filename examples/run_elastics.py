import numpy as np
import igl
import polyscope as ps
import scipy as sp
from simkit import massmatrix
from simkit.sims.elastic import *
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.grad import grad


sim_params = ElasticFEMSimParams()

[X, _, _, T, _, _] = igl.read_obj("../data/2d/beam/beam.obj")

# [X, _, _, T, _, _] = igl.read_obj("../data/2d/bingby/bingby.obj")
print(X.shape[0])

X = X[:, 0:2]
X = X / max(X.max(axis=0) - X.min(axis=0))

x = X.reshape(-1, 1)
x_dot = np.zeros(x.shape)

g = np.zeros((X.shape[0], 2))
g[:, 1] = 10
M = massmatrix(X, T,  1e3)
bg =  (M @ g).reshape(-1, 1)

bI =  np.where(X[:, 0] < 0.1 + X[:, 0].min())[0]
bc = (X[bI, :])
[Qp, bp] = dirichlet_penalty(bI, bc, X.shape[0],  1e7)

sim_params.b = bp + bg
sim_params.Q = Qp

sim_params.ym = 1e6
sim_params.h = 1e-2
sim_params.rho = 1e3

sim_params.solver_p.max_iter= 1
sim_params.solver_p.do_line_search = False #True
sim = ElasticFEMSim(X, T, sim_params)


# sim_state = init_state
ps.init()
# ps.look_at([0, 0, 2], [0, 0, 0])
ps.set_ground_plane_mode("none")
mesh = ps.register_surface_mesh("mesh", X, T, edge_width=1)
# ps.show()

for i in range(1000):
    x_next = sim.step(x, x_dot)

    x_dot = (x_next - x) / sim_params.h
    x = x_next.copy()
    mesh.update_vertex_positions(x.reshape(-1, 2))

    ps.frame_tick()
#
# p = Elastic2DWorldParams(render=True, init_state=init_state, sim_params=sim_params)
# world = Elastic2DWorld(p)
#
# for i in range(100):
#     world.step()
#     pass


