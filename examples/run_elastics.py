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

X = X[:, 0:2]


x = X.reshape(-1, 1)
x_dot = np.zeros(x.shape)

# init_state = ElasticFEMState(x, x)

g = np.zeros((X.shape[0], 2))
g[:, 1] = 10
M = massmatrix(X, T,  1e3)
bg =  (M @ g).reshape(-1, 1)

bI = np.where(X[:, 0] == X[:, 0].min())[0]
bc = (X[bI, :])
[Qp, bp] = dirichlet_penalty(bI, bc, X.shape[0],  10000)


sim_params.b = bp + bg
sim_params.Q = Qp

sim_params.ym = 1e4
sim_params.h = 1e-2
sim_params.rho = 1e3

sim_params.solver_p.max_iter=10
sim_params.solver_p.do_line_search = False
sim = ElasticFEMSim(X, T, sim_params)

# sim_state = init_state
ps.init()
ps.look_at([0, 0, 20], [0, 0, 0])
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


