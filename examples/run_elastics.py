import numpy as np
import igl
import polyscope as ps
from simkit.sims.elastic import *


init_state = ElasticFEMState(np.array([[1.0], [0]]))

sim_params = ElasticFEMSimParams()

[X, _, _, T, _, _] = igl.read_triangle_mesh("../data/2d/beam/beam.obj")


sim = ElasticFEMSim(X, T, sim_params)



ps.init()
ps.register_surface_mesh("mesh", X, F)
ps.show()

#
# p = Elastic2DWorldParams(render=True, init_state=init_state, sim_params=sim_params)
# world = Elastic2DWorld(p)
#
# for i in range(100):
#     world.step()
#     pass


