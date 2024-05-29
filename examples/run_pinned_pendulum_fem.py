import numpy as np

from sims import *
# from sims.PinnedPendulumFEM import PinnedPendulumFEMState

from worlds import PinnedPendulumWorld, PinnedPendulumWorldParams

x0 = np.array([[1.0], [0]])
# sim_params = PinnedPendulumFEMParams(mu=1e5)

sim_params = PinnedPendulumMFEMParams(mu=1e10, eta=1)
p = PinnedPendulumWorldParams(render=True, init_state=PinnedPendulumFEMState(x0), sim_params=sim_params)
world = PinnedPendulumWorld(p)


for i in range(100):
    world.step()
    pass


