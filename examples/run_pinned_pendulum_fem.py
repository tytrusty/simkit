import numpy as np

from simkit.sims import *
from simkit.sims.pinned_pendulum import *
from simkit.worlds import PinnedPendulumWorld, PinnedPendulumWorldParams



init_state = PinnedPendulumState(np.array([[1.0], [0]]))

sim_params = PinnedPendulumSimParams.from_args(mu=1e5, disc='fem')

p = PinnedPendulumWorldParams(render=True, init_state=init_state, sim_params=sim_params) #, init_state=PinnedPendulumFEMState(x0), sim_params=sim_params)
world = PinnedPendulumWorld(p)

for i in range(100):
    world.step()
    pass


