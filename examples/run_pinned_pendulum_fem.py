import numpy as np

from sims.PinnedPendulumState import PinnedPendulumState
from sims.PinnedPendulum import PinnedPendulumParams
from worlds import PinnedPendulumWorld, PinnedPendulumWorldParams


x0 = np.array([[1.0], [0]])
sim_params = PinnedPendulumParams(mu=1e5)
p = PinnedPendulumWorldParams(render=True, init_state=PinnedPendulumState(x0), sim_params=sim_params)
world = PinnedPendulumWorld(p)


for i in range(100):
    world.step()
    pass


