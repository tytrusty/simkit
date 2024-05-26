import numpy as np
from .WorldState import WorldState
class PinnedPendulumState(WorldState):

    def __init__(self, x=None):
        super().__init__()
        if x is None:
            self.x = np.array([[1], [0]])
        self.x = x
        return

