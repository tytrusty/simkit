import numpy as np
from .SimState import SimState
class PinnedPendulumState(SimState):

    def __init__(self, x=None):
        super().__init__()
        if x is None:
            self.x = np.array([[0], [-1.0]])
        else:
            self.x = x
        return

