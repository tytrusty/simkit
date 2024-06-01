import numpy as np

from ..State import State


class PinnedPendulumMFEMState(State):

    def __init__(self, p=None):
        super().__init__()
        if p is None:
            self.p = np.array([[1], [0.0], [1], [0] ])
        else:
            self.p = p
        return

    def primary(self):
        return self.p[:2]

    def mixed(self):
        return self.p[2]
