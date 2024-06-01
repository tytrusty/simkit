import numpy as np


from ..State import State
class PinnedPendulumFEMState(State):

    def __init__(self, x=None):
        super().__init__()
        if x is None:
            self.x = np.array([[0], [-1.0]])
        else:
            self.x = x
        return

    def primary(self):
        return self.x

    def mixed(self):
        return np.zeros((0, 1))