import numpy as np

from ..State import State
class ElasticFEMState(State):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        return

    def primary(self):
        return self.x


    # def mixed(self):
    #     return np.zeros((0, 1))