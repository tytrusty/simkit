import numpy as np

from ..State import State

class ElasticMFEMState(State):

    def __init__(self, x, s, l, x_dot):
        self.x = x
        self.s = s
        self.l = l
        self.x_dot = x_dot
        return



    # def mixed(self):
    #     return np.zeros((0, 1))