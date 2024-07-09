import numpy as np

from ..State import State
class ElasticFEMState(State):

    def __init__(self, x, x_dot):
        self.x = x
        self.x_dot = x_dot
        return


