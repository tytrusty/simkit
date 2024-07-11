import numpy as np

from ..State import State

class ElasticROMFEMState(State):

    def __init__(self, z, z_dot):
        self.z = z
        self.z_dot = z_dot
        return


