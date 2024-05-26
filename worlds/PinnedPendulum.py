from .World import World
from .WorldState import WorldState

from .PinnedPendulumState import PinnedPendulumState
from ..solvers import Solver

class PinnedPendulum(World):

    def __init__(self, solver : Solver):
        self.solver = solver

    def step(self, state : PinnedPendulumState = PinnedPendulumState()):


        x  = solver.solve(state.x)
        state = PinnedPendulumState(x)
        return state



