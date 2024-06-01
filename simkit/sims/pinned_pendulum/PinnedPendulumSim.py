
import numpy as np

from ..Sim import Sim

from .PinnedPendulumFEMSim import  PinnedPendulumFEMSimParams, PinnedPendulumFEMSim
from .PinnedPendulumMFEMSim import  PinnedPendulumMFEMSimParams, PinnedPendulumMFEMSim

from .PinnedPendulumState import PinnedPendulumState

from ...solvers import NewtonSolverParams
class PinnedPendulumSimParams():
    def __init__(self, p :  PinnedPendulumFEMSimParams | PinnedPendulumMFEMSimParams = PinnedPendulumFEMSimParams()):

        self.sub_p = p

        return



    @classmethod
    def from_args(self, disc='fem', m=1, l0=1, mu=1, g=0, gamma=1, y=np.array([[0], [-1]]),
                 solver_p : NewtonSolverParams = NewtonSolverParams()):
        if disc == 'fem':
            return PinnedPendulumSimParams(PinnedPendulumFEMSimParams(m=m, l0=l0, mu=mu, g=g, gamma=gamma, y=y, solver_p=solver_p))
        elif disc == 'mfem':
            return PinnedPendulumSimParams(PinnedPendulumMFEMSimParams(m=m, l0=l0, mu=mu, g=g, gamma=gamma, y=y, solver_p=solver_p))
class PinnedPendulumSim():

    def __init__(self, p : PinnedPendulumSimParams):
        self.sub_sim = p


        if isinstance(self.sub_sim.sub_p, PinnedPendulumMFEMSimParams):
            self.sim = PinnedPendulumMFEMSim(self.sub_sim.sub_p)
        elif isinstance(self.sub_sim.sub_p, PinnedPendulumFEMSimParams):
            self.sim = PinnedPendulumFEMSim(self.sub_sim.sub_p)
        else:
            assert(False, "Error: sim_params of type " + str(type(self.sub_sim.sub_p)) +
                          " is not a valid instance of PinnedPendulumFEMSimParams or PinnedPendulumMFEMSimParams. Exiting.")
        return

    def energy(self, state : np.ndarray):
        return self.sim.energy(state)

    def gradient(self, state : np.ndarray):
        return self.sim.gradient(state)

    def hessian(self, state : np.ndarray):
        return self.sim.hessian(state)

    def step(self, x : np.ndarray):
        return self.sim.step(x)

    def step_sim(self, st : PinnedPendulumState):


        return PinnedPendulumState(self.sim.step_sim(st.sub_state))


