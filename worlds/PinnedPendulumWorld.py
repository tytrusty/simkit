import polyscope as ps
import numpy as np

from .World import WorldParams

from sims import *

class PinnedPendulumWorldRenderer():
    def __init__(self, x: np.ndarray, y : np.ndarray):
        ps.init()
        self.X = np.concatenate((np.array([[0, 0]]), x.reshape(-1, 2)), axis=0)
        self.Y = np.concatenate((np.array([[0, 0]]), y.reshape(-1, 2)), axis=0)
        ps.register_curve_network("pendulum", self.X, np.array([[0, 1]]), enabled=True, radius=0.03, color=[0, 0, 1])
        ps.register_point_cloud("points", self.X, radius=0.1, color=[0, 0, 1])
        ps.register_curve_network("pendulum_target", self.Y, np.array([[0, 1]]), enabled=True, radius=0.03, color=[1, 0, 0])
        ps.register_point_cloud("points_target", self.Y, radius=0.1, color=[1, 0, 0])
        ps.set_ground_plane_mode("none")
        ps.look_at([0, -0.5, 3], [0, -0.5, 0])
        return

    def render(self, x):
        self.X[1, :] = x.reshape(-1, 2)
        ps.get_curve_network("pendulum").update_node_positions(self.X)
        ps.get_point_cloud("points").update_point_positions(self.X)
        ps.frame_tick()
        return


class PinnedPendulumWorldParams(WorldParams):
    def __init__(self, render=False, init_state : PinnedPendulumFEMState | PinnedPendulumMFEMState = PinnedPendulumFEMState(), sim_params : PinnedPendulumFEMParams =PinnedPendulumFEMParams()):
        self.render = render
        self.sim_params = sim_params
        self.init_state = init_state

        if isinstance(self.sim_params, PinnedPendulumMFEMParams):
            if (isinstance(self.init_state, PinnedPendulumFEMState)):
                print("Warning: init_state is of type PinnedPendulumFEMState, but sim_params is of type PinnedPendulumMFEMParams. This may cause issues. Doing automatic conversion for now")
                p = np.concatenate((self.init_state.x, np.array([[1], [0]])), axis=0)
                self.init_state = PinnedPendulumMFEMState(p)
        return


class PinnedPendulumWorld():
    def __init__(self, p : PinnedPendulumWorldParams = PinnedPendulumWorldParams()):
        self.p = p


        if isinstance(self.p.sim_params, PinnedPendulumMFEMParams):
            self.sim = PinnedPendulumMFEMSim(self.p.sim_params)
        elif isinstance(self.p.sim_params, PinnedPendulumFEMParams):
            self.sim = PinnedPendulumFEMSim(self.p.sim_params)
        else:
            assert(False, "Error: sim_params of type " + str(type(self.p.sim_params)) +
                          " is not a valid instance of PinnedPendulumFEMParams or PinnedPendulumMFEMParams. Exiting.")

        if self.p.render:
            self.renderer = PinnedPendulumWorldRenderer(self.p.init_state.primary(), self.p.sim_params.y)
        self.reset()

    def step(self):
        self.sim_state = self.sim.step_sim(self.sim_state)

        if self.p.render:
            self.renderer.render(self.sim_state.primary())

        return

    def reset(self):

        self.sim_state = self.p.init_state
        if self.p.render:
            self.renderer.render(self.sim_state.primary())

        pass



