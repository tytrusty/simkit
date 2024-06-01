import polyscope as ps
import numpy as np

from .World import WorldParams

from ..sims.pinned_pendulum import *

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
    def __init__(self, render=False, init_state : PinnedPendulumState = PinnedPendulumState(), sim_params : PinnedPendulumSimParams =PinnedPendulumSimParams()):
        self.render = render
        self.sim_params = sim_params
        self.init_state = init_state



        if type(self.sim_params.sub_p) ==  PinnedPendulumMFEMSimParams:
            if type(self.init_state.sub_state) ==  PinnedPendulumFEMState:
                  self.init_state = PinnedPendulumState(PinnedPendulumMFEMState(np.concatenate((self.init_state.sub_state.x, [[np.linalg.norm(self.init_state.sub_state.x)], [0]]), axis=0)))

        if type(self.sim_params.sub_p) == PinnedPendulumFEMSimParams:
            if type(self.init_state.sub_state) ==  PinnedPendulumMFEMState:
                self.init_state = PinnedPendulumState(PinnedPendulumFEMState(self.init_state.sub_state.p[:2]))

        return


class PinnedPendulumWorld():
    def __init__(self, p : PinnedPendulumWorldParams = PinnedPendulumWorldParams()):
        self.p = p

        self.sim = PinnedPendulumSim(self.p.sim_params)

        if self.p.render:
            self.renderer = PinnedPendulumWorldRenderer(self.p.init_state.position(), self.p.sim_params.sub_p.y)
        self.reset()

    def step(self):
        self.sim_state = self.sim.step_sim(self.sim_state)

        if self.p.render:
            self.renderer.render(self.sim_state.position())

        return

    def reset(self):

        self.sim_state = self.p.init_state
        if self.p.render:
            self.renderer.render(self.sim_state.position())

        pass



