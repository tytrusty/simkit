import numpy as np
import os

from simkit.sims import *
from simkit.sims.pinned_pendulum.PinnedPendulumMFEMSim import *
from simkit.sims.pinned_pendulum.PinnedPendulumFEMSim import *
import matplotlib.pyplot as plt
from simkit.matplotlib.PointCloud import PointCloud
from simkit.matplotlib.Curve import Curve
from simkit.matplotlib.colors import *
dir = os.path.dirname(__file__)

class Experiment:

    def __init__():
        pass

    def render(self, X):

        x0 = np.vstack([[0, 0], X[:, 0].T])
        E = np.array([[0, 1]])
        
        y0 = np.array([[0, 0], [0, -1]])

        ref = Curve(y0, E, color=light_red, linewidth=4)
        pc_ref = PointCloud(y0, size=200, color=light_red, edgecolor='black', linewidth=0.1)
        
        cu = Curve(x0,E, color=light_blue, linewidth=4)
        pc = PointCloud(x0, size=200, color=light_blue, edgecolor='black', linewidth=0.1)

        for i in range(X.shape[1]):
            xi = np.vstack([[0, 0], X[:, i].T])

            pc.update_vertex_positions(xi)
            cu.update_vertex_positions(xi)
            plt.pause(0.001)
        return


class FEMExperiment(Experiment):
    def __init__(self, mu=1):
        fem_sim_params = PinnedPendulumFEMSimParams(mu=mu)
        fem_sim_params.solver_p.max_iter = 100
        fem_sim = PinnedPendulumFEMSim(fem_sim_params)
        self.sim = fem_sim
    
    def simulate(self, num_timesteps=500):
        x = self.sim.rest_state()
        Xs = np.zeros((2, num_timesteps + 1))

        Xs[:, 0] = x.flatten()
        for i in range(num_timesteps):
            x = self.sim.step(x)
            Xs[:, i+1] = x.flatten()
        return Xs
    
class MFEMExperiment(Experiment):
    def __init__(self, mu=1):
        mfem_sim_params = PinnedPendulumMFEMSimParams(mu=mu)
        mfem_sim_params.solver_p.max_iter = 100
        mfem_sim = PinnedPendulumMFEMSim(mfem_sim_params)
        self.sim = mfem_sim

    def simulate(self, num_timesteps=500):
        [x, s, l] = self.sim.rest_state()
        Xs = np.zeros((2, num_timesteps +1))

        for i in range(num_timesteps):
            [x, s, l] = self.sim.step(x, s, l)
            Xs[:, i+1] = x.flatten()
        return Xs
    

results_dir = os.path.join(dir, 'results')

exp = FEMExperiment(mu=1e12)
X = exp.simulate()

plt.ion()
plt.figure()
plt.axis('equal')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.axis('off')
exp.render(X)

# mfem_sim_params = PinnedPendulumMFEMSimParams(mu=1e12)
# mfem_sim = PinnedPendulumMFEMSim(mfem_sim_params)
# [x, s, l] = mfem_sim.rest_state()
# for i in range(100):
#     [x, s, l] = mfem_sim.step(x, s, l)
#     print(x)

