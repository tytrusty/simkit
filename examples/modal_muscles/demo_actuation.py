import os
import igl
import gpytoolbox as gpy
import matplotlib.pyplot as plt

from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.gravity_force import gravity_force
from simkit.matplotlib.Curve import Curve
from simkit.matplotlib.PointCloud import PointCloud
from simkit.matplotlib.TriangleMesh import TriangleMesh
from simkit.matplotlib.colors import *
from simkit.orthonormalize import orthonormalize
from simkit.polyscope.view_displacement_modes import view_displacement_modes
from simkit.polyscope.view_scalar_modes import view_scalar_modes
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.modal_analysis import modal_analysis
from simkit.spectral_cubature import spectral_cubature
from simkit.normalize_and_center import normalize_and_center
from simkit.sims.elastic.ModalMuscleSim import *


from ModalMuscleTestSim import ModalMuscleTestSim, ModalMuscleTestSimParams
dir = os.path.join(os.path.dirname(__file__))

m = 10
nc = 30
k = 10
modeset = [3, 4, 5]
num_timesteps = 1000

# [X, T] = gpy.regular_square_mesh(10, 10)
[X, _, _, T, _, _] = igl.read_obj(dir + "/../../data/2d/rectangle2D/rectangle.obj")
X = X[:, :2]
# X[:, 1] *= 0.5
X = normalize_and_center(X)
dim = X.shape[1]


[W, _E, B] = skinning_eigenmodes(X, T, m)
B = orthonormalize(B, M=sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim)))

[_E, D] = modal_analysis(X, T, max(modeset) + 1)    
D = D[:, modeset]
D = np.hstack([D, X.reshape(-1, 1)])

[_cI, _cW, l] = spectral_cubature(X, T, W, k, return_labels=True)
d = np.zeros(T.shape[0])

g = B.T @ gravity_force(X, T, a=-9.8, rho=1e3).reshape(-1, 1)


fI = np.unique(igl.boundary_facets(T))
cfI = farthest_point_sampling(X[fI, :], nc)
cI = fI[cfI]
# cI = np.where( X[:, 1] < (X[:, 1].min() + 0.01))[0]

sim_params = ModalMuscleSimParams(mu=1e5,gamma=1e5, rho=1e3, alpha=1.0, contact=True)
sim_params.solver_p.max_iter = 10

sim = ModalMuscleSim(X, T, B, D, l, d, cI=cI, params=sim_params, plane_pos=np.array([[0, -2]]).T)
z, z_dot, a = sim.rest_state()

# sim_params_test = ModalMuscleTestSimParams(mu=1e5, rho=1e3, g=-10, dt=sim_params.h, iters=1, ground_contact=True, threshold=1e-6, ground_height=-2)
# # cI = np.arange(X.shape[0])
# sim_test = ModalMuscleTestSim(X, T, B, l, cI, p= sim_params_test)
# z_test, z_dot_test = sim_test.rest_state()

v = np.zeros(X.shape)
v[:, 0] = 10
z_dot = project_into_subspace(v.reshape(-1, 1), B, M= sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim)))
# z_test0 = z_test.copy()
# Y = X + np.array([0, -1])
# z_test = project_into_subspace((Y - X).reshape(-1, 1), B, M= sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim)))

# z_test += np.random.randn(*z_test.shape) * 0.1


# Xr = X.copy()
# # Yr = X.copy()
# Xr[:, 0] = -X[:, 1]
# Xr[:, 1] = X[:, 0]
# z = project_into_subspace(Xr.reshape(-1,1), B, M= sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim)))
# z += 0.0 * np.random.randn(*z.shape)

Mv = sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim))
BMD = B.T @ Mv @ D 


amp = 0.2
period = 50

plt.ion()
[fig, ax] = plt.subplots(dpi=100, figsize=(10, 10))
plt.axis('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xticks([-5, -1.5, 2])
ax.set_yticks([-2, 0, 2])

mesh = TriangleMesh(X, T, edgecolors=gray, linewidths=0.1,  outlinewidth=2)
pc = PointCloud(X[cI, :], size=10)

# offset = np.array([-3.0, 0])[None, :]
# actuated_mesh = TriangleMesh(X + offset, T, facecolors=light_red , edgecolors=gray,  linewidths=0.1, outlinewidth=2)

# import polyscope as ps

# ps.init()
# mesh = ps.register_surface_mesh("mesh", X, T)
# import polyscope as ps
# ps.init()
# mesh = ps.register_surface_mesh('mesh2',  (sim_test.B @ z_test).reshape(-1, 2) + sim.X, sim_test.T)
# mesh = ps.register_surface_mesh('rest',  (sim_test.B @ z_test0).reshape(-1, 2) + sim.X, sim_test.T, color=[1, 0, 0])

# mesh.add_vector_quantity('y', (sim_test.B @ z_dot_test).reshape(-1, 2))
# ps.show()
# ps.set_ground_plane_mode("none")
plt.axline((0, -2), slope=0, color='black', linewidth=4, zorder=-1)
rv = (np.random.randn(a.shape[0] - 1, 1)-0.5)*2.0
for i in range(num_timesteps):

    a[:-1] = amp * rv * (np.sin(2 * np.pi * (i / period)) > 0)
    # v =a.copy()
    # v[-1] = 0
    f = np.zeros((B.shape[1], 1))# -BMD @ v * 1e9
    z_next = sim.step(z, z_dot, a, b_ext= f - g)
    z_dot = (z_next - z) / sim_params.h
    z = z_next
    U = (B @ z).reshape(-1, dim)

    mesh.update_vertex_positions(U)
    pc.update_vertex_positions(U[cI, :])

    plt.pause(0.0001)

    