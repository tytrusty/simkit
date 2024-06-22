
import igl
import numpy as np

from simkit import deformation_jacobian
from simkit.dirichlet_laplacian import dirichlet_laplacian



# 2D Random transforms
[X, _, _, T, _, _] = igl.read_obj("./data/2d/beam/beam.obj")
X = X[:, :2]

N = 100
import time
J = deformation_jacobian(X, T)
for i in range(N):
    A = np.random.rand(2, 2)
    U = (A @ X.T).T

    F = (J @ U.reshape(-1 , 1)).reshape(-1, 2, 2)

    assert(np.allclose(F, A))
    

# 3D Random transforms
[X,T, F] = igl.read_mesh("./data/3d/treefrog/treefrog.mesh")

N = 100
import time
J = deformation_jacobian(X, T)
for i in range(N):
    A = np.random.rand(3, 3)
    U = (A @ X.T).T

    F = (J @ U.reshape(-1 , 1)).reshape(-1, 3, 3)


    assert(np.allclose(F, A))
    
    



