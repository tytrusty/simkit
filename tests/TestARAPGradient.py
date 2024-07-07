import numpy as np

from simkit.arap_gradient import arap_gradient_dS
from simkit.symmetric_stretch_map import symmetric_stretch_map
num_trials = 100

dim = 2
num_t = 100

C, Ci = symmetric_stretch_map(num_t, dim)

# test the stretch part
for i in range(num_trials):
    mu = np.abs(np.random.rand(num_t, 1) * 100)
    vol = np.abs(np.random.rand(num_t, 1) * 100)

    
    S = np.random.rand(num_t, dim, dim)
    S = (S  + S.transpose(0, 2, 1 ))/2
    s = (Ci @ S.reshape(-1, 1)).reshape(-1, int(dim * (dim + 1)/ 2))

    S_t = (C @ s.reshape(-1, 1)).reshape(-1, dim, dim)
    g_S = arap_gradient_dS(S, mu, vol)
    g_s = Ci.T @ arap_gradient_dS(s, mu, vol)



    err = np.linalg.norm(g_S - g_s)

    assert(np.abs(g_S - g_s) < 1e-10)

     


