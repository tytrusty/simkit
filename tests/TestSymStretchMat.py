import numpy as np
from simkit.symmetric_stretch_map import symmetric_stretch_map


num_trials = 100

num_tets = 500
dim = 4
for i in range(num_trials):
    S = np.random.rand(num_tets, dim, dim)
    S = S  + S.transpose(0, 2, 1)

    [C, Ci] =symmetric_stretch_map(S.shape[0], dim)

    vecS = S.reshape(-1, 1)

    s = Ci @ vecS
    vecS_test = C @ s

    err = np.linalg.norm(vecS - vecS_test)
    print(err)
    assert(err < 1e-10)