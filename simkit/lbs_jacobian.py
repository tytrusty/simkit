import numpy as np


def lbs_jacobian(V, W):

    n = V.shape[0]
    d = V.shape[1]
    k = W.shape[1]

    one_d1 = np.ones((d+1, 1))
    one_k = np.ones((k, 1))

    # append 1s to V to make V1 , homogeneous
    V1 = np.hstack((V, np.ones((V.shape[0], 1))))


    Wexp = np.kron( W, one_d1.T)
    V1exp = np.kron( one_k.T, V1)
    J = Wexp * V1exp
    Jexp = np.kron( J, np.identity(d))

    return Jexp