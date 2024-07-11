import numpy as np

def pairwise_displacement(X : np.ndarray, Y : np.ndarray):
    """
    Returns the pairwise displacement between each row of X to each row of Y, in a 3-tensor D

    D[i, j, :] = X[i, :] - Y[j, :]
    """
    assert(X.ndim == 2)
    assert(Y.ndim == 2)
    D = X[:, None, :]
    D = np.repeat(D, Y.shape[0],axis=1)
    D = D - Y[None, :, :]

    return D