import numpy as np


def uniform_line(n, return_simplex=False):
    X = np.arange(0, n, dtype=float)[:, None] / (n - 1)
    ret = X
    if return_simplex:
        T = np.hstack(
            (np.arange(X.shape[0] - 1)[:, None], np.arange(1, X.shape[0])[:, None])
        )
        ret = (ret, )
        ret = ret + (T,)
    return ret
