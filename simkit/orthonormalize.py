import scipy as sp
import numpy as np


def orthonormalize(B, M=None):
    if M is None:
        M = sp.sparse.identity(B.shape[0])
    # M = sp.sparse.identity(B.shape[0])

    msqrt = np.sqrt(M.diagonal())
    msqrti = 1 / msqrt
    Msqrt = sp.sparse.diags(msqrt, 0)
    Msqrti = sp.sparse.diags(msqrti, 0)
    Bm = Msqrt @ B


    [Q, R] = np.linalg.qr(Bm)

    sing = np.abs(R).sum(axis=0)
    nonsing = np.abs(R).sum(axis=1) > 1e-4  # check which rows are singular
    B3 = Msqrti @ Q[:, nonsing]

    return B3