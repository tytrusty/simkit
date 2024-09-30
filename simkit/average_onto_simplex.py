
import numpy as np


def average_onto_simplex(A, T):
    At = np.zeros((T.shape[0], A.shape[1]))
    for td in range(T.shape[1]):
        At += (A[T[:, td], :])/T.shape[1]

    return At