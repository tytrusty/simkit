import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
def farthest_point_sampling(V, d, sI=None):

    #pick random first index

    if sI is None:
        idx = np.argmin(V[:, 0]) #np.random.randint(0, V.shape[0])
        I = np.array([idx])
    else:
        I = np.array(sI)
        I = I.reshape(1)
    for i in range(0, d-1):

        D = pairwise_distances(V, V[I, :])
        idx = np.argmax(np.min(D, axis=1))
        I = np.append(I, idx)

    return I
