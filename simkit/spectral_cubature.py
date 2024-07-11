
import numpy as np


from .volume import volume
from .pairwise_distance import pairwise_distance
from .average_onto_simplex import average_onto_simplex
from .spectral_clustering import spectral_clustering

def spectral_cubature(X, T, W, k, return_labels=False, return_centroids=False):
    Wt = average_onto_simplex(W, T)

    [labels, centroids] = spectral_clustering(Wt, k)
    D = pairwise_distance(centroids, Wt)
    lI = np.argmin(D, axis=1)
    m = volume(X, T)
    mc = np.bincount(labels, m.flatten())

    ret = (lI, mc)

    if return_labels:
        ret = ret + (labels,)
    if return_centroids:
        ret = ret + (centroids,)

    return ret

