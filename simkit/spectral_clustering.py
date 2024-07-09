from sklearn.cluster import KMeans
import numpy as np

def spectral_clustering(W: np.ndarray, k: int, D=None, seed : int =0):

    if D is None:
        D = np.ones((W.shape[0], 1))

    B = W * D 

    kmeans = KMeans(n_clusters=k, random_state=seed).fit(B)
    l = kmeans.labels_
    c = kmeans.cluster_centers_
    return l, c