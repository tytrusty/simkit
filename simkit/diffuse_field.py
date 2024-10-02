
import numpy as np
import scipy as sp

from .edge_lengths import edge_lengths
from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix



def diffuse_field(Vv, Tv,bI,  phi, dt=None, normalize=True):
    """ Performs a diffusion on the tet mesh Vv, Tv at nodes bI for time dt.

    Parameters
    ----------
    Vv : (n, 3) float numpy array
        Mesh vertices
    Tv : (t, 4) int numpy array
        Mesh tets
    bI : (c, b) int numpy array
        Indices at diffusion points
    phi : (c, b) float numpy array
        Quantity to diffuse
    dt : float
        Time to diffuse for
    normalize : bool
        Whether to normalize the weights

    Returns
    -------
    W : (n, b) float numpy array
        Diffused quantities over entire mesh

    """
    if (dt is None):
        dt = np.mean(edge_lengths(Vv, Tv)) ** 2

    L = dirichlet_laplacian(Vv, Tv)
    M = massmatrix(Vv, Tv)

    Q = L * dt + M

    ii = np.setdiff1d(  np.arange(Q.shape[0]), bI)
    # selection matrix for indices bI
    Qii = Q[ii, :][:, ii]
    Qib = Q[ii, :][:, bI]


    Wii = sp.sparse.linalg.spsolve(Qii, -Qib @ phi)
    Wii = Wii.reshape(-1, phi.shape[1])
    W = np.zeros((L.shape[0], Wii.shape[1]))
    W[ii, :] = Wii
    W[bI, :] = phi


    if W.ndim == 1:
        W = W[:, None]
    if normalize:
        W = (W - np.min(W, axis=0)) / (np.max(W, axis=0) - np.min(W, axis=0))
    return W