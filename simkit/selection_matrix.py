import scipy as sp
import numpy as np

def selection_matrix(cI, n):
    """
    Returns a selection matrix that selects the rows of a matrix
    corresponding to the indices in cI

    Parameters
    ----------
    cI : list
        list of indices to select
    n : int
        number of rows in the matrix

    Returns
    -------
    G : (len(cI), n) sparse matrix
        selection matrix
    """

    I = np.arange(cI.shape[0])
    J = cI

    v = np.ones(cI.shape[0])

    G = sp.sparse.csc_matrix((v, (I, J)), 
                             shape=(cI.shape[0], n))
    return G