import numpy as np
import scipy as sp

def simplex_vertex_map(T : np.ndarray, nv : float = None):

    dt = T.shape[-1]
    T = T.reshape(-1, dt)
    nt = T.shape[0]    

    if nv is None:
        nv = T.max() + 1

    # Slist = [ sp.sparse.csc_matrix((dt,nv)) ] * nt
    
    J = T
    I = np.repeat(np.arange(dt)[None, :], nt, axis=0) + np.arange(nt)[:, None]*dt
    V = np.ones(J.shape)
    S = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), (dt*nt, nv))

    return S
