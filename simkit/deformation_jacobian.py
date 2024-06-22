import igl
import scipy as sp
import numpy as np


from .simplex_vertex_map import simplex_vertex_map

def deformation_jacobian(X : np.array, T : np.array):
    '''
    Linear mapping between positions and deformation gradients, assuming x has been flattened with default order="C":

    Parameters
    ----------
    V : (n, dim) array
        The vertices of the mesh

    T : (t, 3|4) array
        Simplex indices

    Returns
    -------
    J : (d*d*t, d*n) sparse matrix
        The deformation Jacobian matrix

    Example
    -------
    ```python
    x = X.reshape(-1, 1)
    f = J @ x
    F = f.reshape(-1, 3, 3)
    ```

    In the above, F will have the form:
    F = [[Fxx Fxy Fxz]]
        [[Fyx Fyy Fyz]]
        [[Fzx Fzy Fzz]]
    '''
    dt = T.shape[-1]
    T = T.reshape(-1, dt)
    nt = T.shape[0]    
    dim = X.shape[1]

    if dim == 2:
        H = np.array([[-1, -1],
                        [1, 0],
                        [0, 1]])
    if dim == 3:
        H = np.array([[-1, -1, -1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    XT = X[T].transpose(0, 2, 1)
    XH = XT @ H
    XHi = np.linalg.inv(XH)
    D = (H @ XHi).transpose(0, 2, 1)

    De = np.zeros((nt, dim*dim,  dt*dim))
   
    for i in range(dim):
        Di = np.zeros((nt, dim, dt*dim))
        Ii = np.arange(dt)*dim + i
        Di[:, :, Ii] = D
        
        De[:, dim*i:dim*(i + 1), :] = Di      
      

    Ii = np.arange(dim*dim*nt).reshape(nt, dim*dim, 1)
    Ii = np.repeat(Ii, dim*dt, axis=2 )

    Ji = np.arange(dim*dt*nt).reshape(nt, 1, dim*dt)
    Ji = np.repeat(Ji, dim*dim, axis=1 )

    dims = np.prod(De.shape).item()
    H = sp.sparse.csc_matrix((De.flatten(), (Ii.flatten(), Ji.flatten())), (nt*dim*dim, nt*dim*dt))

    S = simplex_vertex_map(T)
    Se = sp.sparse.kron(S, sp.sparse.identity(dim))
    J = H @ Se
    return J



# Otman : I think the deformation jacobian function should be written explicitly for each element,
# for a better learning opportunity, and for less dependence on libigl.

# '''
# Gradient of triangle or tet mesh . For now, U is scalar
# '''
# def gradient(X, F, U, HXHi=None):

#     if (len(U.shape) == 2):
#         U = U[None, :, :]
#     t = F.shape[1]  # simplex size
#     dx = X.shape[1]
#     du = U.shape[2]

#     TU =(U[:, F]).transpose([0, 1, 3, 2])
#     assert (U.dtype == X.dtype)
#     if (HXHi is None):
#         if t == 3:
#             # triangle mesh!
#             H = np.array([[-1.0, -1],
#                           [1., 0],
#                           [0, 1.0]], dtype=U.dtype)

#         # for each triangle, get the three vertex positions dealing with it
#         elif F.shape[1] == 4:
#             # triangle mesh!
#             H = np.array([[-1, -1, -1],
#                           [1.0, 0, 0],
#                           [0, 1.0, 0],
#                           [0, 0, 1.0]], dtype=U.dtype)
#         else:
#             raise ValueError("Only triangle and tet meshes are supported")
#         Tx = X[F].transpose([0, 2, 1])
#         XH = Tx @ H
#         XHi = np.linalg.pinv(XH)
#         HXHi = H @ XHi

#         grad = TU @ HXHi
#         # grad = torch.squeeze(grad, dim=1)

#         # squeze first two dims
#         grad = np.squeeze(grad)
#         return grad, HXHi
#     else:
#         grad = TU @ HXHi
#         grad = np.squeeze(grad)
#         return grad, HXHi
