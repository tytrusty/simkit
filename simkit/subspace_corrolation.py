import numpy as np

import scipy as sp
# def subspace_corrolation(C : np.ndarray | sp.sparse.csc_matrix,  sI : np.ndarray, tI : np.ndarray, B : np.ndarray | sp.sparse.csc_matrix  =None, M : sp.sparse.csc_matrix = None):

#     var_sI = C[sI, sI]
#     var_tI = C[tI, tI]

#     if B is None:
#         H = C
#     else:
#         if M is None:
#             M = sp.sparse.identity(B.shape[0])
        
#         BMB = B.T @ M @ B
#         P = B @ np.linalg.inv(BMB) @ (B.T @ M)
#         H = P @ C @ P.T

#     cov_sI_tI = H[sI, tI]

#     rho = cov_sI_tI / np.sqrt(var_sI * var_tI)



    
#     return rho
    


def subspace_corrolation(C : np.ndarray | sp.sparse.csc_matrix, B : np.ndarray | sp.sparse.csc_matrix  =None, M : sp.sparse.csc_matrix = None):


    if B is None:
        H = C
    else:
        if M is None:
            M = sp.sparse.identity(B.shape[0])
        
        BMB = B.T @ M @ B
        P = B @ np.linalg.inv(BMB) @ (B.T @ M)
        H = P @ C @ P.T

    A = H.diagonal()[:, None]
    AA = A @ A.T
    inv_sqAA = 1 / np.sqrt(AA)
    
    rho = H * inv_sqAA    
    return rho
    