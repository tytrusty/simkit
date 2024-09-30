

def density_ratio(A):
    nnz = A.nnz
    nelem = A.shape[0] * A.shape[1]
    return nnz / nelem