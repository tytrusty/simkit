import numpy as np

def kinetic_hessian(M, h : float):
   
    H = M *(1/ (h**2))
    return H
