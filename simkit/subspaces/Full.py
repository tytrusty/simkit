import scipy as sp
import numpy as np



from .Subspace import Subspace, SubspaceParams
from ..volume import volume

class FullParams(SubspaceParams):
    def __init__(self, cache_dir=None):

        self.cache_dir = cache_dir + "/full/"
        self.name = "full"
        pass


class Full(Subspace):

    def __init__(self, X, T, params: FullParams):

        self.params = params
        self.B, self.cI, self.cW = self.compute_subspace(X, T, params)
        self.labels = np.arange(T.shape[0])
        return
    

    def compute_subspace(self, X, T, params):
        B = sp.sparse.identity(X.shape[0] * X.shape[1])
        cI =  np.arange(T.shape[0])
        cW = volume(X, T)    
        return B, cI, cW


    def vis_subspace(self, eye_pos=None, eye_target=None):
        return