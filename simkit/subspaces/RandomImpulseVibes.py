import os
import warnings
import scipy as sp
import numpy as np
import igl


from ..lbs_jacobian import lbs_jacobian


from .Subspace import Subspace, SubspaceParams

from ..spectral_basis_localization import spectral_basis_localization

from ..farthest_point_sampling import farthest_point_sampling
from ..selection_matrix import selection_matrix
from ..skinning_eigenmodes import skinning_eigenmodes
from ..orthonormalize import orthonormalize
from ..spectral_cubature import spectral_cubature
from ..massmatrix import massmatrix
from ..random_impulse_vibes import random_impulse_vibes

class RandomImpulseVibesParams(SubspaceParams):
    def __init__(self,  m, k,c,h=1e-1, cache_dir = None, read_cache=False, sparse=False, threshold=1e-3, ord=1):

        self.h = h
        self.m = m  
        self.k = k
        self.c = c
        self.read_cache = read_cache

        self.name = "spectral_localized_skinning_eigenmodes_m"+str(m)+"_k"+str(k)+"_c"+str(c)
        if cache_dir is not None:
            self.cache_dir = cache_dir + "/" + self.name + "/"
        else:
            self.cache_dir = None
        self.sparse = sparse
        self.threshold = threshold
        self.ord = ord
        pass


class RandomImpulseVibes(Subspace):

    def __init__(self, X, T, params: RandomImpulseVibesParams):


        self.params = params
        cache_dir = params.cache_dir

        well_read = False

        self.X = X
        self.T = T
        if params.read_cache:
            try:
                [ W, B, cI, cW, labels, cId, conI, SB] = self.read_subspace(cache_dir)
                well_read = True
            except:
                warnings.warn("Warning : Couldn't read subspace from cache. Recomputing from scratch...")
        if not well_read:
            [ W, B, cI, cW, labels, cId, conI, SB] = self.compute_subspace(X, T, params)
            if params.cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                self.save_subspace(cache_dir, W,  B, cI, cW, labels, cId, conI, SB)
        self.W = W
        self.B = B
        self.cI = cI
        self.cW = cW
        self.labels = labels
        self.cId = cId

        self.conI = conI
        self.SB = SB
        return
    
    def read_subspace(self, cache_dir):
        W = np.load(cache_dir + "W.npy")
        try:
            B = np.load(cache_dir + "B.npy", allow_pickle=True).item()
        except:
            B = np.load(cache_dir + "B.npy")
        # B = np.load(cache_dir + "B.npy", allow_pickle=True).item()
        try:
            cI = np.load(cache_dir + "cI.npy", allow_pickle=True).item()
        except:
            cI = np.load(cache_dir + "cI.npy")
        cW = np.load(cache_dir + "cW.npy")
        labels = np.load(cache_dir + "labels.npy")
        cId = np.load(cache_dir + "cId.npy")
        conI = np.load(cache_dir + "conI.npy")

        try:
            SB = np.load(cache_dir + "SB.npy", allow_pickle=True).item()
        except:
            SB = np.load(cache_dir + "SB.npy")

        return W, B, cI, cW, labels, cId, conI, SB
    
    def save_subspace(self,cache_dir,  W,  B, cI, cW, labels, cId, conI, SB):

        np.save(cache_dir + "W.npy", W)
        np.save(cache_dir + "B.npy", B)
        np.save(cache_dir + "cI.npy", cI)
        np.save(cache_dir + "cW.npy", cW)
        np.save(cache_dir + "labels.npy", labels)
        np.save(cache_dir + "cId.npy", cId)
        np.save(cache_dir + "conI.npy", conI)
        np.save(cache_dir + "SB.npy", SB)
        return

    def compute_subspace(self, X, T, params):
          
        dim = X.shape[1]
        # [Wd, E,  Bd] = skinning_eigenmodes(X, T, params.d)

        # W, cId, slabels, scluster_means = spectral_basis_localization(X, T, params.m, W=Wd, return_clustering_info=True)

        W, cId, W_full = random_impulse_vibes(X, T, params.m, params.h, ord=params.ord)
        
        Wdense = W.copy()
        is_sparse = False
        
        
        if params.sparse:
            Wa = np.abs(W)
            Wt = W.copy()
            Wt[Wa < params.threshold] = 0
            Ws = sp.sparse.csr_matrix(Wt)

            # density = density_ratio(Ws)
            # if density < 0.6:
            # print("Subspace density is " + str(density) +  " ... making it sparse")
            W = Ws
            is_sparse = True
        
        B= lbs_jacobian(X, Wdense)
        # B = orthonormalize(B, M=sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim)))
        if is_sparse:
            B = sp.sparse.csc_matrix(B)


        [Wd, E,  Bd] = skinning_eigenmodes(X, T, 10)

        [cI, cW, labels] = spectral_cubature(X, T, Wd,params.k, return_labels=True)

        n = X.shape[0]
        faceI = np.unique(igl.boundary_facets(T))
        sI = farthest_point_sampling(X[faceI], params.c)
        conI = faceI[sI]
        S = selection_matrix(conI, n)
        Se = sp.sparse.kron(S, sp.sparse.identity(dim))
        SB = Se @ B 

        # density = density_ratio(SB)
        return W, B, cI, cW, labels, cId, conI, SB
    

    def vis_subspace(self, eye_pos=None, eye_target=None):
        # view_scalar_modes(self.X, self.T, self.Wd, dir = self.params.cache_dir + "/Wd/", normalize=True, eye_pos=eye_pos, eye_target=eye_target)
        # view_scalar_modes(self.X, self.T, self.W, dir = self.params.cache_dir + "/W/", normalize=True, eye_pos=eye_pos, eye_target=eye_target)

        # view_clusters(self.X, igl.boundary_facets(self.T), self.slabels, path = self.params.cache_dir + "/slabels.png", eye_pos=eye_pos, eye_target=eye_target, pI=self.cId)
    
        # view_clusters(self.X, self.T, self.labels, path = self.params.cache_dir + "/labels.png", eye_pos=eye_pos, eye_target=eye_target)
    
        # import polyscope as ps
        # ps.show()
        return

