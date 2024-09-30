import igl
import numpy as np
import scipy as sp


from simkit.cluster_grouping_matrices import cluster_grouping_matrices
from simkit.polar_svd import polar_svd
from simkit.project_into_subspace import project_into_subspace

from ...solvers import BlockCoordSolver, BlockCoordSolverParams
from ... import ympr_to_lame
from ... import elastic_energy_z, elastic_gradient_dz, elastic_hessian_d2z, ElasticEnergyZPrecomp
from ... import quadratic_energy, quadratic_gradient, quadratic_hessian
from ... import kinetic_energy_z, kinetic_gradient_z, kinetic_hessian_z, KineticEnergyZPrecomp
from ... import volume
from ... import massmatrix
from ... import deformation_jacobian, quadratic_hessian, selection_matrix



from ..Sim import  *
from ..State import State

class ElasticROMPDState(State):

    def __init__(self, z, z_dot):
        self.z = z
        self.z_dot = z_dot
        return



class ElasticROMPDSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0,
                 solver_params : BlockCoordSolverParams  = BlockCoordSolverParams(), Q0=None, b0=None, cache_dir=None, read_cache=False):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------

        """
        self.h = h
        self.rho = rho
        self.ym = ym
        self.pr = pr
        self.material = 'arap'
        self.solver_p = solver_params
        self.Q0 = Q0
        self.b0 = b0
        self.cache_dir = cache_dir
        self.read_cache = read_cache
        return


class ElasticROMPDSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray, labels=None, params : ElasticROMPDSimParams = ElasticROMPDSimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        
        dim = X.shape[1]
        self.dim = dim

        self.params = params
        
        # preprocess some quantities
        self.X = X
        self.T = T

        # i hate initializing state here. get rid of this.
        x = X.reshape(-1, 1)
        self.x = x
        self.x_dot = None
        self.x0 = None

        self.B = B
     
        ## cubature precomp
        if labels is None:
            labels = np.arange(0, T.shape[0])

        G, Gm = cluster_grouping_matrices(labels, X, T)

        vol = volume(X, T)
 
        ## ym, pr to lame parameters
        mu, _lam = ympr_to_lame(self.params.ym, 0)
        if isinstance(mu, float):
            mu = np.ones((T.shape[0], 1)) * mu

        AMu = sp.sparse.diags(mu.flatten() * vol.flatten() )
        
        AMue = sp.sparse.kron(AMu, sp.sparse.identity(dim*dim))

        GAMue = sp.sparse.kron(G @ AMu , sp.sparse.identity(dim*dim))
        J = deformation_jacobian(X, T)

                                                                                        
        self.AMuGJB = (  GAMue @ J) @ B
        self.B = B
                                                                                                                                                                                                                                                                                                                                                                 

        L = J.T @ AMue @ J


        self.AMuGJB_full =  GAMue @ J @ B


        Mv = sp.sparse.kron(massmatrix(X, T, rho=params.rho), sp.sparse.identity(dim))
        self.BLB = B.T @ L @ B 
        self.BMB = B.T @ Mv @ B
        self.BMy = B.T @ Mv @ X.reshape(-1, 1)

        self.kin_pre = KineticEnergyZPrecomp(B, Mv)
        
        # [self.kin_pre, self.el_pre, self.mu, self.vol] =  \
        #     self.initial_precomp(X, T, B, cI, cW, p.rho, p.ym, dim)

        if params.Q0 is None:
            self.Q = np.zeros((B.shape[1], B.shape[1])) #sp.sparse.csc_matrix((B.shape[1], B.shape[1]))
        else:
            self.Q = self.params.Q0
        if params.b0 is None:
            self.b = np.zeros((B.shape[1], 1))
        else:
            assert(params.b0.shape[0] == B.shape[-1])
            self.b = self.params.b0.reshape(-1, 1)

        self.recompute_system()

        # should also build the solver parameters
        assert(isinstance(params.solver_p, BlockCoordSolverParams))
        self.solver = BlockCoordSolver(self.global_step, self.local_step, params.solver_p)
        
        return

    # def initial_precomp(self, X, T, B,  cI, cW, rho, ym, dim):
        
    #     # kinetic energy precomp
    #     M = massmatrix(self.X, self.T, rho=rho)
    #     Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
    #     kin_z_precomp = KineticEnergyZPrecomp(B, Mv)
        
        
    #     # elastic energy precomp
    #     if cW is None:
    #         vol = volume(X, T)
    #     else:
    #         vol = cW.reshape(-1, 1)
            
    #     ## ym, pr to lame parameters
    #     mu, lam = ympr_to_lame(ym, pr)
    #     if isinstance(mu, float):
    #         mu = np.ones((T.shape[0], 1)) * mu

    #     mu = mu[cI]

    #     ## selection matrix from cubature precomp.
    #     G = selection_matrix(cI, T.shape[0])
    #     Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
    #     J = deformation_jacobian(self.X, self.T)
    #     elastic_z_precomp = ElasticEnergyZPrecomp(B, Ge, J, self.dim)



    #     self.dynamic_precomp_done = False
    #     return  kin_z_precomp, elastic_z_precomp, mu, lam, vol

    def recompute_system(self):

        self.H = self.BLB + self.BMB / self.params.h**2 + self.Q

        self.is_sparse = False
        if (sp.sparse.issparse(self.B)):
            self.is_sparse = True
            self.chol_H = sp.sparse.linalg.factorized(self.H)
        else:
            self.chol_H = sp.linalg.cho_factor(self.H)

        return
    
    def dynamic_precomp(self, z, z_dot, b_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = z + self.params.h * z_dot

        # same for b
        if b_ext is not None:
            if self.params.b0 is None:
                self.b = b_ext
            else:
                self.b = b_ext + self.params.b0
        
        self.dynamic_precomp_done = True

    def energy(self, z : np.ndarray):
        k = kinetic_energy_z(z, self.y ,self.params.h, self.kin_pre)

        r = self.local_step(z)
        
        v = 0.5 *  z.T @ self.BLB @ z + z.T @ self.AMuGJB_full.T @ r
        
        quad =  quadratic_energy(z, self.Q, self.b)
        
        total = k + v + quad
        return total

    # def gradient(self, z : np.ndarray):
    #     k = kinetic_gradient_z(z, self.y, self.params.h, self.kin_pre)
    #     v = elastic_gradient_dz(z, self.mu, self.lam, self.vol, self.params.material, self.el_pre)
    #     quad = quadratic_gradient(z, self.Q, self.b)
    #     total = v  + k + quad
    #     return total

    # def hessian(self, z : np.ndarray):
    #     v = elastic_hessian_d2z(z, self.mu, self.lam, self.vol, self.params.material, self.el_pre)
    #     k = kinetic_hessian_z(self.params.h, self.kin_pre)
    #     quad = quadratic_hessian(self.Q)
    #     total = v + k + quad
    #     return total


    def local_step(self, z : np.ndarray):
        c = self.AMuGJB @ z
        C = c.reshape(-1, self.dim, self.dim)
        
        R = polar_svd(C)[0]
        # R = R.transpose(0, 2, 1)
        r = R.reshape(-1, 1)
        
        # p =  sp.linalg.spsolve(  self.G r)
        return r
    
    def global_step(self, z : np.ndarray, r: np.ndarray):
        # k = kinetic_gradient_z(z, self.y, self.params.h, self.kin_pre)
        # q = quadratic_gradient(z, self.Q, self.b)
        # || JB z - r ||
        k = self.BMB @ self.y / self.params.h**2
        e =   self.AMuGJB.T @  r 

        rhs = k + e - self.b
        if self.is_sparse:
            z_next = self.chol_H(rhs)
        else:
            z_next = sp.linalg.cho_solve(self.chol_H, rhs)

        # q = quadratic_gradient(z, self.Q, self.b)
        # z_next = np.linalg.solve(self.H, k + e  - self.b)
        return z_next

    def step(self, z : np.ndarray, z_dot : np.ndarray, b_ext=None):
        # call this to set up inertia forces that change each timestep.
        self.dynamic_precomp(z, z_dot, b_ext)

        z0 = z.copy() # very important to copy this here so that x does not get over-written
        z_next = self.solver.solve(z0)
        # z_next = project_into_subspace(x_next, self.B)

        self.dynamic_precomp_done = False
        return z_next
    
    def rest_state(self):


        z = project_into_subspace( self.X.reshape(-1, 1), self.B,
                        M=sp.sparse.kron(massmatrix(self.X, self.T), sp.sparse.identity(self.dim)), BMB=self.BMB, BMy=self.BMy)# 

        z_dot = np.zeros_like(z)
        return z, z_dot




