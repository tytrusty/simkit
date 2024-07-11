import igl
import numpy as np
import scipy as sp
from simkit import deformation_jacobian, project_into_subspace, quadratic_hessian

from ...solvers import NewtonSolver, NewtonSolverParams
from ... import ympr_to_lame
from ... import elastic_energy_z, elastic_gradient_dz, elastic_hessian_d2z, ElasticEnergyZPrecomp
from ... import quadratic_energy, quadratic_gradient, quadratic_hessian
from ... import kinetic_energy_z, kinetic_gradient_z, kinetic_hessian_z, KineticEnergyZPrecomp
from ... import volume
from ... import massmatrix
from ...project_into_subspace import project_into_subspace


from ..Sim import  *
from .ElasticFEMState import ElasticFEMState


class ElasticROMFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0, g=0,  material='arap',
                 solver_p : NewtonSolverParams  = NewtonSolverParams(), f_ext = None, Q=None, b=None):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------

        """
        self.h = h
        self.rho = rho
        self.ym = ym
        self.pr = pr
        self.material = material
        self.solver_p = solver_p
        self.f_ext = f_ext
        self.Q = Q
        self.b = b
        return


class ElasticROMFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray, cI=None,  cW=None, p : ElasticROMFEMSimParams = ElasticROMFEMSimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        
        dim = X.shape[1]
        self.dim = dim

        self.p = p
        self.mu, self.lam = ympr_to_lame(self.p.ym, self.p.pr)

        # preprocess some quantities
        self.X = X
        self.T = T
        x = X.reshape(-1, 1)
        self.x = x
        self.x_dot = None
        self.x0 = None

        
        M = massmatrix(self.X, self.T, rho=self.p.rho)
        self.M = M
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        self.Mv = Mv

        # elastic energy, gradient and hessian
        self.J = deformation_jacobian(self.X, self.T)
        self.B = B


        self.kin_z_precomp = KineticEnergyZPrecomp(self.B, self.Mv)
        self.elastic_z_precomp = ElasticEnergyZPrecomp(self.J, self.B, self.dim)

        if cI is not None:
            assert cW is not None

        if cI is None:
            cI = np.arange(0, self.T.shape[0])

        if cW is None:
            self.vol = volume(self.X, self.T)
        else:
            self.vol = cW



        if p.Q is None:
            self.Q = sp.sparse.csc_matrix((x.shape[0], x.shape[0]))
        else:
            assert(p.Q.shape[0] == B.shape[-1])
            self.Q = self.p.Q
        if p.b is None:
            self.b = np.zeros((x.shape[0], 1))
        else:
            assert(p.b.shape[0] == B.shape[-1])
            self.b = self.p.b.reshape(-1, 1)

        if p.f_ext is None:
            self.f_ext = np.zeros((x.shape[0], 1))
        else:
            assert(p.f_ext.shape[0] == x.shape[0])
            self.f_ext = self.p.f_ext.reshape(-1, 1)


        # should also build the solver parameters
        if isinstance(p.solver_p, NewtonSolverParams):
            self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, p.solver_p)
        else:
            # print error message and terminate programme
            assert(False, "Error: solver_p of type " + str(type(p.solver_p)) +
                          " is not a valid instance of NewtonSolverParams. Exiting.")
        return


    def energy(self, z : np.ndarray):
        k = kinetic_energy_z(z, self.y ,self.p.h, self.kin_z_precomp)
        v = elastic_energy_z(z,  self.mu, self.lam, self.vol, self.p.material,  self.elastic_z_precomp)
        quad =  quadratic_energy(z, self.Q, self.b)
        total = k + v + quad
        return total


    def gradient(self, z : np.ndarray):
        k = kinetic_gradient_z(z, self.y, self.p.h, self.kin_z_precomp)
        v = elastic_gradient_dz(z, self.mu, self.lam, self.vol, self.p.material, self.elastic_z_precomp)
        quad = quadratic_gradient(z, self.Q, self.b)
        total = v  + k + quad
        return total


    def hessian(self, z : np.ndarray):
        v = elastic_hessian_d2z(z, self.mu, self.lam, self.vol, self.p.material, self.elastic_z_precomp)
        k = kinetic_hessian_z( self.p.h, self.kin_z_precomp)
        quad = quadratic_hessian(self.Q)
        total = v + k + quad
        return total


    def step(self, z : np.ndarray, z_dot : np.ndarray):
        self.y = z + self.p.h * z_dot
        z0 = z.copy() # very important to copy this here so that x does not get over-written
        z_next = self.solver.solve( z0)
        # z_next = project_into_subspace(x_next, self.B)
        return z_next


    def step_sim(self, state : ElasticFEMState ):
        z= self.step(state.z, state.z_dot)
        z_dot = (z - state.z)/ self.p.h
        state = ElasticFEMState(z, z_dot)
        return state






