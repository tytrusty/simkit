import igl
import numpy as np
import scipy as sp

from simkit import deformation_jacobian, quadratic_hessian

from ...solvers import NewtonSolver, NewtonSolverParams
from ... import ympr_to_lame
from ... import elastic_energy_x, elastic_gradient_dx, elastic_hessian_d2x
from ... import quadratic_energy, quadratic_gradient, quadratic_hessian
from ... import kinetic_energy, kinetic_gradient, kinetic_hessian
from ... import volume

from ... import massmatrix


from ..Sim import  *
from .ElasticFEMState import ElasticFEMState


class ElasticFEMSimParams():
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


class ElasticFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, p : ElasticFEMSimParams = ElasticFEMSimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        dim = X.shape[1]

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
        self.vol = volume(self.X, self.T)
        
        self.elastic_energy = lambda x: elastic_energy_x(x.reshape(-1, dim), self.J, self.mu, self.lam, self.vol, self.p.material)
        self.elastic_gradient = lambda x: elastic_gradient_dx(x.reshape(-1, dim), self.J, self.mu, self.lam, self.vol, self.p.material)
        self.elastic_hessian = lambda x: elastic_hessian_d2x(x.reshape(-1, dim), self.J, self.mu, self.lam, self.vol, self.p.material)

        # quadratic energy, gradient and hessian
        self.quadratic_energy = lambda x: quadratic_energy(x, self.Q, self.b)
        self.quadratic_gradient = lambda x: quadratic_gradient(x, self.Q, self.b)
        self.quadratic_hessian = lambda : quadratic_hessian(self.Q)

        # kinetic energy, gradient and hessian
        self.y = x
        self.kinetic_energy = lambda x: kinetic_energy(x, self.y, self.Mv, self.p.h)
        self.kinetic_gradient = lambda x: kinetic_gradient(x, self.y, self.Mv, self.p.h)
        self.kinetic_hessian = lambda : kinetic_hessian(self.Mv, self.p.h)


        if p.Q is None:
            self.Q = sp.sparse.csc_matrix((x.shape[0], x.shape[0]))
        else:
            assert(p.Q.shape[0] == x.shape[0] and p.Q.shape[1] == x.shape[0])
            self.Q = self.p.Q
        if p.b is None:
            self.b = np.zeros((x.shape[0], 1))
        else:
            assert(p.b.shape[0] == x.shape[0])
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


    def energy(self, x : np.ndarray):
        """
        Computes the energy of the elastic system

        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        y : (n*d, 1) numpy array
            inertial target positions (2 x_curr -  x_prev)
        Returns
        -------
        total : float
            Total energy of the elastic system
        """
        k = self.kinetic_energy(x)
        v = self.elastic_energy(x)
        quad = self.quadratic_energy(x)
        total = k + v + quad
        return total


    def gradient(self, x : np.ndarray):
        """
        Computes the gradient of the energy of the elastic system

        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        Returns
        -------
        g : (n*d, 1) numpy array
            Gradient of the energy of the  elastic system
        """

        k = self.kinetic_gradient(x)
        v = self.elastic_gradient(x)
        quad = self.quadratic_gradient(x)
        total = k + v + quad
        return total


    def hessian(self, x : np.ndarray):
        """
        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        Returns
        -------
        g : (n*d, n*d)  sparse matrix
            Hessian of the energy of the elastic system
        """
        v = self.elastic_hessian(x)
        k = self.kinetic_hessian()
        quad = self.quadratic_hessian()
        total = v + k + quad
        return total


    def step(self, x : np.ndarray, x_dot : np.ndarray):
        """
        Steps the simulation forward in time.

        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        x_dot : (n*d, 1) numpy array
            velocity of the elastic system

        Returns
        -------
        x : (n*d, 1) numpy array
            Next positions of the pinned pendulum system
        """
        self.y = x + self.p.h * x_dot
        x0 = x.copy() # very important to copy this here so that x does not get over-written
        x_next = self.solver.solve(x0)
        return x_next


    def step_sim(self, state : ElasticFEMState ):
        """
        Steps the simulation forward in time.

        Parameters
        ----------

        state : Elastic2DFEMState
            state of the pinned pendulum system

        Returns
        ------
        state : Elastic2DFEMState
            next state of the pinned pendulum system

        """
        x= self.step(state.x, state.x_dot)
        x_dot = (x - state.x)/ self.p.h
        state = ElasticFEMState(x, x_dot)
        return state






