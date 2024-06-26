import igl
import numpy as np
import scipy as sp

from ...solvers import NewtonSolver, NewtonSolverParams
from ... import ympr_to_lame
from ... import elastic_energy, elastic_gradient, elastic_hessian
from ... import massmatrix


from ..Sim import  *
from .ElasticFEMState import ElasticFEMState


class ElasticFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=1, g=0,  material='arap',
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

    def kinetic_energy(self, x : np.ndarray):
        d = (x - self.y)
        M = self.Mv
        e = 0.5 * d.T @ M @ d * (1/ (self.p.h**2))
        return e

    def elastic_energy(self, x: np.ndarray):
        e = elastic_energy(x, self.X, self.T, self.mu, self.lam, self.p.material)
        return e

    def quadratic_energy(self, x : np.ndarray):
        Q = self.Q
        b = self.b
        e = 0.5 * x.T @ Q @ x + b.T @ x
        return e

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

    def elastic_gradient(self, x: np.ndarray):
        g = elastic_gradient(x, self.X, self.T, self.mu, self.lam, self.p.material)
        return g

    def kinetic_gradient(self, x : np.ndarray):
        g = self.Mv @ (x - self.y) * (1/ (self.p.h**2))
        return g

    def quadratic_gradient(self, x : np.ndarray):
        Q = self.Q
        b = self.b
        g = Q @ x + b
        return g


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


    def elastic_hessian(self, x: np.ndarray):
        H = elastic_hessian(x, self.X, self.T, self.mu, self.lam, self.p.material)
        return H

    def kinetic_hessian(self):
        M = self.Mv * (1/ (self.p.h**2))
        return M

    def quadratic_hessian(self):
        Q = self.Q
        return Q


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
        x0 = x.copy()
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
        x = self.step(state.x, state.x_prev)
        state = ElasticFEMState(x)
        return state






