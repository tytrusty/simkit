import igl
import numpy as np


from ...solvers import NewtonSolver, NewtonSolverParams
from ... import ympr_to_lame
from ... import elastic_energy, elastic_gradient, elastic_hessian
from ... import massmatrix


from ..Sim import  *
from .ElasticFEMState import ElasticFEMState


class ElasticFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=1, g=0,  material='arap',
                 solver_p : NewtonSolverParams  = NewtonSolverParams()):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------

        """
        self.h = h
        self.rho = rho
        self.ym = ym
        self.pr = pr
        self.g = g
        self.material = material
        self.solver_p = solver_p

        return
class ElasticFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, p : ElasticFEMSimParams = ElasticFEMSimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        self.p = p

        # preprocess some quantities
        self.X = X
        self.T = T

        self.mu, self.lam = ympr_to_lame(self.ym, self.pr)


        # should also build the solver parameters
        if isinstance(p.solver_p, NewtonSolverParams):
            self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, p.solver_p)
        else:
            # print error message and terminate programme
            assert(False, "Error: solver_p of type " + str(type(p.solver_p)) +
                          " is not a valid instance of NewtonSolverParams. Exiting.")
        return


    def kinetic_energy(self, x : np.ndarray, y : np.ndarray):
        d = (x - y)
        M = massmatrix(self.X, self.T, self.rho)
        e = 0.5 * d.T @ M @ d
    def elastic_energy(self, x: np.ndarray):
        e = elastic_energy(x, self.X, self.T, self.mu, self.lam, self.material)
        return e
    def energy(self, x : np.ndarray, y : np.ndarray):
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

        K = self.kinetic_energy(x, y)
        V = self.elastic_energy(x, y)
        total = K + V
        return total



    def elastic_gradient(self, x: np.ndarray):
        g = elastic_gradient(x, self.X, self.T, self.mu, self.lam, self.material)
        return g

    def kinetic_gradient(self, x : np.ndarray, y : np.ndarray):
        M = massmatrix(self.X, self.T, self.rho)
        g = M @ (x - y)
        return g


    def gradient(self, x : np.ndarray, y : np.ndarray):
        """
        Computes the gradient of the energy of the elastic system

        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        y : (n*d, 1) numpy array
            inertial target positions (2 x_curr -  x_prev)

        Returns
        -------
        g : (n*d, 1) numpy array
            Gradient of the energy of the  elastic system
        """

        k = self.kinetic_gradient(x, y)
        v = self.elastic_gradient(x)
        f = self.g

        total = k + v + f
        return total


    def elastic_hessian(self, x: np.ndarray):
        H = elastic_hessian(x, self.X, self.T, self.mu, self.lam, self.material)
        return H

    def kinetic_hessian(self, y : np.ndarray):
        M = massmatrix(self.X, self.T, self.rho)
        return M

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
        k = self.kinetic_hessian(x)
        total = v + k
        return total


    def step(self, x : np.ndarray):
        """
        Steps the simulation forward in time.

        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        y : (n*d, 1) numpy array
            inertial target positions (2 x_curr -  x_prev)

        Returns
        -------
        x : (n*d, 1) numpy array
            Next positions of the pinned pendulum system
        """
        x = self.solver.solve(x)
        return x


    def step_sim(self, state : ElasticFEMState = ElasticFEMState()):
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
        x = self.step(state.x, state.y)
        state = ElasticFEMState(x)
        return state






