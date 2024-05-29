import numpy as np
from numpy import vstack, hstack
from .Sim import  *
from solvers import NewtonSolver, NewtonSolverParams



class PinnedPendulumFEMState(SimState):

    def __init__(self, x=None):
        super().__init__()
        if x is None:
            self.x = np.array([[0], [-1.0]])
        else:
            self.x = x
        return

    def primary(self):
        return self.x

    def mixed(self):
        return np.zeros((0, 1))


class PinnedPendulumFEMParams():
    def __init__(self, m=1, l0=1, mu=1, g=0, gamma=1, y=np.array([[0], [-1]]),
                 solver_p : NewtonSolverParams  = NewtonSolverParams()):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------
        m : float
            Mass of the free endpoint of the pendulum
        l0 : float
            Rest length of the spring
        mu : float
            Stiffness of the spring
        g : float
            Acceleration due to gravity
        gamma : float
            Attractive coefficient determining strength of attraction to a target point y
        y : (2, 1) numpy array
            Target point to which the pendulum is attracted
        """
        self.m = m
        self.l0 = l0
        self.mu = mu
        self.g = g
        self.gamma = gamma
        self.y = y
        self.solver_p = solver_p
        return
class PinnedPendulumFEMSim(Sim):

    def __init__(self, p : PinnedPendulumFEMParams = PinnedPendulumFEMParams()):
        """
        A simulation of an elastic pinned pendulum, which is modelled as a spring of rest length l0 and stiffness mu with one endpoint fixed at (0, 0), and the other
        end point of mass m free to move in 2D space. The pendulum is subject to gravity g, and is attracted to a target point y with attractive force gamma.

        Parameters
        ----------
        p : PinnedPendulumFEMParams
            Parameters of the pinned pendulum system
        """
        self.p = p

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
        Computes the energy of the pinned pendulum system

        Parameters
        ----------
        x : (2, 1) numpy array
            Raw state (free endpoint's 2D position) of the pinned pendulum system

        Returns
        -------
        total : float
            Total energy of the pinned pendulum system
        """
        xnorm = np.linalg.norm(x)
        elastic = 0.5 * self.p.mu * (xnorm - self.p.l0) ** 2
        gravity = self.p.m * np.array([[0], [self.p.g]]).T @ x
        target = 0.5 * self.p.gamma * np.linalg.norm(x - self.p.y) ** 2
        total = elastic + gravity + target
        return total

    def gradient(self, x : np.ndarray):
        """
        Computes the gradient of the energy of the pinned pendulum system

        Parameters
        ----------
        x : (2, 1) numpy array
            Raw state (free endpoint's 2D position) of the pinned pendulum system

        Returns
        -------
        g : (2, 1) numpy array
            Gradient of the energy of the pinned pendulum system
        """
        xnorm = np.linalg.norm(x)
        xdir = x / xnorm
        elastic = self.p.mu * xdir * (xnorm - self.p.l0)
        gravity = self.p.m * np.array([[0], [self.p.g]])
        target = self.p.gamma * (x - self.p.y)
        total = elastic + gravity + target
        return total

    def hessian(self, x : np.ndarray):
        """
        Computes the Hessian of the energy of the pinned pendulum system

        Parameters
        ----------
        x : (2, 1) numpy array
            Raw state (free endpoint's 2D position) of the pinned pendulum system

        Returns
        -------
        H : (2, 2) numpy array
            Hessian of the energy of the pinned pendulum system
        """
        xnorm = np.linalg.norm(x)
        xdir = x / xnorm
        Inorm = np.eye(2) / (xnorm)
        elastic = self.p.mu * (Inorm * (xnorm - self.p.l0) + xdir @ xdir.T)
        target = self.p.gamma * np.eye(2)
        total = elastic + target
        return total



    def step(self, x : np.ndarray):
        """
        Steps the simulation forward in time.

        Parameters
        ----------
        x : (2, 1) numpy array
            Raw state (free endpoint's 2D position) of the pinned pendulum system

        Returns
        -------
        x : (2, 1) numpy array
            Next raw state (free endpoint's 2D position) of the pinned pendulum system
        """
        x = self.solver.solve(x)
        return x


    def step_sim(self, state : PinnedPendulumFEMState = PinnedPendulumFEMState()):
        """
        Steps the simulation forward in time.

        Parameters
        ----------

        state : PinnedPendulumFEMState
            state of the pinned pendulum system

        Returns
        ------
        state : PinnedPendulumFEMState
            next state of the pinned pendulum system

        """
        x = self.step(state.x)
        state = PinnedPendulumFEMState(x)
        return state






