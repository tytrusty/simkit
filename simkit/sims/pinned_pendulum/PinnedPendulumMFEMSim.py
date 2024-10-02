import numpy as np
from numpy import vstack, hstack

from ...solvers import  NewtonSolverParams, NewtonSolver

class PinnedPendulumMFEMSimParams():
    def __init__(self, m=1, l0=1, mu=1, g=0, gamma=1, y=np.array([[0], [-1]]),
                 solver_p : NewtonSolverParams  = NewtonSolverParams(),eta=1):
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
        eta : float
            Coefficient of the constraint term in the merit function
        """
        self.m = m
        self.l0 = l0
        self.mu = mu
        self.g = g
        self.gamma = gamma
        self.y = y
        self.solver_p = solver_p
        self.eta = eta

        return
class PinnedPendulumMFEMSim():

    def __init__(self, p : PinnedPendulumMFEMSimParams = PinnedPendulumMFEMSimParams()):
        """
        A simulation of an elastic pinned pendulum, which is modelled as a spring of rest length l0 and stiffness mu with one endpoint fiped at (0, 0), and the other
        end point of mass m free to move in 2D space. The pendulum is subject to gravity g, and is attracted to a target point y with attractive force gamma.

        Parameters
        ----------
        p : PinnedPendulumMFEMSimParams
            Parameters of the pinned pendulum system
        """
        self.p = p

        self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, p.solver_p)
        
        return

    def energy(self, p):
        """
        Computes the energy of the pinned pendulum system following a miped discretization of lengths and positions.

        Parameters
        ----------
        p : (4, 1) numpy array
            State of the pinned pendulum system, consisting of the free endpoint's 2D position p, the length of the spring s, and the Lagrange multiplier l

        Returns
        -------
        e : float
            Total energy of the pinned pendulum system following a miped discretization of lengths and positions

        """
        x = p[:2]
        s = p[2]
        l = p[3]
        elastic = 0.5 * self.p.mu * (s - self.p.l0) ** 2
        gravity = self.p.m * np.array([[0], [self.p.g]]).T @ x
        target = 0.5 * self.p.gamma * np.linalg.norm(x - self.p.y) ** 2
        constraint =  self.p.eta * (np.linalg.norm(x) - s)**2
        total = elastic + gravity + target  + constraint
        return total

    def gradient(self, p):
        """
        Computes the gradient of the energy of the pinned pendulum system following a miped discretization of lengths and
        positions.

        Parameters
        ----------
        p : (4, 1) numpy array
            State of the pinned pendulum system, consisting of the free endpoint's 2D position p, the length of the spring s, and the Lagrange multiplier l

        Returns
        -------
        g : (4, 1) numpy array
            Gradient of the energy of the pinned pendulum system following a miped discretization of lengths and positions


        """
        x = p[:2]
        s = p[2]
        l = p[3]
        pnorm = np.linalg.norm(x)
        pdir = x / pnorm

        dEdp = l * pdir + self.p.gamma * (x - self.p.y)
        dEds = self.p.mu * (s - self.p.l0) - l
        dEdl = pnorm - s

        g = np.vstack([dEdp, dEds, dEdl])
        return g

    def hessian(self, p):
        """
        Computes the Hessian of the energy of the pinned pendulum system following a miped discretization of lengths and
        positions.

        Parameters
        ----------
        p : (4, 1) numpy array
            State of the pinned pendulum system, consisting of the free endpoint's 2D position p, the length of the spring s, and the Lagrange multiplier l

        Returns
        -------
        H : (4, 4) numpy array
            Hessian of the energy of the pinned pendulum system following a miped discretization of lengths and positions

        """
        x = p[:2]
        s = p[2]
        l = p[3]

        pnorm = np.linalg.norm(x)
        pdir = x / pnorm

        z21 = np.zeros((2, 1))
        H = vstack([hstack([self.p.gamma * np.identity(2), z21, pdir]),
                    hstack([z21.T, self.p.mu * np.identity(1), -np.identity(1)]),
                    hstack([pdir.T, -np.identity(1), np.array([[0]])])])
        return H

    def step(self, x: np.ndarray, s: np.ndarray, l: np.ndarray):
        """
        Steps the simulation forward in time.

        Parameters
        ----------
        p : (4, 1) numpy array
            Raw state (free endpoint's 2D position) of the pinned pendulum system

        Returns
        -------
        p : (4, 1) numpy array
            Nept raw state (free endpoint's 2D position) of the pinned pendulum system
        """
        p = np.vstack([x, s, l])
        p0 = p.copy()
        p_next = self.solver.solve(p0)

        x = p_next[:2]
        s = p_next[2]
        l = p_next[3]

        return x, s, l


    def rest_state(self):
        x =  np.array([1., 0])[:, None]
        s = np.array([1.])
        l = np.array([0.])
        return x, s, l

