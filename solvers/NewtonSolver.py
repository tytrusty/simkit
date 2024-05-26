import scipy as sp

from .Solver import Solver, SolverParams



class NewtonSolverParams(SolverParams):
    def __init__(self, tolerance= 1e-6, max_iter = 100):
        self.tolerance = tolerance
        self.max_iter = max_iter
        return

class NewtonSolver(Solver):

    def __init__(self, gradient_func, hessian_func):
        self.gradient_func = gradient_func
        self.hessian_func = hessian_func
        pass


    def solve(self, x):
        g = self.gradient_func(x)
        H = self.hessian_func(x)

        dx = sp.linalg.solve(H, -g)

        return dx