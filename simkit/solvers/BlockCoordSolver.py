import scipy as sp
import numpy as np

from .Solver import Solver, SolverParams
from ..backtracking_line_search import backtracking_line_search


class BlockCoordSolverParams(SolverParams):
    def __init__(self, tolerance= 1e-6, max_iter = 1, do_line_search = True):
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.do_line_search = do_line_search
        return

class BlockCoordSolver(Solver):

    def __init__(self, global_step_func, local_step_func, p : BlockCoordSolverParams = BlockCoordSolverParams()):
        self.p = p
        self.global_step_func = global_step_func
        self.local_step_func = local_step_func
        pass


    def solve(self, x0):
        
        x = x0.copy()
        for _i in range(self.p.max_iter):
            
            x_prev = x.copy()
            r = self.local_step_func(x)
            x = self.global_step_func(x, r)

            delta = (x - x_prev)
            if np.linalg.norm(delta) < self.p.tolerance:
                break

        return x
    

