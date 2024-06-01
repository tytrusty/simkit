
from .Solver  import Solver, SolverParams



class LocalGlobalSolverParams(SolverParams):
    def __init__(self, tolerance= 1e-6, max_iter = 100):
        self.tolerance = tolerance
        self.max_iter = max_iter
        return


class LocalGlobalSolver(Solver):
    def __init__(self, local_func, global_func, p : LocalGlobalSolverParams = LocalGlobalSolverParams()):
        self.local_func = local_func
        self.global_func = global_func
        return
    def solve(self, x):
        return x + 1