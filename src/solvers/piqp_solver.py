import numpy as np
import scipy.sparse as sp
import time
import warnings
import importlib
from src.problems.qp_problem import QPProblem
from src.solvers.base_solver import BaseSolver

class PIQPSolver(BaseSolver):
    def __init__(self, verbose=True, eps=1e-6, use_multistage=True, isa=None, compute_timings=False):
        super().__init__()
        
        import piqp
        import piqp.instruction_set

        if isa == 'sse':
            module = importlib.import_module('piqp.piqp_python', 'piqp')
            self.solver = module.SparseSolver()
        elif isa == 'avx2':
            if piqp.instruction_set.avx2:
                module = importlib.import_module('piqp.piqp_python_avx2', 'piqp')
                self.solver = module.SparseSolver()
            else:
                warnings.warn('avx2 not supported, falling back to default')
                self.solver = piqp.SparseSolver()
        elif isa == 'avx512':
            if piqp.instruction_set.avx512:
                module = importlib.import_module('piqp.piqp_python_avx512', 'piqp')
                self.solver = module.SparseSolver()
            else:
                warnings.warn('avx512 not supported, falling back to default')
                self.solver = piqp.SparseSolver()
        else:
            self.solver = piqp.SparseSolver()

        self.solver.settings.eps_abs = eps
        self.solver.settings.eps_rel = eps
        self.solver.settings.verbose = verbose
        self.solver.settings.compute_timings = compute_timings
        if use_multistage:
            self.solver.settings.kkt_solver = piqp.KKTSolver.sparse_multistage
        
    def supports_problem(self, problem):
        return isinstance(problem, QPProblem)

    def setup(self, problem: QPProblem):
        self.problem = problem
        start = time.time()
        self.solver.setup(problem.P, problem.c, problem.Aeq, problem.beq, problem.Aineq, problem.bineq_lb, problem.bineq_ub, problem.xlb, problem.xub)
        self.stats['setup_time'] = time.time() - start
        self.stats['piqp_setup_time'] = self.solver.result.info.setup_time
        
    def solve(self):
        self.solver.update(x_l=self.problem.xlb, x_u=self.problem.xub)

        start = time.time()
        self.result = self.solver.solve()
        self.stats['solve_time'] = time.time() - start
        self.stats['iterations'] = self.solver.result.info.iter
        self.stats['piqp_solve_time'] = self.solver.result.info.solve_time
        self.stats['piqp_kkt_factor_time'] = self.solver.result.info.kkt_factor_time
        self.stats['piqp_kkt_solve_time'] = self.solver.result.info.kkt_solve_time
        
    def get_solution(self):
        return self.problem.get_solution_from_qp_solution(self.solver.result.x)
