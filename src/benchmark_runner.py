import itertools
import json
import platform
from datetime import datetime
from pathlib import Path
from src.benchmark import Benchmark
from src.solvers.piqp_solver import PIQPSolver
# from src.solvers.hpipm_solver import HPIPMSolver
# from src.solvers.qpalm_solver import QPALMSolver
# from src.solvers.osqp_solver import OSQPSolver


class BenchmarkRunner:
    def __init__(self, problem_class, params, runs=100, verbose=False, eps=1e-6, name='default', solver_list=None, compute_timings=False):
        self.problem_class = problem_class
        self.params = params
        self.runs = runs
        self.verbose = verbose
        self.eps = eps
        self.name = name
        self.compute_timings = compute_timings
        self.solvers = self._get_compatible_solvers()
        
        if solver_list is not None:
            self.solvers = {key: self.solvers[key] for key in self.solvers.keys() if key in solver_list}

    def _create_solvers(self):
        machine = platform.machine().lower()
        if machine in ['x86_64', 'amd64']:
            piqp_multistage_variants = {
                'piqp_sse': PIQPSolver(
                    verbose=self.verbose, 
                    eps=self.eps, 
                    isa='sse',
                    compute_timings=self.compute_timings
                ),
                'piqp_avx2': PIQPSolver(
                    verbose=self.verbose, 
                    eps=self.eps, 
                    isa='avx2',
                    compute_timings=self.compute_timings
                ),
                'piqp_sse_p': PIQPSolver(
                    verbose=self.verbose, 
                    eps=self.eps, 
                    isa='sse',
                    compute_timings=self.compute_timings,
                    parallel=True
                ),
                'piqp_avx2_p': PIQPSolver(
                    verbose=self.verbose, 
                    eps=self.eps, 
                    isa='avx2',
                    compute_timings=self.compute_timings,
                    parallel=True
                ),
                # 'piqp_avx512': PIQPSolver(
                #     verbose=self.verbose, 
                #     eps=self.eps, 
                #     isa='avx512'
                # ),
            }
        else:
            piqp_multistage_variants = {
                'piqp_block': PIQPSolver(
                    verbose=self.verbose, 
                    eps=self.eps,
                    compute_timings=self.compute_timings
                ),
                'piqp_block_p': PIQPSolver(
                    verbose=self.verbose, 
                    eps=self.eps,
                    compute_timings=self.compute_timings,
                    parallel=True
                ),
            }

        return {
            # 'piqp_sparse': PIQPSolver(
            #     verbose=self.verbose, 
            #     eps=self.eps,
            #     use_multistage=False,
            #     compute_timings=self.compute_timings
            # ),
            **piqp_multistage_variants,
            # 'hpipm': HPIPMSolver(
            #     verbose=self.verbose, 
            #     eps=self.eps
            # ),
            # 'qpalm': QPALMSolver(
            #     verbose=self.verbose, 
            #     eps=self.eps
            # ),
            # 'osqp': OSQPSolver(
            #     verbose=self.verbose, 
            #     eps=self.eps
            # ),
        }

    def _get_compatible_solvers(self):
        """Get solvers that support the problem type"""
        all_solvers = self._create_solvers()
        
        # Create test problem with first set of parameters
        first_params = {k: v[0] for k, v in self.params.items()}
        test_problem = self.problem_class(**first_params)
        
        return {
            solver_id: solver 
            for solver_id, solver in all_solvers.items() 
            if solver.supports_problem(test_problem)
        }

    def _create_problem_key(self, params):
        """Create a unique key for the parameter combination"""
        return '_'.join(f"{k}{v}" for k, v in params.items())

    def run(self):
        """Run the benchmarks"""
        if not self.solvers:
            raise ValueError(f"No compatible solvers found for {self.problem_class.__name__}")

        # Create all parameter combinations
        param_combinations = [
            dict(zip(self.params.keys(), v)) 
            for v in itertools.product(*self.params.values())
        ]
        
        results = {}
        Path('results').mkdir(exist_ok=True)
        
        total_combinations = len(param_combinations) * len(self.solvers)
        current_combination = 0
        
        for params in param_combinations:
            problem = self.problem_class(**params)
            problem_key = self._create_problem_key(params)
            results[problem_key] = {}
            
            print(f"\nTesting parameter combination: {params}")
            
            for solver_id, solver in self.solvers.items():
                current_combination += 1
                print(f"\nProgress: {current_combination}/{total_combinations}")
                print(f"Testing solver: {solver_id}")
                
                benchmark = Benchmark(problem, solver)
                result = benchmark.run(self.runs)
                results[problem_key][solver_id] = {
                    'solver_name': solver_id,
                    **result
                }
                
                self._print_stats(result)
        
        self._save_results(results)
        return results

    def _print_stats(self, result):
        """Print statistics for a benchmark run"""
        stats = result['solve_time']
        print(f"Average solve time: {stats['mean']*1000:.2f}ms ± {stats['std']*1000:.2f}ms")
        stats = result['iterations']
        print(f"Average iterations: {stats['mean']:.2f} ± {stats['std']:.2f}")

    def _save_results(self, results):
        """Save results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f'results/benchmark_{self.problem_class.__name__}_{self.name}_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump({
                'metadata': {
                    'problem_class': self.problem_class.__name__,
                    'name': self.name,
                    'timestamp': timestamp,
                    'runs': self.runs,
                    'eps': self.eps,
                    'parameters': self.params
                },
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {result_file}")
