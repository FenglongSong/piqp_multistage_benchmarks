import time
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from src.solvers.base_solver import BaseSolver

@dataclass
class BenchmarkStatistics:
    mean: float
    std: float
    median: float
    min: float
    max: float
    samples: List[float]
    
    @classmethod
    def from_samples(cls, samples):
        return cls(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            median=float(np.median(samples)),
            min=float(np.min(samples)),
            max=float(np.max(samples)),
            samples=[float(s) for s in samples]
        )
    
    def to_dict(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'min': self.min,
            'max': self.max,
            'samples': self.samples
        }

class Benchmark:
    def __init__(self, problem, solver: BaseSolver):
        self.problem = problem
        self.solver = solver
        
    def run(self, runs: int = 100) -> Dict[str, Any]:
        self.solver.setup(self.problem)
        self.solver.solve()

        stats_samples = {}
        for key in self.solver.stats.keys():
            stats_samples[key] = []
        
        # Run multiple solves
        np.random.seed(42)
        for _ in range(runs):
            self.problem.randomize_x0()
            self.solver.solve()
            
            for key, value in self.solver.stats.items():
                stats_samples[key].append(value)
        
        # Compute statistics
        result = {}
        for key, samples in stats_samples.items():
            stats = BenchmarkStatistics.from_samples(samples)
            result[key] = stats.to_dict()
        
        return result
