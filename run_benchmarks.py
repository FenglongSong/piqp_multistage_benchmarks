import numpy as np
import json
from typing import Dict, Any

from src.problems.chain_mass_ocp_problem import ChainMassOCPProblem
from src.problems.chain_mass_scenario_problem import ChainMassScenarioProblem

from src.benchmark_runner import BenchmarkRunner

from src.plotter.runtime_log_plotter import RuntimeLogPlotter
from src.plotter.runtime_relative_plotter import RuntimeRelativePlotter
from src.plotter.speedup_heatmap_plotter import SpeedupHeatmapPlotter

def analyze_results(results):
    """Analyze and print summary of benchmark results with relative speedups and their ranges."""
    print("\nBenchmark Summary:")
    print("=" * 80)
    
    # Collect all solver IDs
    solver_ids = sorted(set(
        solver_id 
        for problem_results in results.values() 
        for solver_id in problem_results.keys()
    ))
    
    # First, calculate average solve times for each solver and collect all solve times
    solver_times = {solver_id: [] for solver_id in solver_ids}
    solver_avg_times = {}
    
    for solver_id in solver_ids:
        solver_results = []
        for problem_results in results.values():
            if solver_id in problem_results:
                solver_results.append(problem_results[solver_id])
                solver_times[solver_id].append(problem_results[solver_id]['solve_time']['mean'])
        
        avg_setup_time = np.mean([r['setup_time']['mean'] if not isinstance(r['setup_time'], float) else r['setup_time'] for r in solver_results])
        avg_solve_time = np.mean([r['solve_time']['mean'] for r in solver_results])
        avg_iterations = np.mean([r['iterations']['mean'] for r in solver_results])
        
        solver_avg_times[solver_id] = {
            'setup_time': avg_setup_time,
            'solve_time': avg_solve_time,
            'iterations': avg_iterations
        }
        
        print(f"\nSolver: {next(iter(results.values()))[solver_id]['solver_name']}")
        print("-" * 40)
        print(f"Average setup time: {avg_setup_time*1000:.2f}ms")
        print(f"Average solve time: {avg_solve_time*1000:.2f}ms")
        print(f"Average iterations: {avg_iterations:.1f}")

    # Calculate and print speedups with ranges
    print("\nRelative Speedups (solve time) - Format: avg [min, max]:")
    print("=" * 100)
    print("\nReference    │", end=" ")
    for solver_id in solver_ids:
        print(f"{solver_id:>23}", end=" │ ")
    print("\n" + "─" * 170)
    
    for ref_solver in solver_ids:
        print(f"{ref_solver:12} │", end=" ")
        ref_times = np.array(solver_times[ref_solver])
        
        for comp_solver in solver_ids:
            comp_times = np.array(solver_times[comp_solver])
            
            # Calculate speedups for each problem instance
            speedups = []
            for ref_t, comp_t in zip(ref_times, comp_times):
                speedups.append(ref_t / comp_t)
            
            avg_speedup = np.mean(speedups)
            min_speedup = np.min(speedups)
            max_speedup = np.max(speedups)
            
            print(f"{avg_speedup:6.2f} [{min_speedup:6.2f}, {max_speedup:6.2f}]", end=" │ ")
        print()
    
    # Find the fastest and slowest solvers
    fastest_solver = min(solver_avg_times.items(), key=lambda x: x[1]['solve_time'])
    slowest_solver = max(solver_avg_times.items(), key=lambda x: x[1]['solve_time'])
    
    print("\nPerformance Summary:")
    print("=" * 80)
    print(f"Fastest solver: {fastest_solver[0]} ({fastest_solver[1]['solve_time']*1000:.2f}ms)")
    print(f"Slowest solver: {slowest_solver[0]} ({slowest_solver[1]['solve_time']*1000:.2f}ms)")
    
    # Calculate overall speedup range
    speedups = []
    for ref_t, comp_t in zip(solver_times[slowest_solver[0]], solver_times[fastest_solver[0]]):
        speedups.append(ref_t / comp_t)
    avg_speedup = np.mean(speedups)
    min_speedup = np.min(speedups)
    max_speedup = np.max(speedups)
    
    print(f"Maximum speedup: {avg_speedup:.2f}x [{min_speedup:.2f}x, {max_speedup:.2f}x] "
          f"({slowest_solver[0]} → {fastest_solver[0]})")
    
def load_benchmark_results(file_path: str) -> Dict[str, Any]:
    """
    Load benchmark results from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing benchmark results

    Returns:
        Dict[str, Any]: Dictionary containing the benchmark results

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract just the results part if it's in the new format
        if isinstance(data, dict) and 'results' in data:
            print(f"Loaded benchmark results from {file_path}")
            print("Metadata:")
            for key, value in data['metadata'].items():
                print(f"  {key}: {value}")
            return data['results']
        
        # If it's in the old format (just results without metadata)
        print(f"Loaded benchmark results from {file_path}")
        return data

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        raise
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON")
        raise


if __name__ == '__main__':
    # For ChainMassOCPProblem
    params = {
        'M': [2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70],
        'N': [15],
        'use_u_diff_cost': [False],
        'use_u_diff_constr': [False],
    }
    # runner = BenchmarkRunner(ChainMassOCPProblem, params, runs=30, name='M2-70_N15_default')
    # results_default = runner.run()
    results_default = load_benchmark_results('results/benchmark_M2-70_N15_default_20250304_121610.json')
    analyze_results(results_default)

    plotter = RuntimeLogPlotter(results_default)
    plotter.plot('M', save_path='results/benchmark_M2-70_N15_default.pdf')
    
    params['use_u_diff_cost'] = [True]
    # runner = BenchmarkRunner(ChainMassOCPProblem, params, runs=30, name='M2-70_N15_cost_diff')
    # results_cost_diff = runner.run()
    results_cost_diff = load_benchmark_results('results/benchmark_M2-70_N15_cost_diff_20250304_125958.json')
    analyze_results(results_cost_diff)

    plotter = RuntimeRelativePlotter(results_cost_diff)
    plotter.plot('M', save_path='results/benchmark_M2-70_N15_cost_diff.pdf')

    # For ChainMassScenarioProblem
    scenario_params = {
        'M': [2, 5, 10, 20, 50],
        'Ns': [1, 5, 10, 15, 20],
        'N': [15],
    }
    # runner = BenchmarkRunner(ChainMassScenarioProblem, scenario_params, runs=30, name='test', solver_list=['piqp_sparse', 'piqp_avx2'], compute_timings=True)
    # results = runner.run()
    results = load_benchmark_results('results/benchmark_ChainMassScenarioProblem_test_20250812_160505.json')
    analyze_results(results)

    # Create speedup heatmap
    heatmap_plotter = SpeedupHeatmapPlotter(results)
    heatmap_plotter.plot('piqp_sparse', 'piqp_avx2', save_path='results/scenario_speedup_heatmap.pdf')
