import numpy as np
import matplotlib.pyplot as plt

from run_benchmarks import load_benchmark_results


def compute_theory_speedup(N, P):
    if N < 2*P:
        return np.nan, np.nan

    ratio = 19/7
    # N1/Ni cannot be perfectly the optimal ratio, we have to decide how to round out, either floor Ni or ceil Ni
    # therefore we compare the time complexity of parallel part
    Ni_ceil =  np.ceil((N - P + 1) / (P - 1 + ratio))
    Ni_floor = np.floor((N - P + 1) / (P - 1 + ratio))
    def factor_cost_parallel_part(N, Ni, P):
        N1 = N - (P - 1) - Ni * (P - 1)
        return np.maximum(7/3*N1 - 1, 19/3*Ni - 1)
    def factor_cost_serial_part(P):
        return 1/3 * (10*P - 16)
    def solve_cost_parallel_part(N, Ni, P):
        N1 = N - (P - 1) - Ni * (P - 1)
        return np.maximum(5*N1 - 2, 9*Ni - 2)
    def solve_cost_serial_part(P):
        return 7*P - 11

    # Compute both costs elementwise. tc means time complexity
    tc_floor = factor_cost_parallel_part(N, Ni_floor, P)
    tc_ceil = factor_cost_parallel_part(N, Ni_ceil, P)
    Ni = Ni_floor if tc_floor < tc_ceil else Ni_ceil
    N1_mesh = N - (P - 1) - Ni * (P - 1)

    factor_cost_total = factor_cost_parallel_part(N, Ni, P) + factor_cost_serial_part(P)
    factor_cost_serial = 7/3*N - 2
    factor_speedup = (factor_cost_serial) / factor_cost_total

    solve_cost_total = solve_cost_parallel_part(N, Ni, P) + solve_cost_serial_part(P)
    solve_cost_serial = 5*N - 4
    solve_speedup = (solve_cost_serial) / solve_cost_total

    return factor_speedup, solve_speedup
    

def extract(results_files):
    all_num_threads = []
    for file_name in results_files:
        parts = file_name.split('_')
        # Find the part that starts with T
        for part in parts:
            if part.startswith('T'):
                # Extract the numeric value after the prefix
                value = int(part[1:len(part)])
                all_num_threads.append(value)
                break

    all_num_threads = sorted(set(all_num_threads))
    return all_num_threads


def _collect_solver_data(results, solver_id, param_values, x_param, field):
    """Collect timing data for a specific solver"""
    times = []
    times_std = []

    if field in ['kkt_factor_time', 'kkt_solve_time']:
        field = 'piqp_' + field

    for param_val in param_values:
        # Find matching result
        for problem_key, problem_results in results.items():
            if f"{x_param}{param_val}" in problem_key:
                if solver_id not in problem_results.keys():
                    break
                if field not in problem_results[solver_id]:
                    break
                times.append(problem_results[solver_id][field]['mean'])
                times_std.append(problem_results[solver_id][field]['std'])
                break

    return times, times_std


def main():
    fontsize = 18
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    # fields = ['solve_time', 'kkt_factor_time', 'kkt_solve_time']
    fields = ['kkt_factor_time', 'kkt_solve_time', 'solve_time']
    title_fields = {'solve_time': 'Solve Time',
                    'kkt_factor_time': 'KKT Factor Time',
                    'kkt_solve_time': 'KKT Solve Time'}
    solver_labels = {'piqp_avx2': 'PIQP (block AVX2 seq)', 'piqp_avx2_p': 'PIQP (block AVX2 par)',}
    
    sequential_result_file = 'results/benchmark_ChainMassOCPProblem_M20_N20-200_T0_20251008_164532.json'

    parallel_results_files = ['benchmark_ChainMassOCPProblem_M20_N20-200_T2_20251008_161544.json',
                     'benchmark_ChainMassOCPProblem_M20_N20-200_T4_20251008_161826.json',
                     'benchmark_ChainMassOCPProblem_M20_N20-200_T6_20251008_162057.json',
                     'benchmark_ChainMassOCPProblem_M20_N20-200_T8_20251008_162320.json',
                     'benchmark_ChainMassOCPProblem_M20_N40-200_T10_20251008_164253.json',
                     'benchmark_ChainMassOCPProblem_M20_N40-200_T12_20251008_162835.json',
                     ]
    parallel_results_files = ['results/' + result for result in parallel_results_files]
    
    all_num_threads = extract(parallel_results_files)

    def _plot(results, num_threads = 0):
        param_values = []
        for key in results.keys():
            parts = key.split('_')
            # Find the part that starts with x_param
            for part in parts:
                if part.startswith('N'):
                    # Extract the numeric value after the prefix
                    value = int(part[len('N'):])
                    param_values.append(value)
                    break
        param_values = sorted(set(param_values))

        for solver_id in solver_labels.keys():
            if not solver_id.endswith('_p') and num_threads >= 2:  # if the solver is a sequential solver but num_threads >= 2, doesn't make sense, pass
                continue
            if solver_id.endswith('_p') and num_threads < 2:  # if the solver is a parallel solver but num_threads < 2, doesn't make sense, pass
                continue
            times, times_std = _collect_solver_data(results, solver_id, param_values, 'N', field)
            times = np.array(times)
            times_std = np.array(times_std)
            if len(times) == 0:
                continue

            label = rf"{solver_labels[solver_id]}"
            if num_threads > 1: label += rf' $p={num_threads}$'
            plt.plot(param_values, times, '-', marker='o', label=label)
            plt.fill_between(param_values, times - times_std, times + times_std, alpha=0.3)


    for field in fields:
        fig, ax = plt.subplots(figsize=(8, 4))

        # plot for sequential
        results = load_benchmark_results(sequential_result_file)
        _plot(results, 0)

        # plot for parallel
        filename_with_T_threads = None
        for num_threads in all_num_threads:
            for file_name in parallel_results_files:
                if 'T'+str(num_threads) in file_name:
                    filename_with_T_threads = file_name
                    break

            results = load_benchmark_results(filename_with_T_threads)
            _plot(results, num_threads)
        
        ax.set_xlabel(r'Horizon $N$', fontsize=fontsize)
        ax.set_ylabel(rf'CPU Time [s]', fontsize=fontsize)
        ax.set_title(rf'Benchmark Results: {title_fields[field]}', fontsize=fontsize)
        ax.legend()
        ax.grid(True)
        
        plt.savefig(f'benchmark_{field}_T{min(all_num_threads)}-{max(all_num_threads)}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'benchmark_{field}_T{min(all_num_threads)}-{max(all_num_threads)}.pdf', bbox_inches='tight')
        plt.close()




if __name__ == '__main__':
    main()