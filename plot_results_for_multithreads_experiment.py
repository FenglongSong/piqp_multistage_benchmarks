import numpy as np
import matplotlib.pyplot as plt

from run_benchmarks import load_benchmark_results

def extract(results_files):
    all_num_threads = []
    for file_name in results_files:
        parts = file_name.split('_')
        # Find the part that starts with T
        for part in parts:
            if part.startswith('T'):
                # Extract the numeric value after the prefix
                value = int(part[1])
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
                times.append(problem_results[solver_id][field]['mean'])
                times_std.append(problem_results[solver_id][field]['std'])
                break

    return times, times_std


def main():
    fontsize = 18
    fields = ['solve_time', 'kkt_factor_time', 'kkt_solve_time']
    title_fields = {'solve_time': 'Solve Time',
                    'kkt_factor_time': 'KKT Factor Time',
                    'kkt_solve_time': 'KKT Solve Time'}
    # parallel_solver_labels = {'piqp_sse_p': 'PIQP (block SSE par)',
    #                  'piqp_avx2_p': 'PIQP (block AVX2 par)',}
    # sequential_solver_labels = {'piqp_sse': 'PIQP (block SSE)',
    #                  'piqp_avx2': 'PIQP (block AVX2)',}

    parallel_solver_labels = {'piqp_sse_p': 'PIQP (block SSE par)',}
    sequential_solver_labels = {'piqp_sse': 'PIQP (block SSE seq)',}

    parallel_solver_labels = {'piqp_avx2_p': 'PIQP (block AVX2 par)',}
    sequential_solver_labels = {'piqp_avx2': 'PIQP (block AVX2 seq)',}
    
    sequential_result_file = 'results/benchmark_ChainMassOCPProblem_M20_N20-200_T0.json'
    # parallel_results_files = ['benchmark_ChainMassOCPProblem_M20_N20-200_T2.json',
    #                  'benchmark_ChainMassOCPProblem_M20_N20-200_T3.json',
    #                  'benchmark_ChainMassOCPProblem_M20_N20-200_T4.json',
    #                  ]
    parallel_results_files = ['benchmark_ChainMassOCPProblem_M20_N20-200_T2_20251008_135727.json',
                     'benchmark_ChainMassOCPProblem_M20_N20-200_T3_20251008_140140.json',
                     'benchmark_ChainMassOCPProblem_M20_N20-200_T4_20251008_140540.json',
                     ]
    parallel_results_files = ['results/' + result for result in parallel_results_files]

    
    all_num_threads = extract(parallel_results_files)

    for field in fields:
        fig, ax = plt.subplots(figsize=(8, 4))

        # plot for sequential
        results = load_benchmark_results(sequential_result_file)
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

        for solver_id in sequential_solver_labels.keys():
            times, times_std = _collect_solver_data(results, solver_id, param_values, 'N', field)
            times = np.array(times)
            times_std = np.array(times_std)

            if len(times) == 0:
                continue

            plt.plot(param_values, times, '-', marker='o',
                        label=sequential_solver_labels[solver_id],
                        color='k'
                        )

            plt.fill_between(param_values, times - times_std, times + times_std,
                                alpha=0.3, color='k'
                                )


        # plot for parallel
        filename_with_T_threads = None
        for num_threads in all_num_threads:
            for file_name in parallel_results_files:
                if 'T'+str(num_threads) in file_name:
                    filename_with_T_threads = file_name
                    break


            results = load_benchmark_results(filename_with_T_threads)

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

            for solver_id in parallel_solver_labels.keys():
                times, times_std = _collect_solver_data(results, solver_id, param_values, 'N', field)
                times = np.array(times)
                times_std = np.array(times_std)

                if len(times) == 0:
                    continue

                label = rf"{parallel_solver_labels[solver_id]}  $p={num_threads}$"
                plt.plot(param_values, times, '-', marker='o', label=label)

                plt.fill_between(param_values, times - times_std, times + times_std,
                                 alpha=0.3
                                 )
        
        ax.set_xlabel(r'Horizon $N$', fontsize=fontsize)
        ax.set_ylabel(rf'CPU Time [s]', fontsize=fontsize)
        ax.set_title(rf'Benchmark Results: {title_fields[field]}')
        ax.legend()
        ax.grid(True)

        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        
        plt.savefig(f'benchmark_{field}_T{min(all_num_threads)}-{max(all_num_threads)}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'benchmark_{field}_T{min(all_num_threads)}-{max(all_num_threads)}.pdf', bbox_inches='tight')
        plt.close()




if __name__ == '__main__':
    main()