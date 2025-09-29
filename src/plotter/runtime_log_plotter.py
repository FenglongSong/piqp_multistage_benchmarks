import numpy as np
import matplotlib.pyplot as plt

from src.plotter.base_plotter import BasePlotter

class RuntimeLogPlotter(BasePlotter):

    def plot(self, x_param, save_path=None, fig_width=5.5):
        """Plot benchmark results with specified parameter on x-axis"""
        fig_height = fig_width * 0.66
        plt.figure(figsize=(fig_width, fig_height))
        
        plt.grid(True, which="major", ls="-", alpha=0.2)
        plt.grid(True, which="minor", ls=":", alpha=0.2)
        # plt.yscale('log')
        
        param_values = []
        for key in self.results.keys():
            parts = key.split('_')
            # Find the part that starts with x_param
            for part in parts:
                if part.startswith(x_param):
                    # Extract the numeric value after the prefix
                    value = int(part[len(x_param):])
                    param_values.append(value)
                    break
        param_values = sorted(set(param_values))
        
        self._plot_solver_results(param_values, x_param)
        self._customize_plot(x_param, param_values)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

    def _plot_solver_results(self, param_values, x_param):
        """Plot results for each solver"""
        available_solvers = next(iter(self.results.values())).keys()
        # solver_order = ['hpipm', 'qpalm', 'osqp', 'piqp_block', 'piqp_sse', 'piqp_avx2', 'piqp_avx512', 'piqp_sparse']
        solver_order = ['piqp_block', 'piqp_block_p']
        # FOR x86 platform
        solver_order = ['piqp_sse', 'piqp_avx2', 'piqp_sse_p', 'piqp_avx2_p']
        ordered_solvers = [solver for solver in solver_order if solver in available_solvers]
        for solver_id in ordered_solvers:
            times, times_std = self._collect_solver_data(solver_id, param_values, x_param)
            
            x = np.array(param_values)
            y = np.array(times)
            std = np.array(times_std)
            
            self._plot_solver_line(x, y, std, solver_id)

    def _collect_solver_data(self, solver_id, param_values, x_param):
        """Collect timing data for a specific solver"""
        times = []
        times_std = []
        
        for param_val in param_values:
            # Find matching result
            for problem_key, problem_results in self.results.items():
                if f"{x_param}{param_val}" in problem_key:
                    times.append(problem_results[solver_id]['solve_time']['mean'])
                    times_std.append(problem_results[solver_id]['solve_time']['std'])
                    break
        
        return times, times_std

    def _plot_solver_line(self, x, y, std, solver_id):
        """Plot a single solver's results with confidence band"""
        color = self.colors.get(solver_id, '#333333')
        
        plt.plot(x, y, '-', marker='o', 
                label=self.labels.get(solver_id, solver_id),
                color=color)
        
        plt.fill_between(x, y - std, y + std, 
                        alpha=0.3,
                        color=color)

    def _customize_plot(self, x_param, param_values):
        """Customize plot appearance"""
        param_labels = {
            'M': 'Number of masses $M$',
            'Ns': 'Number of scenarios $N_s$',
            'N': 'Horizon length $N$'
        }
        
        plt.xlabel(param_labels.get(x_param, x_param), fontsize=14)
        plt.ylabel('Average solver run time [s]', fontsize=14)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = range(len(labels))
        plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower right', ncol=2, columnspacing=0.5, borderaxespad=0.2, handletextpad=0.5) 
        # plt.legend(loc='lower right', ncol=2)

        plt.xlim(min(param_values), max(param_values))
        plt.minorticks_on()
        plt.tight_layout()
