import numpy as np
import matplotlib.pyplot as plt

from src.plotter.base_plotter import BasePlotter

class RuntimeRelativePlotter(BasePlotter):

    def plot(self, x_param, save_path=None, fig_width=5.5):
        """Plot benchmark results as a bar chart showing percentage difference from HPIPM for each mass"""
        fig_height = fig_width * 0.66
        plt.figure(figsize=(fig_width, fig_height))
        
        # Get parameter values
        param_values = []
        for key in self.results.keys():
            parts = key.split('_')
            for part in parts:
                if part.startswith(x_param):
                    value = int(part[len(x_param):])
                    param_values.append(value)
                    break
        param_values = sorted(set(param_values))
        
        # Get available solvers
        available_solvers = next(iter(self.results.values())).keys()
        # solver_order = ['hpipm', 'qpalm', 'osqp', 'piqp_block', 'piqp_sse', 'piqp_avx2', 'piqp_avx512', 'piqp_sparse']
        solver_order = ['hpipm', 'piqp_sparse', 'piqp_block', 'piqp_sse', 'piqp_avx2', 'piqp_avx512']
        ordered_solvers = [solver for solver in solver_order if solver in available_solvers]
        
        # Remove HPIPM from solvers as it's our baseline
        ordered_solvers.remove('hpipm')
        
        # Number of groups (param values) and bars per group (solvers)
        n_groups = len(param_values)
        n_solvers = len(ordered_solvers)
        
        # Set up the bar positions
        group_spacing = 0.2  # Increase this value to create bigger gaps between groups
        bar_width = (1.0 - group_spacing) / n_solvers  # Reduced width to make room for gaps
        index = np.arange(n_groups) * (1 + group_spacing)  # Multiply by (1 + spacing) to spread out groups
        
        # Store all percentage differences for y-axis scaling
        all_percentages = []
        
        # Plot bars for each solver
        for i, solver in enumerate(ordered_solvers):
            percentage_diff = []
            
            for param_val in param_values:
                # Find the problem with this parameter value
                for problem_key, problem_results in self.results.items():
                    if f"{x_param}{param_val}" in problem_key:
                        # Get HPIPM baseline
                        hpipm_time = problem_results['hpipm']['solve_time']['mean']
                        
                        # Get solver performance
                        mean_time = problem_results[solver]['solve_time']['mean']
                        
                        # Calculate percentage speedup/slowdown
                        # Positive now means faster than HPIPM (improvement)
                        # Negative means slower than HPIPM
                        perc_diff = (hpipm_time / mean_time - 1) * 100
                        
                        # Clip very slow performances at -100%
                        perc_diff = max(perc_diff, -70)
                        
                        percentage_diff.append(perc_diff)
                        all_percentages.append(perc_diff)
                        break
            
            # Create bars for this solver
            position = index + i * bar_width - (n_solvers-1) * bar_width/2
            bars = plt.bar(position, percentage_diff,
                        bar_width,
                        label=self.labels[solver],
                        color=self.colors[solver])
            
            # Add value labels on top of bars
            for j, bar in enumerate(bars):
                height = percentage_diff[j]
                
                # Adjust label position based on bar direction
                if height > 0:
                    label_pos = height + 0.8
                else:
                    label_pos = height - 0.8
                    
                if height == -70:
                    label_text = ""
                else:
                    label_text = f"{'+' if height > 0 else ''}{height:.0f}\%"
                
                x_pos = bar.get_x() + bar_width/2
                if i == 0 and j == 0:
                    x_pos -= 0.2
                if i == 2 and j == 0:
                    x_pos += 0.2
                if i == 1 and j > 0:
                    x_pos += 0.33

                plt.text(x_pos, label_pos,
                        label_text,
                        ha='center', 
                        va='bottom' if height > 0 else 'top',
                        fontsize=8)
        
        # Customize plot
        plt.ylabel('Relative Speed-up to HPIPM (\%)\n← slower | faster →', fontsize=14)
        plt.xlabel('Number of masses $M$', fontsize=14)
        plt.xticks(index, param_values)
        
        # Add horizontal line at y=0 (HPIPM baseline)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        plt.grid(True, which="major", ls="-", alpha=0.2, axis='y')
        # handles, labels = plt.gca().get_legend_handles_labels()
        # order = [0, 1, 2] 
        # plt.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left')
        plt.legend(loc='upper left', borderaxespad=0.2, handletextpad=0.5)
        
        # Adjust y-axis limits to make room for labels
        y_min = -70  # Give some space below -100%
        y_max = 70  # Add 20% padding above highest bar
        plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
