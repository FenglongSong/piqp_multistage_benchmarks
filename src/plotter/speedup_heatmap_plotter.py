import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from src.plotter.base_plotter import BasePlotter

class SpeedupHeatmapPlotter(BasePlotter):

    def _extract_param_values(self, param):
        """Extract unique values for a given parameter"""
        return sorted(set(
            int(val) for key in self.results.keys()
            for param_val in key.split('_')
            if param_val.startswith(param)
            for val in [param_val[len(param):]]
        ))

    def _get_solver_time(self, M, Ns, solver_id, time_type='solve_time'):
        """Get solve time for specific M, Ns combination"""
        key = f"M{M}_Ns{Ns}_N15"
        return self.results[key][solver_id][time_type]['mean']
    
    def plot(self, solver1_id, solver2_id, save_path=None, fig_width=10):
        """
        Plot grouped stacked bars in a single line showing time breakdown.
        Each experiment has two bars (one per solver) grouped together.
        Normalized based on solver1 time. Uses hatches to distinguish solvers.
        """
        M_values = self._extract_param_values('M')
        Ns_values = self._extract_param_values('Ns')
        
        # Create experiment combinations and labels with M grouping
        experiments = []
        exp_labels = []
        x_positions = []
        current_pos = 0
        
        for i, M in enumerate(M_values):
            if i > 0:  # Add extra spacing between M groups
                current_pos += 0.4
            
            for j, Ns in enumerate(Ns_values):
                experiments.append((M, Ns))
                exp_labels.append(rf'$\begin{{array}}{{c}} {M} \\ {Ns} \end{{array}}$')
                x_positions.append(current_pos)
                current_pos += 0.8  # Regular spacing within M groups
        
        n_experiments = len(experiments)
        
        # Collect data for all experiments
        solver1_data = {'total': [], 'factor': [], 'solve': [], 'rest': []}
        solver2_data = {'total': [], 'factor': [], 'solve': [], 'rest': []}
        
        for M, Ns in experiments:
            # Solver 1 times
            s1_total = self._get_solver_time(M, Ns, solver1_id, 'solve_time')
            s1_factor = self._get_solver_time(M, Ns, solver1_id, 'piqp_kkt_factor_time')
            s1_solve = self._get_solver_time(M, Ns, solver1_id, 'piqp_kkt_solve_time')
            s1_rest = s1_total - s1_factor - s1_solve
            
            solver1_data['total'].append(s1_total)
            solver1_data['factor'].append(s1_factor)
            solver1_data['solve'].append(s1_solve)
            solver1_data['rest'].append(s1_rest)
            
            # Solver 2 times
            s2_total = self._get_solver_time(M, Ns, solver2_id, 'solve_time')
            s2_factor = self._get_solver_time(M, Ns, solver2_id, 'piqp_kkt_factor_time')
            s2_solve = self._get_solver_time(M, Ns, solver2_id, 'piqp_kkt_solve_time')
            s2_rest = s2_total - s2_factor - s2_solve
            
            solver2_data['total'].append(s2_total)
            solver2_data['factor'].append(s2_factor)
            solver2_data['solve'].append(s2_solve)
            solver2_data['rest'].append(s2_rest)
        
        # Convert to numpy arrays
        for key in solver1_data:
            solver1_data[key] = np.array(solver1_data[key])
            solver2_data[key] = np.array(solver2_data[key])
        
        # Create figure
        fig_height = fig_width * 0.4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Define colors for the three components
        factor_color = '#1f77b4'
        solve_color = '#ff7f0e'
        rest_color = '#2ca02c'
        
        # Define hatches for the two solvers
        solver1_hatch = ''      # No hatch (solid)
        solver2_hatch = '///'     # Diagonal lines
        
        bar_width = 0.35
        
        # Plot stacked bars for each experiment
        for i in range(n_experiments):
            x_pos = x_positions[i]
            
            # Normalize based on solver1 total time
            s1_total_time = solver1_data['total'][i]
            
            # Normalize solver1 heights (will always sum to 1.0)
            s1_factor_norm = solver1_data['factor'][i] / s1_total_time
            s1_solve_norm = solver1_data['solve'][i] / s1_total_time
            s1_rest_norm = solver1_data['rest'][i] / s1_total_time
            
            # Normalize solver2 heights based on solver1 total time
            s2_factor_norm = solver2_data['factor'][i] / s1_total_time
            s2_solve_norm = solver2_data['solve'][i] / s1_total_time
            s2_rest_norm = solver2_data['rest'][i] / s1_total_time
            
            # Solver 1 bar (left) with solid colors
            s1_x = x_pos - bar_width/2
            ax.bar(s1_x, s1_rest_norm, bar_width, color=rest_color,
                   edgecolor='white', linewidth=0.5, hatch=solver1_hatch,
                   label='Other' if i == 0 else "")
            ax.bar(s1_x, s1_solve_norm, bar_width, bottom=s1_rest_norm, color=solve_color,
                   edgecolor='white', linewidth=0.5, hatch=solver1_hatch,
                   label='Solve' if i == 0 else "")
            ax.bar(s1_x, s1_factor_norm, bar_width, bottom=s1_rest_norm + s1_solve_norm, 
                   color=factor_color, edgecolor='white', linewidth=0.5, hatch=solver1_hatch,
                   label='Factor' if i == 0 else "")
            
            # Solver 2 bar (right) with hatched colors
            s2_x = x_pos + bar_width/2
            ax.bar(s2_x, s2_rest_norm, bar_width, color=rest_color,
                   edgecolor='white', linewidth=0.5, hatch=solver2_hatch)
            ax.bar(s2_x, s2_solve_norm, bar_width, bottom=s2_rest_norm, color=solve_color,
                   edgecolor='white', linewidth=0.5, hatch=solver2_hatch)
            ax.bar(s2_x, s2_factor_norm, bar_width, bottom=s2_rest_norm + s2_solve_norm,
                   color=factor_color, edgecolor='white', linewidth=0.5, hatch=solver2_hatch)
            
            # Add waterfall-style speedup indicators
            s1_total_height = 1.0  # Always 1.0 since normalized by solver1
            s2_total_height = s2_factor_norm + s2_solve_norm + s2_rest_norm
            speedup = solver1_data['total'][i] / solver2_data['total'][i]
            
            # Determine which bar is lower and higher
            if s1_total_height < s2_total_height:
                lower_height = s1_total_height
                higher_height = s2_total_height
                lower_x = s1_x
                higher_x = s2_x
            else:
                lower_height = s2_total_height
                higher_height = s1_total_height
                lower_x = s2_x
                higher_x = s1_x
            
            # Draw horizontal dashed line from lower bar top to above higher bar
            if s1_total_height < s2_total_height:
                line_start_x = lower_x - bar_width / 2
                line_end_x = higher_x + bar_width / 2
            else:
                line_start_x = lower_x + bar_width / 2
                line_end_x = higher_x - bar_width / 2
            line_y = higher_height
            
            ax.plot([line_start_x, line_end_x], [line_y, line_y], 
                   'k--', alpha=0.6, linewidth=0.8)
            
            # Draw vertical arrow centered above the lower bar
            arrow_x = lower_x  # Centered on the lower bar
            ax.annotate('', xy=(arrow_x, lower_height), xytext=(arrow_x, line_y),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1))
            
            # Add speedup text above the arrow
            text_y = line_y + 0.01
            ax.text((line_start_x + line_end_x) / 2, text_y, f'{speedup:.1f}×',
                   ha='center', va='bottom', fontsize=8, weight='bold', 
                   color='black')
        
        # Add vertical lines to separate M groups
        current_M = None
        for i, (M, Ns) in enumerate(experiments):
            if current_M is not None and M != current_M:
                boundary_x = (x_positions[i-1] + x_positions[i]) / 2
                ax.axvline(x=boundary_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            current_M = M
        
        # Calculate y-axis limit based on maximum normalized height
        max_height = max([solver2_data['total'][i] / solver1_data['total'][i] for i in range(n_experiments)] + [1.0])
        y_limit = max_height * 1.15  # Add 15% padding for labels
        
        # Customize axes
        ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
        ax.set_ylim(0, y_limit)
        ax.set_ylabel(f'Time Normalized by {self.labels.get(solver1_id, solver1_id) if hasattr(self, "labels") else solver1_id}')
        ax.set_xlabel('Experiment Configuration', labelpad=8)
        
        # Set x-tick labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(exp_labels, fontsize=9, ha='center')
        # Add shared labels for M and N_s
        x_pos = -0.5
        y_pos = -0.107
        gap = 0.095
        ax.text(x_pos, y_pos, r'$M$', transform=ax.transData, fontsize=9, weight='bold', ha='center')
        ax.text(x_pos, y_pos - gap, r'$N_s$', transform=ax.transData, fontsize=9, weight='bold', ha='center')
        
        # Get solver labels
        solver1_label = self.labels[solver1_id] if hasattr(self, 'labels') else solver1_id
        solver2_label = self.labels[solver2_id] if hasattr(self, 'labels') else solver2_id
        
        # Add title
        ax.set_title(f'Solve Time Breakdown: {solver1_label} vs {solver2_label}')
        
        # Create legend with colors and hatches
        legend_elements = [
            # Stack components
            Line2D([0], [0], marker='s', color='w', markerfacecolor=factor_color, 
                   markersize=8, label='Factor time', ls=''),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=solve_color, 
                   markersize=8, label='Solve time', ls=''),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=rest_color, 
                   markersize=8, label='Other time', ls=''),
            # Solver identification with hatches
            Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.7, hatch=solver1_hatch,
                     label=f'{solver1_label}'),
            Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.7, hatch=solver2_hatch,
                     label=f'{solver2_label}')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, ncol=2)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return solver1_data, solver2_data
