import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from run_benchmarks import load_benchmark_results


def _collect_solver_data(results, solver_id, param_values, x_param, field):
    """Collect timing data for a specific solver"""
    times_mean = []
    times_std = []

    if field in ['kkt_factor_time', 'kkt_solve_time'] and solver_id.startswith('piqp_'):
        field = 'piqp_' + field
    elif field == 'rest_time' and solver_id.startswith('piqp_'):
        total_solve_times, _ = _collect_solver_data(results, solver_id, param_values, x_param, 'solve_time')
        kkt_factor_times, _  = _collect_solver_data(results, solver_id, param_values, x_param, 'kkt_factor_time')
        kkt_solve_times, _   = _collect_solver_data(results, solver_id, param_values, x_param, 'kkt_solve_time')
        times_mean = [total_solve_times[i] - kkt_factor_times[i] - kkt_solve_times[i] for i in range(len(total_solve_times))]
        times_std = []  # std not yet supported for rest_time
        return times_mean, times_std

    for param_val in param_values:
        # Find matching result
        for problem_key, problem_results in results.items():
            if f"{x_param}{param_val}" in problem_key:
                if field not in problem_results[solver_id]:
                    break
                times_mean.append(problem_results[solver_id][field]['mean'])
                times_std.append(problem_results[solver_id][field]['std'])
                break

    return times_mean, times_std


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

    factor_cost_total = factor_cost_parallel_part(N, Ni, P) + factor_cost_serial_part(P)
    factor_cost_serial = 7/3*N - 2
    factor_speedup = (factor_cost_serial) / factor_cost_total

    solve_cost_total = solve_cost_parallel_part(N, Ni, P) + solve_cost_serial_part(P)
    solve_cost_serial = 5*N - 4
    solve_speedup = (solve_cost_serial) / solve_cost_total

    return factor_speedup, solve_speedup


def _gather_speedups(horizons, solver_labels, solver_data):
    """
    Build matrices indexed as [p_index, N_index] for experimental speedups
    vs PIQP (seq), and compute theory speedups with compute_theory_speedup.
    Returns:
        p_list           : sorted list of p values (e.g., [2,4,6,8,10,12])
        exp_factor[p,i]  : experimental factorization speedup for (p_list[p], horizons[i])
        exp_solve[p,i]   : experimental triangular-solve speedup
        th_factor[p,i]   : theoretical factorization speedup
        th_solve[p,i]    : theoretical triangular-solve speedup
    """
    base_label = "PIQP (seq)"
    base_factor = np.array(solver_data[base_label]["factor"], dtype=float)
    base_solve  = np.array(solver_data[base_label]["solve"],  dtype=float)

    # Collect p -> series maps
    p_to_exp_factor = {}
    p_to_exp_solve  = {}

    for label in solver_labels:
        if label.startswith("PIQP (par)"):
            p_val = int(label.split("p=")[1].strip(")$ "))
            fac = np.array(solver_data[label]["factor"], dtype=float)
            sol = np.array(solver_data[label]["solve"],  dtype=float)
            p_to_exp_factor[p_val] = base_factor / fac
            p_to_exp_solve[p_val]  = base_solve  / sol

    p_list = sorted(p_to_exp_factor.keys())

    # Build matrices [len(p_list) x len(horizons)]
    P = len(p_list); H = len(horizons)
    exp_factor = np.full((P, H), np.nan, dtype=float)
    exp_solve  = np.full((P, H), np.nan, dtype=float)
    th_factor  = np.full((P, H), np.nan, dtype=float)
    th_solve   = np.full((P, H), np.nan, dtype=float)

    for pi, p in enumerate(p_list):
        exp_factor[pi, :] = p_to_exp_factor[p]
        exp_solve[pi, :]  = p_to_exp_solve[p]
        for hi, N in enumerate(horizons):
            tf, ts = compute_theory_speedup(N, p)
            th_factor[pi, hi] = tf
            th_solve[pi,  hi] = ts

    return p_list, exp_factor, exp_solve, th_factor, th_solve

def plot_experiment_theory_speedup_ratio_heatmap(horizons, p_list, th_mat, exp_mat,
                                                title, filename,
                                                vcenter=1.0, vspan=0.4,
                                                fmt="{:.2f}", fontsize=20, txtsize=14):
    """
    Plot a labeled heatmap (using imshow) for the ratio (experiment/theory)
    across horizons (x-axis) and number of threads (y-axis).
    """
    horizons = np.asarray(horizons, dtype=int)
    p_list = np.asarray(p_list, dtype=int)

    # Compute ratio
    ratio = np.array(exp_mat, dtype=float) / np.array(th_mat, dtype=float)
    invalid = ~np.isfinite(th_mat) | (th_mat <= 0) | ~np.isfinite(exp_mat)
    ratio = np.where(invalid, np.nan, ratio)

    # Color settings
    vmin, vmax = vcenter - vspan, vcenter + vspan
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#d0d0d0")  # gray for invalid cells

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(ratio, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", aspect="auto")

    # Axis setup
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_yticks(np.arange(len(p_list)))
    ax.set_xticklabels([f"${N}$" for N in horizons], fontsize=fontsize-2)
    ax.set_yticklabels([f"${p}$" for p in p_list], fontsize=fontsize-2)
    ax.set_xlabel("Horizon $N$", fontsize=fontsize)
    ax.set_ylabel("Number of threads $p$", fontsize=fontsize)

    # Add numeric annotations
    for i in range(len(p_list)):
        for j in range(len(horizons)):
            val = ratio[i, j]
            if np.isnan(val):
                continue
            # choose text color for contrast
            r, g, b, _ = cmap((val - vmin) / (vmax - vmin))
            L = 0.299*r + 0.587*g + 0.114*b
            txt_col = "black" if L > 0.6 else "white"
            txt_col = 'black'
            ax.text(j, i, fmt.format(val),
                    ha="center", va="center", color=txt_col, fontsize=txtsize)

    # Title and colorbar
    # ax.set_title(title, fontsize=fontsize+1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Experiment / Theory (×)", fontsize=fontsize)

    plt.tight_layout()
    # plt.savefig(filename + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(filename + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close()


def print_speedup_tables(horizons, solver_labels, solver_data):
    """
    Print combined theoretical and experimental speedup tables
    for KKT factorization and triangular solve (vs PIQP (seq)).
    Each cell shows: theory | experiment
    """

    base_label = "PIQP (seq)"
    base_factor = np.array(solver_data[base_label]["factor"], dtype=float)
    base_solve  = np.array(solver_data[base_label]["solve"],  dtype=float)

    # Collect parallel solvers
    rows_factor = []
    rows_solve = []
    for label in solver_labels:
        if label.startswith("PIQP (par)"):
            p_val = int(label.split("p=")[1].strip(")$ "))
            fac = np.array(solver_data[label]["factor"], dtype=float)
            sol = np.array(solver_data[label]["solve"],  dtype=float)

            exp_fac_speedup = base_factor / fac
            exp_sol_speedup = base_solve / sol

            # Compute theoretical values for each horizon
            theory_fac = []
            theory_sol = []
            for N in horizons:
                t_fac, t_sol = compute_theory_speedup(N, p_val)
                theory_fac.append(t_fac)
                theory_sol.append(t_sol)

            rows_factor.append((p_val, theory_fac, exp_fac_speedup))
            rows_solve.append((p_val, theory_sol, exp_sol_speedup))

    # Sort by p
    rows_factor.sort(key=lambda x: x[0])
    rows_solve.sort(key=lambda x: x[0])

    def format_table(rows, title):
        print("\n" + "=" * 110)
        print(f"{title}")
        print("=" * 110)
        header = "p\\N".ljust(8) + "".join([f"{N:>10}" for N in horizons])
        print(header)
        print("-" * len(header))
        for p, theory_arr, exp_arr in rows:
            values = ""
            for t, e in zip(theory_arr, exp_arr):
                if np.isnan(t) or np.isnan(e):
                    values += f"{'---':>11}"
                else:
                    values += f"{t:5.2f}/{e:4.2f}"
            print(f"{p:<12}{values}")
        print()

    format_table(rows_factor, "KKT Factorization Speedup (Theory | Experiment)")
    format_table(rows_solve,  "KKT Triangular Solve Speedup (Theory | Experiment)")


def main():
    fontsize = 18
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    new_colors = ['k'] + default_colors
    plt.rcParams['axes.prop_cycle'] = cycler(color=new_colors)

    solver_ids = [
        'hpipm',
        'piqp_avx2',
        'piqp_avx2_p', 'piqp_avx2_p', 'piqp_avx2_p',
        'piqp_avx2_p', 'piqp_avx2_p', 'piqp_avx2_p',
    ]
    solver_labels = [
        'HPIPM',
        'PIQP (seq)',
        'PIQP (par) $p=2$',
        'PIQP (par) $p=4$',
        'PIQP (par) $p=6$',
        'PIQP (par) $p=8$',
        'PIQP (par) $p=10$',
        'PIQP (par) $p=12$',
    ]
    solver_result_files = [
        'benchmark_ChainMassOCPProblem_M20_N20-200_T0_hpipm_20251008_171536.json',
        'benchmark_ChainMassOCPProblem_M20_N20-200_T0_20251008_164532.json',
        'benchmark_ChainMassOCPProblem_M20_N20-200_T2_20251008_161544.json',
        'benchmark_ChainMassOCPProblem_M20_N20-200_T4_20251008_161826.json',
        'benchmark_ChainMassOCPProblem_M20_N20-200_T6_20251008_162057.json',
        'benchmark_ChainMassOCPProblem_M20_N20-200_T8_20251008_162320.json',
        'benchmark_ChainMassOCPProblem_M20_N40-200_T10_20251008_164253.json',
        'benchmark_ChainMassOCPProblem_M20_N40-200_T12_20251008_162835.json',
    ]

    # horizons must match your JSON (order matters)
    horizons = [40, 60, 80, 100, 120, 140, 160, 180, 200]

    # collect data
    solver_data = {label: None for label in solver_labels}
    # gold for HPIPM first, then tab10 palette
    solver_colors = ('goldenrod',) + plt.cm.tab10.colors

    for solver_id, solver_label, result_file in zip(solver_ids, solver_labels, solver_result_files):
        results = load_benchmark_results('results/' + result_file)
        solver_times = {'total': [], 'factor': [], 'solve': [], 'rest': []}
        for field in ['solve_time', 'kkt_factor_time', 'kkt_solve_time', 'rest_time']:
            mean_times, _ = _collect_solver_data(results, solver_id, horizons, 'N', field)
            key = 'total' if field == 'solve_time' else field.split('_')[-2]  # factor/solve/rest
            solver_times[key] = mean_times
        solver_data[solver_label] = solver_times

    # --- PRINT SPEEDUPS ---
    print_speedup_tables(horizons, solver_labels, solver_data)

    # ----------------- PLOTTING -----------------
    S = len(solver_labels)
    H = len(horizons)

    fig, ax = plt.subplots(figsize=(12, 4.8))

    # layout
    bar_width   = 0.15
    group_gap   = 0.25
    group_width = S * bar_width
    centers     = np.arange(H) * (group_width + group_gap)

    # transparencies for components (used in bars & legend)
    alpha_factor = 1.00
    alpha_solve  = 0.70
    alpha_rest   = 0.40

    # draw stacked bars (color = solver, alpha = component)
    totals_ms_by_solver = {}  # keep for speedup annotations
    for j, label in enumerate(solver_labels):
        x = centers - (group_width - bar_width)/2 + j * bar_width

        total = 1e3 * np.array(solver_data[label]['total'],  dtype=float)
        factor = 1e3 * np.array(solver_data[label]['factor'], dtype=float)
        solve  = 1e3 * np.array(solver_data[label]['solve'],  dtype=float)
        rest   = 1e3 * np.array(solver_data[label]['rest'],   dtype=float)

        totals_ms_by_solver[label] = total  # store total for fastest-solver search

        color = solver_colors[j % len(solver_colors)]
        edgecolor = 'white'  # keeps translucent stacks visually separated
        if solver_labels[j].startswith('PIQP'):
            ax.bar(x, factor, width=bar_width, color=color, edgecolor=edgecolor,
                   linewidth=0.5, alpha=alpha_factor)
            ax.bar(x, solve, width=bar_width, color=color, edgecolor=edgecolor,
                   linewidth=0.5, bottom=factor, alpha=alpha_solve)
            ax.bar(x, rest, width=bar_width, color=color, edgecolor=edgecolor,
                   linewidth=0.5, bottom=factor+solve, alpha=alpha_rest)
        else:
            # HPIPM only has total
            ax.bar(x, total, width=bar_width, color=color, edgecolor=edgecolor,
                   linewidth=0.5, alpha=alpha_factor, hatch='///')
            mpl.rcParams['hatch.linewidth'] = 1.0

    # axis & labels
    ax.set_xticks(centers)
    ax.set_xticklabels([f"${N}$" for N in horizons], fontsize=fontsize-1)
    ax.set_xlabel(f"Horizon $N$", fontsize=fontsize)
    ax.set_ylabel("Time [ms]", fontsize=fontsize)
    ax.set_title("Benchmark Results: Chain of Masses OCP", fontsize=fontsize)
    ax.minorticks_on()
    ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.grid(True, axis='y', which='minor', linestyle=':', linewidth=0.6, alpha=0.4)

    # -------- legends: two separate blocks, side-by-side (top-left) --------
    # Solvers (color-only)
    solver_handles = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=solver_colors[j % len(solver_colors)],
               markeredgecolor=None, markersize=10,
               label=solver_labels[j], ls='')
        for j in range(S)
    ]
    solver_handles = []
    for j, lab in enumerate(solver_labels):
        face = solver_colors[j % len(solver_colors)]
        if lab == 'HPIPM':
            # hatched golden patch to match the bar
            h = Patch(facecolor=face, edgecolor='white', hatch='///', label=lab)
        else:
            # plain colored patch for PIQP variants
            h = Patch(facecolor=face, edgecolor=None, label=lab)
        solver_handles.append(h)

    leg_solvers = ax.legend(
        handles=solver_handles,
        title="Solvers",
        loc="upper left",
        bbox_to_anchor=(0.00, 0.995),
        fontsize=fontsize-5,
        title_fontsize=fontsize-4,
        framealpha=1.0, # 0.70,
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=2.0,
        handleheight=1.0,
        columnspacing=1.0,
    )
    ax.add_artist(leg_solvers)

    # Components (neutral gray patches with different alphas)
    comp_handles = [
        Patch(facecolor="#555555", edgecolor='black', linewidth=1., alpha=alpha_factor, label="Factorization"),
        Patch(facecolor="#555555", edgecolor='black', linewidth=1., alpha=alpha_solve,  label="Triangular solve"),
        Patch(facecolor="#555555", edgecolor='black', linewidth=1., alpha=alpha_rest,   label="Other"),
    ]
    # comp_handles.reverse()
    leg_components = ax.legend(
        handles=comp_handles,
        title="Components",
        loc="upper left",
        bbox_to_anchor=(0.2, 0.995),  # place to the right of the solvers legend
        fontsize=fontsize-5,
        title_fontsize=fontsize-4,
        framealpha=1.0, #0.70,
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=2.0,
        handleheight=1.0,
        columnspacing=1.0,
    )
    ax.add_artist(leg_components)

    # ----------------- SPEEDUP ANNOTATIONS (per-horizon) -----------------
    # Baseline: PIQP (seq) total time (ms)
    baseline = totals_ms_by_solver['PIQP (seq)']

    # Precompute x positions for solver j in group i
    def x_pos(i, j):
        return centers[i] - (group_width - bar_width)/2 + j * bar_width

    for i in range(H):
        # Collect total time at horizon i for each solver
        totals_i = np.array([totals_ms_by_solver[label][i] for label in solver_labels], dtype=float)
        if not np.isfinite(baseline[i]):
            continue

        # Index of fastest solver at this horizon
        j_fast = int(np.nanargmin(totals_i))
        fast_time = totals_i[j_fast]
        base_time = baseline[i]

        # Only annotate if faster than baseline
        if not np.isfinite(fast_time) or fast_time >= base_time:
            continue

        speedup = base_time / fast_time  # >1 means faster

        # x position for the fastest bar in this group
        x_baseline = x_pos(i, 1)  # baseline is always PIQP (seq) at j=1
        xf = x_pos(i, j_fast)

        # small horizontal dashes at baseline & fastest levels
        dash_half = bar_width * 0.45
        ax.hlines([base_time], x_baseline - dash_half, xf + dash_half,
                  colors='k', linestyles='--', linewidth=0.4)

        # downward arrow from baseline to fastest bar top
        ax.annotate(
            '', xy=(xf, fast_time), xytext=(xf, base_time),
            arrowprops=dict(arrowstyle='->', color='k', lw=0.6),
            ha='center', va='center'
        )

        # text label centered between the two levels
        y_text = base_time + 3
        ax.text((x_baseline+xf)/2, y_text, f'{speedup:.2f}×', ha='center', va='center',
                fontsize=fontsize-6, color='k')

    # clean frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # give legends some breathing room above axes
    plt.subplots_adjust(top=0.86)

    plt.tight_layout()
    # plt.savefig("benchmark_barplot_chain_mass_ocp.png",
    #             dpi=300, bbox_inches="tight", edgecolor='none', facecolor='white')
    plt.savefig("benchmark_barplot_chain_mass_ocp.pdf",
                bbox_inches="tight", edgecolor='none', facecolor='white')
    plt.close()


    # --- compute speedups matrices ---
    p_list, exp_factor, exp_solve, th_factor, th_solve = _gather_speedups(horizons, solver_labels, solver_data)

    # Factorization: ratio heatmap
    plot_experiment_theory_speedup_ratio_heatmap(
        horizons, p_list,
        th_mat=th_factor, exp_mat=exp_factor,
        title="Experiment/Theory Speedup Ratio — KKT Factorization",
        filename="results/speedup_ratio_heatmap_factor",
        vcenter=1.0, vspan=0.4  # color scale: [0.6, 1.4]
    )

    # Triangular solve: ratio heatmap
    plot_experiment_theory_speedup_ratio_heatmap(
        horizons, p_list,
        th_mat=th_solve, exp_mat=exp_solve,
        title="Experiment/Theory Speedup Ratio — KKT Triangular Solve",
        filename="results/speedup_ratio_heatmap_solve",
        vcenter=1.0, vspan=0.4  # triangular solve often higher; widen if needed
    )


if __name__ == '__main__':
    main()