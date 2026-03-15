"""
fig2b.py
Reproduces Fig.2B from Baym et al. 2016:
  Adaptation rate (1 / t_adapt) vs intermediate antibiotic concentration.
  Runs both CA and CPM side by side for comparison.

Layout:
  Antibiotic field: [ 0 | middle_conc | 3000 | middle_conc | 0 ]
  middle_conc swept: 0, 3, 30, 300 x MIC

Usage:
  python fig2b.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pde import (
    Nx, Ny, MIC_base, dt,
    init_nutrients,
    build_implicit_solver,
    L2D_A, L2D_N,
    D_A, D_N,
)
import CA as ca_module
import cpm as cpm_module

# ══════════════════════════════════════════════════════════════════════════════
# Experiment configuration
# ══════════════════════════════════════════════════════════════════════════════

MIDDLE_STEPS = [0, 3, 30, 300]   # intermediate band concentrations (x MIC)
N_RUNS       = 5                  # independent runs per condition
MAX_STEPS    = 20_000             # give up after this many steps
CHECK_EVERY  = 10                 # check front position every N steps

# shared CA / CPM parameters
PARAMS = {
    'dt'                  : dt,
    'D_A'                 : 0.005,     # FIX: lowered 10x, must match pde.py
    'delta_A'             : 0.002,
    'MIC_base'            : MIC_base,
    'D_N'                 : D_N,
    'alpha'               : 0.005,
    'K_N'                 : 1.0,
    'nutrient_consumption': 0.005,
    # CA
    'p_repro_base'        : 0.08,
    # mutation (shared)
    'mu_base'             : 0.0003,    # FIX: lowered 10x — prevents mutation explosion
    'mic_fold_mean'       : 1.5,       # FIX: lowered from 3.0 — more realistic per-step gain
    'mic_fold_sigma'      : 0.3,       # FIX: tightened from 0.5
    # CPM energy
    'J_medium'            : 8.0,
    'J_diff'              : 16.0,
    'J_same'              : 2.0,
    'lambda_V'            : 0.5,
    'V_target'            : 25,        # FIX: restored to 25 — cells must grow before competing
    'chi'                 : 3.0,
    'T_cpm'               : 5.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Antibiotic field helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_antibiotic_field(middle_conc):
    """
    Symmetric 5-zone field:
      [ 0 | middle_conc | 3000 | middle_conc | 0 ]  (all x MIC_base)
    """
    levels     = np.array([0, middle_conc, 3000, middle_conc, 0],
                           dtype=float) * MIC_base
    A          = np.zeros((Ny, Nx))
    n_zones    = len(levels)
    zone_width = Nx // n_zones

    for z, level in enumerate(levels):
        x_start = z * zone_width
        x_end   = (z + 1) * zone_width if z < n_zones - 1 else Nx
        A[:, x_start:x_end] = level
    return A


def highest_band_x(middle_conc):
    """x column where the 3000x MIC band begins (zone index 2)."""
    return 2 * (Nx // 5)


def front_x_left(bacteria_2d):
    """Rightmost occupied column in the left half of the plate."""
    half = bacteria_2d[:, : Nx // 2]
    if not half.any():
        return 0
    return int(np.max(np.where(half)[1]))


# ══════════════════════════════════════════════════════════════════════════════
# CA single run
# ══════════════════════════════════════════════════════════════════════════════

def run_ca_single(middle_conc, seed=None):
    if seed is not None:
        np.random.seed(seed)

    A       = build_antibiotic_field(middle_conc)
    A_bl    = A[:, 0].copy()
    A_br    = A[:, -1].copy()
    N_field = init_nutrients(Nx, Ny)
    sol_A   = build_implicit_solver(L2D_A, PARAMS['D_A'], PARAMS['dt'], Nx * Ny)
    sol_N   = build_implicit_solver(L2D_N, PARAMS['D_N'], PARAMS['dt'], Nx * Ny)

    sim      = ca_module.CASimulation(PARAMS, A, N_field, sol_A, sol_N, A_bl, A_br)
    target_x = highest_band_x(middle_conc)

    for step in range(MAX_STEPS):
        sim.step()
        if step % CHECK_EVERY == 0:
            if front_x_left(sim.bacteria_2d) >= target_x:
                return sim.step_count
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CPM helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cpm_seed_pixels(col):
    r0 = Ny // 2
    return [(r0 + dr, col) for dr in range(-3, 4)]


def _reset_cpm(params):
    cpm_module.cell_id[:]    = cpm_module.MEDIUM
    cpm_module.cell_mic.clear()
    cpm_module.cell_lineage.clear()
    cpm_module.cell_volume.clear()
    cpm_module.cell_mic[cpm_module.MEDIUM]     = 0.0
    cpm_module.cell_lineage[cpm_module.MEDIUM] = -1
    cpm_module.cell_volume[cpm_module.MEDIUM]  = Nx * Ny
    cpm_module.next_cell_id = 1

    cpm_module.J_medium       = params['J_medium']
    cpm_module.J_diff         = params['J_diff']
    cpm_module.J_same         = params['J_same']
    cpm_module.lambda_V       = params['lambda_V']
    cpm_module.V_target       = params['V_target']
    cpm_module.chi            = params['chi']
    cpm_module.T              = params['T_cpm']
    cpm_module.mu_base        = params['mu_base']
    cpm_module.mic_fold_mean  = params['mic_fold_mean']
    cpm_module.mic_fold_sigma = params['mic_fold_sigma']

    cpm_module.add_cell(_cpm_seed_pixels(0),      params['MIC_base'], 0)
    cpm_module.add_cell(_cpm_seed_pixels(Nx - 1), params['MIC_base'], 1)


def _step_cpm_pde(A, N_field, A_bl, A_br, sol_A, sol_N):
    rho_flat   = cpm_module.get_rho_grid().ravel()

    A_diff     = sol_A(A.ravel())
    A_flat     = A_diff - PARAMS['dt'] * PARAMS['delta_A'] * rho_flat * A_diff
    A_flat     = np.maximum(A_flat, 0.0)
    A_2d       = A_flat.reshape(Ny, Nx)
    A_2d[:, 0] = A_bl
    A_2d[:, -1]= A_br
    A[:]       = A_2d

    N_diff     = sol_N(N_field.ravel())
    N_flat     = N_diff - PARAMS['dt'] * PARAMS['alpha'] * rho_flat * (
                     N_diff / (PARAMS['K_N'] + N_diff))
    N_field[:] = np.maximum(N_flat, 0.0).reshape(Ny, Nx)


# ══════════════════════════════════════════════════════════════════════════════
# CPM single run
# ══════════════════════════════════════════════════════════════════════════════

def run_cpm_single(middle_conc, seed=None):
    if seed is not None:
        np.random.seed(seed)

    A       = build_antibiotic_field(middle_conc)
    A_bl    = A[:, 0].copy()
    A_br    = A[:, -1].copy()
    N_field = init_nutrients(Nx, Ny)
    sol_A   = build_implicit_solver(L2D_A, PARAMS['D_A'], PARAMS['dt'], Nx * Ny)
    sol_N   = build_implicit_solver(L2D_N, PARAMS['D_N'], PARAMS['dt'], Nx * Ny)

    _reset_cpm(PARAMS)
    target_x = highest_band_x(middle_conc)
    step_n   = 0

    for step in range(MAX_STEPS):
        _step_cpm_pde(A, N_field, A_bl, A_br, sol_A, sol_N)
        cpm_module.mcs_step(N_field, A)
        cpm_module.do_mutations()
        cpm_module.kill_exposed_cells(A)
        step_n += 1

        if step % CHECK_EVERY == 0:
            if front_x_left(cpm_module.get_bacteria_grid()) >= target_x:
                return step_n
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Experiment loop
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(model_name, run_fn):
    """
    Run N_RUNS simulations for each intermediate concentration.
    Returns dict: middle_conc -> list of t_adapt (None = no adaptation).
    """
    results = {}
    total   = len(MIDDLE_STEPS) * N_RUNS
    done    = 0

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    for middle_conc in MIDDLE_STEPS:
        t_list = []
        for run_i in range(N_RUNS):
            done += 1
            seed = run_i * 100 + int(middle_conc) + (0 if model_name == 'CA' else 500)
            print(f"  [{done}/{total}]  middle={middle_conc:4}x MIC"
                  f"  run {run_i+1}/{N_RUNS} ...", end=' ', flush=True)
            t = run_fn(middle_conc, seed=seed)
            if t is None:
                print(f"no adaptation (>{MAX_STEPS} steps)")
            else:
                print(f"t_adapt = {t:,}")
            t_list.append(t)
        results[middle_conc] = t_list

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Summarise
# ══════════════════════════════════════════════════════════════════════════════

def summarise(model_name, results):
    print(f"\n── {model_name} summary ──────────────────────────────")
    for mc in MIDDLE_STEPS:
        t_list  = results[mc]
        adapted = [t for t in t_list if t is not None]
        if adapted:
            print(f"  middle={mc:4}x MIC : {len(adapted)}/{N_RUNS} adapted"
                  f"  |  mean={np.mean(adapted):.0f}"
                  f"  std={np.std(adapted):.0f}")
        else:
            print(f"  middle={mc:4}x MIC : 0/{N_RUNS} adapted")


# ══════════════════════════════════════════════════════════════════════════════
# Calibration — measure front speed in antibiotic-free plate
# ══════════════════════════════════════════════════════════════════════════════

CALIB_TARGET_X = Nx // 4    # front must reach x = Nx/4 from the left edge
CALIB_SEED     = 999


def calibrate_ca():
    """
    Run CA on a fully antibiotic-free plate and measure how many steps it
    takes for the front to advance to x=CALIB_TARGET_X.
    This gives a model-intrinsic time unit for CA.
    """
    np.random.seed(CALIB_SEED)
    A       = np.zeros((Ny, Nx))
    A_bl    = A[:, 0].copy()
    A_br    = A[:, -1].copy()
    N_field = init_nutrients(Nx, Ny)
    sol_A   = build_implicit_solver(L2D_A, PARAMS['D_A'], PARAMS['dt'], Nx * Ny)
    sol_N   = build_implicit_solver(L2D_N, PARAMS['D_N'], PARAMS['dt'], Nx * Ny)

    sim = ca_module.CASimulation(PARAMS, A, N_field, sol_A, sol_N, A_bl, A_br)
    for step in range(MAX_STEPS):
        sim.step()
        if step % CHECK_EVERY == 0:
            if front_x_left(sim.bacteria_2d) >= CALIB_TARGET_X:
                print(f"  CA  calibration: front reached x={CALIB_TARGET_X}"
                      f" at step {sim.step_count}")
                return sim.step_count
    return MAX_STEPS


def calibrate_cpm():
    """
    Same calibration for CPM: antibiotic-free plate, measure steps to
    reach x=CALIB_TARGET_X.
    """
    np.random.seed(CALIB_SEED)
    A       = np.zeros((Ny, Nx))
    A_bl    = A[:, 0].copy()
    A_br    = A[:, -1].copy()
    N_field = init_nutrients(Nx, Ny)
    sol_A   = build_implicit_solver(L2D_A, PARAMS['D_A'], PARAMS['dt'], Nx * Ny)
    sol_N   = build_implicit_solver(L2D_N, PARAMS['D_N'], PARAMS['dt'], Nx * Ny)

    _reset_cpm(PARAMS)
    step_n = 0
    for step in range(MAX_STEPS):
        _step_cpm_pde(A, N_field, A_bl, A_br, sol_A, sol_N)
        cpm_module.mcs_step(N_field, A)
        cpm_module.do_mutations()
        cpm_module.kill_exposed_cells(A)
        step_n += 1
        if step % CHECK_EVERY == 0:
            if front_x_left(cpm_module.get_bacteria_grid()) >= CALIB_TARGET_X:
                print(f"  CPM calibration: front reached x={CALIB_TARGET_X}"
                      f" at step {step_n}")
                return step_n
    return MAX_STEPS


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def compute_rates(results, calib_steps=1):
    """
    Return (x_concs, rate_mean, rate_std, t_norm_mean) for plotting.
    - None t_adapt → MAX_STEPS (rate ≈ 0)
    - t values divided by calib_steps so both models share a common time unit
      (= steps needed to cross one drug-free zone at baseline speed)
    """
    x_concs     = []
    rate_mean   = []
    rate_std    = []
    t_norm_mean = []

    for mc in MIDDLE_STEPS:
        t_vals  = [t if t is not None else MAX_STEPS for t in results[mc]]
        t_norm  = [t / calib_steps for t in t_vals]
        rates   = [1.0 / t for t in t_norm]
        x_concs.append(mc if mc > 0 else 0.5)
        rate_mean.append(float(np.mean(rates)))
        rate_std.append(float(np.std(rates)))
        t_norm_mean.append(float(np.mean(t_norm)))

    return x_concs, rate_mean, rate_std, t_norm_mean


def _normalise_to_peak(values):
    """Scale a list so its maximum = 1.0 (for shape comparison)."""
    peak = max(values)
    if peak == 0:
        return values
    return [v / peak for v in values]


def _paper_reference_curve(x_concs):
    """
    Hand-digitised approximation of the inverted-U shape from Fig.2B (TMP).
    Returned as normalised [0, 1] values for shape comparison only.
    """
    ref_map = {0.5: 0.05, 3: 0.65, 30: 1.0, 300: 0.45}
    return [ref_map.get(x, 0.5) for x in x_concs]


def plot_results(ca_results, cpm_results, ca_calib, cpm_calib):
    ca_x,  ca_rm,  ca_rs,  ca_tm  = compute_rates(ca_results,  ca_calib)
    cpm_x, cpm_rm, cpm_rs, cpm_tm = compute_rates(cpm_results, cpm_calib)

    # normalise each model's rates to [0,1] peak for shape comparison
    ca_norm  = _normalise_to_peak(ca_rm)
    cpm_norm = _normalise_to_peak(cpm_rm)
    ca_rs_n  = [s / max(ca_rm)  if max(ca_rm)  > 0 else 0 for s in ca_rs]
    cpm_rs_n = [s / max(cpm_rm) if max(cpm_rm) > 0 else 0 for s in cpm_rs]
    paper    = _paper_reference_curve(ca_x)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Fig.2B reproduction — CA vs CPM", fontsize=13, fontweight='bold')

    tick_labels = [str(int(c)) if c >= 1 else '0' for c in ca_x]

    # ── panel 1: normalised shape comparison + paper reference ───────────────
    ax = axes[0]
    ax.errorbar(ca_x,  ca_norm,  yerr=ca_rs_n,
                fmt='o-',  color='steelblue', capsize=4,
                linewidth=2, markersize=7, label='CA (normalised)')
    ax.errorbar(cpm_x, cpm_norm, yerr=cpm_rs_n,
                fmt='s--', color='tomato', capsize=4,
                linewidth=2, markersize=7, label='CPM (normalised)')
    ax.plot(ca_x, paper, 'k:', linewidth=2.5,
            label='Paper Fig.2B\n(schematic)')

    ax.set_xscale('log')
    ax.set_xlabel('Intermediate concentration (MIC units)', fontsize=11)
    ax.set_ylabel('Normalised adaptation rate  (peak = 1)', fontsize=11)
    ax.set_title(f'Shape comparison\n'
                 f'Both models normalised to their own peak\n'
                 f'(mean ± std, n={N_RUNS})', fontsize=9)
    ax.set_xticks(ca_x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-0.05, 1.3)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)

    # ── panel 2: calibration-normalised rate (shared time unit) ──────────────
    ax2 = axes[1]
    ax2.errorbar(ca_x,  ca_rm,  yerr=ca_rs,
                 fmt='o-',  color='steelblue', capsize=4,
                 linewidth=2, markersize=7, label='CA')
    ax2.errorbar(cpm_x, cpm_rm, yerr=cpm_rs,
                 fmt='s--', color='tomato', capsize=4,
                 linewidth=2, markersize=7, label='CPM')

    ax2.set_xscale('log')
    ax2.set_xlabel('Intermediate concentration (MIC units)', fontsize=11)
    ax2.set_ylabel('Adaptation rate  (1 / t_normalised)', fontsize=11)
    ax2.set_title(f'Calibration-normalised rate\n'
                  f't divided by antibiotic-free front speed\n'
                  f'CA ref={ca_calib} steps  |  CPM ref={cpm_calib} steps',
                  fontsize=9)
    ax2.set_xticks(ca_x)
    ax2.set_xticklabels(tick_labels)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=9)

    # ── panel 3: raw t_adapt scatter ─────────────────────────────────────────
    ax3 = axes[2]
    jitter = 0.08

    for i, mc in enumerate(MIDDLE_STEPS):
        x = ca_x[i]
        for t in ca_results[mc]:
            y      = (t if t is not None else MAX_STEPS) / ca_calib
            marker = 'x' if t is None else 'o'
            ax3.scatter(x * (1 - jitter), y, marker=marker,
                        color='steelblue', s=50, zorder=3, alpha=0.8)
        for t in cpm_results[mc]:
            y      = (t if t is not None else MAX_STEPS) / cpm_calib
            marker = 'x' if t is None else 's'
            ax3.scatter(x * (1 + jitter), y, marker=marker,
                        color='tomato', s=50, zorder=3, alpha=0.8)

    ax3.plot(ca_x,  ca_tm,  'o-',  color='steelblue', linewidth=1.5,
             alpha=0.6, label='CA mean')
    ax3.plot(cpm_x, cpm_tm, 's--', color='tomato',    linewidth=1.5,
             alpha=0.6, label='CPM mean')

    ax3.set_xscale('log')
    ax3.set_xlabel('Intermediate concentration (MIC units)', fontsize=11)
    ax3.set_ylabel('t_adapt / t_calib  (normalised steps)', fontsize=11)
    ax3.set_title('Raw adaptation times (normalised)\n'
                  '(× = no adaptation within limit)', fontsize=9)
    ax3.set_xticks(ca_x)
    ax3.set_xticklabels(tick_labels)
    ax3.grid(True, which='both', alpha=0.3)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    out = 'fig2b_reproduction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.show()

def debug_plot_antibiotic(middle_conc=30, steps=100):
    """
    Run a short simulation and plot the antibiotic field
    after `steps` timesteps.
    """
    np.random.seed(0)

    # initial fields
    A       = build_antibiotic_field(middle_conc)
    A_bl    = A[:, 0].copy()
    A_br    = A[:, -1].copy()
    N_field = init_nutrients(Nx, Ny)

    # PDE solvers
    sol_A = build_implicit_solver(L2D_A, PARAMS['D_A'], PARAMS['dt'], Nx * Ny)
    sol_N = build_implicit_solver(L2D_N, PARAMS['D_N'], PARAMS['dt'], Nx * Ny)

    # create CA simulation
    sim = ca_module.CASimulation(PARAMS, A, N_field, sol_A, sol_N, A_bl, A_br)

    # run some steps
    for _ in range(steps):
        sim.step()

    # plot antibiotic field
    plt.figure(figsize=(6,4))
    plt.imshow(A, origin='lower', aspect='auto')
    plt.colorbar(label="Antibiotic concentration")
    plt.title(f"Antibiotic field after {steps} steps (middle={middle_conc}x MIC)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Fig.2B reproduction — CA vs CPM")
    print(f"  Middle steps : {MIDDLE_STEPS}")
    print(f"  Runs per cond: {N_RUNS}")
    print(f"  Max steps    : {MAX_STEPS:,}")

    # ── step 0: calibrate each model's intrinsic time unit ───────────────────
    print("\n── Calibration (antibiotic-free front speed) ────────────")
    ca_calib  = calibrate_ca()
    cpm_calib = calibrate_cpm()
    print(f"  CA  time unit : {ca_calib} steps")
    print(f"  CPM time unit : {cpm_calib} steps")
    print(f"  CPM / CA ratio: {cpm_calib / ca_calib:.2f}x  "
          f"(t_adapt will be divided by these before comparison)")

    # ── step 1: run experiments ───────────────────────────────────────────────
    ca_results  = run_experiment('CA',  run_ca_single)
    cpm_results = run_experiment('CPM', run_cpm_single)

    summarise('CA',  ca_results)
    summarise('CPM', cpm_results)

    # ── step 2: plot with calibration and normalisation ───────────────────────
    plot_results(ca_results, cpm_results, ca_calib, cpm_calib)