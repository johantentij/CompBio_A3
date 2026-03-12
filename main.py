import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pde import (
    init_antibiotic, init_nutrients,
    build_implicit_solver,
    L2D_A, L2D_N,
    Nx, Ny, N_init
)
import ca as ca_module
import cpm as cpm_module

# ══════════════════════════════════════════════════════════════════════════════
# parameters — edit here to tune both models simultaneously
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = {
    # grid / time
    'dt'                  : 0.05,

    # antibiotic PDE
    'D_A'                 : 0.05,
    'delta_A'             : 0.002,
    'MIC_base'            : 1.0,

    # nutrient PDE
    'D_N'                 : 2.0,
    'alpha'               : 0.005,
    'K_N'                 : 1.0,
    'nutrient_consumption': 0.005,

    # CA reproduction
    'p_repro_base'        : 0.08,

    # mutation
    'mu_base'             : 0.003,
    'mic_fold_mean'       : 2,
    'mic_fold_sigma'      : 0.3,

    # CPM energy
    'J_medium'            : 8.0,
    'J_diff'              : 16.0,
    'J_same'              : 2.0,
    'lambda_V'            : 0.5,
    'V_target'            : 4,       # lowered: small mutant clusters survive
    'chi'                 : 3.0,
    'T_cpm'               : 5.0,

    # animation
    'steps_per_frame'     : 5,
    'interval_ms'         : 80,
}

# ── run mode ──────────────────────────────────────────────────────────────────
# 'ca'      — run CA only  (lineage map + resistance map)
# 'cpm'     — run CPM only (lineage map + resistance map)
# 'compare' — run CA and CPM side by side (2×2 grid)
MODE = 'compare'

# ══════════════════════════════════════════════════════════════════════════════
# shared PDE initialisation
# ══════════════════════════════════════════════════════════════════════════════

def make_pde_fields():
    A       = init_antibiotic(Nx, Ny)
    N_field = init_nutrients(Nx, Ny)
    A_bl    = A[:, 0].copy()
    A_br    = A[:, -1].copy()
    A_init  = A.copy()
    sol_A   = build_implicit_solver(L2D_A, PARAMS['D_A'], PARAMS['dt'], Nx * Ny)
    sol_N   = build_implicit_solver(L2D_N, PARAMS['D_N'], PARAMS['dt'], Nx * Ny)
    return A, N_field, A_bl, A_br, A_init, sol_A, sol_N


def _cpm_seed_pixels(col):
    """return a small column of pixels to seed a CPM founder cell."""
    r0 = Ny // 2
    return [(r0 + dr, col) for dr in range(-3, 4)]   # 7-pixel seed


def _reset_cpm(params):
    """reset all cpm module globals and re-seed two founder cells."""
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
    """advance antibiotic + nutrient PDE fields by one dt for the CPM loop."""
    rho_flat   = cpm_module.get_rho_grid().ravel()

    A_diffused = sol_A(A.ravel())
    A_flat     = A_diffused - PARAMS['dt'] * PARAMS['delta_A'] * rho_flat * A_diffused
    A_flat     = np.maximum(A_flat, 0.0)
    A_2d       = A_flat.reshape(Ny, Nx)
    A_2d[:, 0] = A_bl
    A_2d[:, -1]= A_br
    A[:]       = A_2d

    N_diffused = sol_N(N_field.ravel())
    N_flat     = N_diffused - PARAMS['dt'] * PARAMS['alpha'] * rho_flat * (
                     N_diffused / (PARAMS['K_N'] + N_diffused))
    N_flat     = np.maximum(N_flat, 0.0)
    N_field[:] = N_flat.reshape(Ny, Nx)


# ══════════════════════════════════════════════════════════════════════════════
# visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

CONTOUR_LEVELS = [3, 30, 300, 3000]
CONTOUR_COLORS = ['lightyellow', 'orange', 'red', 'darkred']


def setup_lineage_panel(ax, title_str, A_init, N_field, show_ylabel=True):
    """nutrient background + lineage colour overlay."""
    ax.set_title(title_str)
    ax.set_xlabel('x (gradient direction)')
    if show_ylabel:
        ax.set_ylabel('y')

    im_n = ax.imshow(
        N_field, cmap='Greens', vmin=0, vmax=N_init,
        origin='lower', aspect='auto', interpolation='nearest'
    )
    blank = np.ma.masked_all((Ny, Nx))
    im_o  = ax.imshow(
        blank, cmap='tab20', vmin=0, vmax=20,
        origin='lower', aspect='auto', interpolation='nearest', alpha=0.9
    )
    ax.contour(A_init, levels=CONTOUR_LEVELS, colors=CONTOUR_COLORS,
               linewidths=1.2, origin='lower')
    return im_n, im_o


def setup_resistance_panel(ax, title_str, A_init, N_field,
                           mic_min, mic_max, show_ylabel=True):
    """nutrient background + MIC heat-map overlay + colorbar."""
    ax.set_title(title_str)
    ax.set_xlabel('x (gradient direction)')
    if show_ylabel:
        ax.set_ylabel('y')

    im_n = ax.imshow(
        N_field, cmap='Greens', vmin=0, vmax=N_init,
        origin='lower', aspect='auto', interpolation='nearest'
    )
    blank = np.ma.masked_all((Ny, Nx))
    im_o  = ax.imshow(
        blank, cmap='hot', vmin=mic_min, vmax=mic_max,
        origin='lower', aspect='auto', interpolation='nearest', alpha=0.9
    )
    ax.contour(A_init, levels=CONTOUR_LEVELS, colors=CONTOUR_COLORS,
               linewidths=1.2, origin='lower')
    plt.colorbar(im_o, ax=ax, label='MIC')
    return im_n, im_o


def redraw_contours(ax, A):
    for coll in ax.collections:
        coll.remove()
    ax.contour(A, levels=CONTOUR_LEVELS, colors=CONTOUR_COLORS,
               linewidths=1.2, origin='lower')


def get_overlay(bacteria_2d, lineage_2d):
    return np.ma.masked_where(~bacteria_2d, lineage_2d % 20)


# ══════════════════════════════════════════════════════════════════════════════
# CA-only mode
# ══════════════════════════════════════════════════════════════════════════════

def run_ca():
    A, N_field, A_bl, A_br, A_init, sol_A, sol_N = make_pde_fields()
    sim = ca_module.CASimulation(PARAMS, A, N_field, sol_A, sol_N, A_bl, A_br)

    mic_min = PARAMS['MIC_base']
    mic_max = PARAMS['MIC_base'] * 3000

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im_n1, im_lin   = setup_lineage_panel(
        axes[0], 'CA — lineage map', A_init, sim.N_field)
    im_n2, im_mic_o = setup_resistance_panel(
        axes[1], 'CA — resistance map', A_init, sim.N_field,
        mic_min, mic_max, show_ylabel=False)

    title = fig.suptitle('t = 0  |  CA  |  bacteria: 2  |  mutations: 0')

    def update(frame):
        for _ in range(PARAMS['steps_per_frame']):
            sim.step()

        b2d = sim.bacteria_2d
        im_n1.set_data(sim.N_field)
        im_n2.set_data(sim.N_field)
        im_lin.set_data(get_overlay(b2d, sim.lineage_2d))
        im_mic_o.set_data(np.ma.masked_where(~b2d, sim.mic_2d))

        for ax in axes:
            redraw_contours(ax, sim.A)

        title.set_text(
            f't = {sim.step_count}  |  CA'
            f'  |  bacteria: {sim.n_bacteria}'
            f'  |  mutations: {sim.n_mutations}'
        )
        return [im_n1, im_n2, im_lin, im_mic_o, title]

    ani = animation.FuncAnimation(
        fig, update, interval=PARAMS['interval_ms'], cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CPM-only mode  — two panels: lineage map + resistance map
# ══════════════════════════════════════════════════════════════════════════════

def run_cpm():
    A, N_field, A_bl, A_br, A_init, sol_A, sol_N = make_pde_fields()
    _reset_cpm(PARAMS)

    mic_min = PARAMS['MIC_base']
    mic_max = PARAMS['MIC_base'] * 50
    step_n  = [0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im_n1, im_lin   = setup_lineage_panel(
        axes[0], 'CPM — lineage map', A_init, N_field)
    im_n2, im_mic_o = setup_resistance_panel(
        axes[1], 'CPM — resistance map', A_init, N_field,
        mic_min, mic_max, show_ylabel=False)

    title = fig.suptitle('t = 0  |  CPM  |  cells: 2  |  mutations: 0')

    def update(frame):
        for _ in range(PARAMS['steps_per_frame']):
            # 1. advance PDE
            _step_cpm_pde(A, N_field, A_bl, A_br, sol_A, sol_N)

            # 2. advance CPM
            cpm_module.mcs_step(N_field, A)
            # FIX: use do_mutations() — called once per MCS, not per sub-step
            cpm_module.do_mutations()
            cpm_module.kill_exposed_cells(A)
            step_n[0] += 1

        # 3. rebuild display grids using the new helper functions
        b2d    = cpm_module.get_bacteria_grid()
        lin_2d = cpm_module.get_lineage_grid()
        mic_2d = cpm_module.get_mic_grid()

        im_n1.set_data(N_field)
        im_n2.set_data(N_field)
        im_lin.set_data(np.ma.masked_where(~b2d, lin_2d % 20))
        im_mic_o.set_data(np.ma.masked_where(~b2d, mic_2d))

        for ax in axes:
            redraw_contours(ax, A)

        title.set_text(
            f't = {step_n[0]}  |  CPM'
            f'  |  cells: {cpm_module.n_cells()}'
            f'  |  mutations: {cpm_module.n_mutations()}'
        )
        return [im_n1, im_n2, im_lin, im_mic_o, title]

    ani = animation.FuncAnimation(
        fig, update, interval=PARAMS['interval_ms'], cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# compare mode — CA and CPM side by side (2 rows × 2 cols)
#   row 0: CA lineage  | CPM lineage
#   row 1: CA MIC map  | CPM MIC map
# ══════════════════════════════════════════════════════════════════════════════

def run_compare():
    # CA — own PDE fields
    A_ca, N_ca, A_bl_ca, A_br_ca, A_init_ca, sol_A_ca, sol_N_ca = make_pde_fields()
    sim_ca = ca_module.CASimulation(
        PARAMS, A_ca, N_ca, sol_A_ca, sol_N_ca, A_bl_ca, A_br_ca
    )

    # CPM — own PDE fields, fresh module state
    A_cpm, N_cpm, A_bl_cpm, A_br_cpm, A_init_cpm, sol_A_cpm, sol_N_cpm = make_pde_fields()
    _reset_cpm(PARAMS)

    mic_min  = PARAMS['MIC_base']
    mic_max  = PARAMS['MIC_base'] * 50
    cpm_step = [0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    ax_ca_lin, ax_cpm_lin = axes[0]
    ax_ca_mic, ax_cpm_mic = axes[1]

    im_ca_n1,  im_ca_lin   = setup_lineage_panel(
        ax_ca_lin,  'CA — lineage map',    A_init_ca,  sim_ca.N_field)
    im_cpm_n1, im_cpm_lin  = setup_lineage_panel(
        ax_cpm_lin, 'CPM — lineage map',   A_init_cpm, N_cpm, show_ylabel=False)
    im_ca_n2,  im_ca_mic_o = setup_resistance_panel(
        ax_ca_mic,  'CA — resistance map', A_init_ca,  sim_ca.N_field,
        mic_min, mic_max)
    im_cpm_n2, im_cpm_mic_o = setup_resistance_panel(
        ax_cpm_mic, 'CPM — resistance map', A_init_cpm, N_cpm,
        mic_min, mic_max, show_ylabel=False)

    title = fig.suptitle('t = 0')

    def update(frame):
        # ── CA ──────────────────────────────────────────────────────────────
        for _ in range(PARAMS['steps_per_frame']):
            sim_ca.step()

        b2d_ca = sim_ca.bacteria_2d
        im_ca_n1.set_data(sim_ca.N_field)
        im_ca_n2.set_data(sim_ca.N_field)
        im_ca_lin.set_data(get_overlay(b2d_ca, sim_ca.lineage_2d))
        im_ca_mic_o.set_data(np.ma.masked_where(~b2d_ca, sim_ca.mic_2d))
        for ax in [ax_ca_lin, ax_ca_mic]:
            redraw_contours(ax, sim_ca.A)

        # ── CPM ─────────────────────────────────────────────────────────────
        for _ in range(PARAMS['steps_per_frame']):
            # FIX: advance PDE once per MCS step, not once per frame
            _step_cpm_pde(A_cpm, N_cpm, A_bl_cpm, A_br_cpm, sol_A_cpm, sol_N_cpm)
            cpm_module.mcs_step(N_cpm, A_cpm)
            # FIX: do_mutations() replaces the per-sub-step try_mutate loop
            cpm_module.do_mutations()
            cpm_module.kill_exposed_cells(A_cpm)
            cpm_step[0] += 1

        b2d_cpm = cpm_module.get_bacteria_grid()
        lin_2d  = cpm_module.get_lineage_grid()
        mic_2d  = cpm_module.get_mic_grid()

        im_cpm_n1.set_data(N_cpm)
        im_cpm_n2.set_data(N_cpm)
        im_cpm_lin.set_data(np.ma.masked_where(~b2d_cpm, lin_2d % 20))
        im_cpm_mic_o.set_data(np.ma.masked_where(~b2d_cpm, mic_2d))
        for ax in [ax_cpm_lin, ax_cpm_mic]:
            redraw_contours(ax, A_cpm)

        title.set_text(
            f't = {sim_ca.step_count}'
            f'  |  CA — bacteria: {sim_ca.n_bacteria}'
            f'  mutations: {sim_ca.n_mutations}'
            f'  |  CPM — cells: {cpm_module.n_cells()}'
            f'  mutations: {cpm_module.n_mutations()}'
        )
        return [im_ca_n1, im_ca_n2, im_ca_lin, im_ca_mic_o,
                im_cpm_n1, im_cpm_n2, im_cpm_lin, im_cpm_mic_o, title]

    ani = animation.FuncAnimation(
        fig, update, interval=PARAMS['interval_ms'], cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if MODE == 'ca':
        run_ca()
    elif MODE == 'cpm':
        run_cpm()
    elif MODE == 'compare':
        run_compare()
    else:
        raise ValueError(f'unknown MODE: {MODE}')