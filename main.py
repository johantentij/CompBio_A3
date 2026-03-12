import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

from pde import (
    init_antibiotic, init_nutrients,
    build_implicit_solver,
    L2D_A, L2D_N,
    Nx, Ny, N_init
)
import ca_2 as ca_module
import cpm as cpm_module

# ══════════════════════════════════════════════════════════════════════════════
# parameters — edit here to tune both models simultaneously
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = {
    # grid / time
    'dt'                  : 0.05,

    # antibiotic PDE
    'D_A'                 : 0.02,
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
  
    'mu_base_ca'          : 0.005,
    'mu_base_cpm'         : 0.05,
    'mic_fold_mean'       : 1.2,
    'mic_fold_sigma'      : 0.3,

    # CPM energy
    'J_medium'            : 8.0,
    'J_diff'              : 16.0,
    'J_same'              : 2.0,
    'lambda_V'            : 0.5,

    # size before competing; small V_target caused runaway cell proliferation
    'V_target'            : 25,
    'chi'                 : 3.0,
    'T_cpm'               : 5.0,

    # animation
    'steps_per_frame'     : 20,
    'interval_ms'         : 40,
}

# ── run mode ──────────────────────────────────────────────────────────────────
# 'ca'      — run CA only
# 'cpm'     — run CPM only
# 'compare' — run both side by side
MODE = 'cpm'

# ══════════════════════════════════════════════════════════════════════════════
# shared PDE initialisation
# ══════════════════════════════════════════════════════════════════════════════

def make_pde_fields():
    A       = init_antibiotic(Nx, Ny)
    N_field = init_nutrients(Nx, Ny)
    A_bl    = A[:, 0].copy()
    A_br    = A[:, -1].copy()
    # save initial A for static contour drawing
    A_init  = A.copy()
    sol_A   = build_implicit_solver(L2D_A, PARAMS['D_A'], PARAMS['dt'], Nx * Ny)
    sol_N   = build_implicit_solver(L2D_N, PARAMS['D_N'], PARAMS['dt'], Nx * Ny)
    return A, N_field, A_bl, A_br, A_init, sol_A, sol_N


def _cpm_seed_pixels(col):
    """return a small block of pixels to seed a CPM founder cell."""
    r0 = Ny // 2
    return [
        (r0 + dr, col)
        for dr in range(-3, 4)   # 7-pixel tall column seed
    ]


def _reset_cpm(params):
    """reset all cpm module globals and re-seed founder cells."""
    cpm_module.cell_id[:]    = cpm_module.MEDIUM
    cpm_module.cell_mic.clear()
    cpm_module.cell_lineage.clear()
    cpm_module.cell_volume.clear()
    cpm_module.cell_mic[cpm_module.MEDIUM]     = 0.0
    cpm_module.cell_lineage[cpm_module.MEDIUM] = -1
    cpm_module.cell_volume[cpm_module.MEDIUM]  = Nx * Ny
    cpm_module.next_cell_id = 1

    cpm_module.J_medium      = params['J_medium']
    cpm_module.J_diff        = params['J_diff']
    cpm_module.J_same        = params['J_same']
    cpm_module.lambda_V      = params['lambda_V']
    cpm_module.V_target      = params['V_target']
    cpm_module.chi           = params['chi']
    cpm_module.T             = params['T_cpm']
    cpm_module.mu_base       = params['mu_base_cpm']
    cpm_module.mic_fold_mean  = params['mic_fold_mean']
    cpm_module.mic_fold_sigma = params['mic_fold_sigma']

    cpm_module.add_cell(_cpm_seed_pixels(0),      params['MIC_base'], 0)
    cpm_module.add_cell(_cpm_seed_pixels(Nx - 1), params['MIC_base'], 1)


# ══════════════════════════════════════════════════════════════════════════════
# visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

CONTOUR_LEVELS = [3, 30, 300, 3000]

# Pick contour colours directly from the 'hot' colormap at the log-scale
# positions of each level.  A cell whose MIC matches the line colour can
# just barely survive on that side of the boundary — visually, only cells
# brighter than the line can cross it.
_MIC_MIN  = 1.0          # MIC_base
_MIC_MAX  = 3000.0       # highest antibiotic level
_hot      = plt.cm.hot

def _contour_color(level):
    """Map an antibiotic level to its 'hot' colormap colour at log scale."""
    import math
    t = (math.log10(level) - math.log10(_MIC_MIN)) / \
        (math.log10(_MIC_MAX) - math.log10(_MIC_MIN))
    return _hot(t)

CONTOUR_COLORS = [_contour_color(l) for l in CONTOUR_LEVELS]
# linewidths: thicker for higher concentrations so they stand out
CONTOUR_WIDTHS = [1.0, 1.2, 1.5, 2.0]


def setup_panel(ax, title_str, A_init, N_field, show_ylabel=True):
    """
    Initialise a lineage-map panel (tab20 overlay on green nutrient background).
    Returns (im_nutrient, im_overlay).
    """
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
               linewidths=CONTOUR_WIDTHS, origin='lower')
    return im_n, im_o


def setup_resistance_panel(ax, title_str, A_init, MIC_base, show_ylabel=False):
    """
    Initialise a resistance-map panel with log-scale 'hot' colormap.
    Contour line colours are sampled from the same colormap so a cell must
    be at least as bright as the line to survive on that side of it.
    Returns (im_nutrient, im_mic_overlay, colorbar).
    """
    ax.set_title(title_str)
    ax.set_xlabel('x (gradient direction)')
    if show_ylabel:
        ax.set_ylabel('y')

    im_n = ax.imshow(
        np.zeros((Ny, Nx)), cmap='Greens', vmin=0, vmax=N_init,
        origin='lower', aspect='auto', interpolation='nearest'
    )
    blank   = np.ma.masked_all((Ny, Nx))
    log_norm = LogNorm(vmin=MIC_base, vmax=MIC_base * 3000)
    im_mic  = ax.imshow(
        blank, cmap='hot', norm=log_norm,
        origin='lower', aspect='auto', interpolation='nearest', alpha=0.95
    )
    ax.contour(A_init, levels=CONTOUR_LEVELS, colors=CONTOUR_COLORS,
               linewidths=CONTOUR_WIDTHS, origin='lower')

    cb = ax.get_figure().colorbar(im_mic, ax=ax, label='MIC')
    # put log ticks at the concentration boundaries so they align with the lines
    import matplotlib.ticker as ticker
    cb.set_ticks([1, 3, 30, 300, 3000])
    cb.set_ticklabels(['1', '3', '30', '300', '3000'])

    return im_n, im_mic, cb


def redraw_contours(ax, A):
    for coll in ax.collections:
        coll.remove()
    ax.contour(A, levels=CONTOUR_LEVELS, colors=CONTOUR_COLORS,
               linewidths=CONTOUR_WIDTHS, origin='lower')


def get_overlay(bacteria_2d, lineage_2d):
    return np.ma.masked_where(~bacteria_2d, lineage_2d % 20)


# ══════════════════════════════════════════════════════════════════════════════
# CA-only mode
# ══════════════════════════════════════════════════════════════════════════════

def run_ca():
    A, N_field, A_bl, A_br, A_init, sol_A, sol_N = make_pde_fields()
    sim = ca_module.CASimulation(PARAMS, A, N_field, sol_A, sol_N, A_bl, A_br)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im_n1, im_lin             = setup_panel(axes[0], 'CA — lineage map',
                                             A_init, sim.N_field)
    im_n2, im_mic_o, _        = setup_resistance_panel(axes[1], 'CA — resistance map',
                                                        A_init, PARAMS['MIC_base'],
                                                        show_ylabel=False)

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
# CPM-only mode
# ══════════════════════════════════════════════════════════════════════════════

def run_cpm():
    A, N_field, A_bl, A_br, A_init, sol_A, sol_N = make_pde_fields()
    _reset_cpm(PARAMS)

    step_n = [0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im_n1, im_lin             = setup_panel(axes[0], 'CPM — lineage map',
                                             A_init, N_field)
    im_n2, im_mic_o, _        = setup_resistance_panel(axes[1], 'CPM — resistance map',
                                                        A_init, PARAMS['MIC_base'],
                                                        show_ylabel=False)

    title = fig.suptitle('t = 0  |  CPM  |  cells: 2  |  mutations: 0')

    def update(frame):
        for _ in range(PARAMS['steps_per_frame']):
            # --- 1. Update PDE fields ---
            rho_flat = cpm_module.get_rho_grid().ravel()

            A_diffused = sol_A(A.ravel())
            A_flat = A_diffused - PARAMS['dt'] * PARAMS['delta_A'] * rho_flat * A_diffused
            A_flat = np.maximum(A_flat, 0.0)

            N_diffused = sol_N(N_field.ravel())
            N_flat = N_diffused - PARAMS['dt'] * PARAMS['alpha'] * rho_flat * (N_diffused / (PARAMS['K_N'] + N_diffused))
            N_flat = np.maximum(N_flat, 0.0)

            A_2d = A_flat.reshape(Ny, Nx)
            A_2d[:, 0]  = A_bl
            A_2d[:, -1] = A_br
            A[:]        = A_2d
            N_field[:]  = N_flat.reshape(Ny, Nx)

            # --- 2. Update CPM state ---
            cpm_module.mcs_step(N_field, A)
            cpm_module.do_mutations()
            cpm_module.kill_exposed_cells(A)
            step_n[0] += 1

        # --- 3. Update visualisation ---
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
# compare mode — CA and CPM side by side
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

    cpm_step = [0]

    # layout: 2 rows × 2 cols
    # [CA lineage  | CPM lineage ]
    # [CA MIC map  | CPM MIC map ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    ax_ca_lin, ax_cpm_lin, ax_ca_mic, ax_cpm_mic = axes.flat

    # CA panels
    im_ca_n1, im_ca_lin            = setup_panel(ax_ca_lin, 'CA — lineage map',
                                                  A_init_ca, sim_ca.N_field)
    im_ca_n2, im_ca_mic_o, _       = setup_resistance_panel(ax_ca_mic, 'CA — resistance map',
                                                             A_init_ca, PARAMS['MIC_base'])

    # CPM panels
    im_cpm_n1, im_cpm_lin          = setup_panel(ax_cpm_lin, 'CPM — lineage map',
                                                  A_init_cpm, N_cpm, show_ylabel=False)
    im_cpm_n2, im_cpm_mic_o, _     = setup_resistance_panel(ax_cpm_mic, 'CPM — resistance map',
                                                             A_init_cpm, PARAMS['MIC_base'],
                                                             show_ylabel=False)

    title = fig.suptitle('t = 0')

    def update(frame):
        # ── CA ──
        for _ in range(PARAMS['steps_per_frame']):
            sim_ca.step()

        b2d_ca = sim_ca.bacteria_2d
        im_ca_n1.set_data(sim_ca.N_field)
        im_ca_n2.set_data(sim_ca.N_field)
        im_ca_lin.set_data(get_overlay(b2d_ca, sim_ca.lineage_2d))
        im_ca_mic_o.set_data(np.ma.masked_where(~b2d_ca, sim_ca.mic_2d))
        for ax in [ax_ca_lin, ax_ca_mic]:
            redraw_contours(ax, sim_ca.A)

        # ── CPM ──
        rho_cpm = cpm_module.get_rho_grid()
        Af = sol_A_cpm(A_cpm.ravel())
        Nf = sol_N_cpm(N_cpm.ravel())
        Nf = np.maximum(
            Nf - PARAMS['dt'] * PARAMS['alpha'] * rho_cpm.ravel()
            * Nf / (PARAMS['K_N'] + Nf), 0.0
        )
        A2 = Af.reshape(Ny, Nx)
        A2[:, 0]  = A_bl_cpm
        A2[:, -1] = A_br_cpm
        A_cpm[:]  = A2
        N_cpm[:]  = Nf.reshape(Ny, Nx)

        for _ in range(PARAMS['steps_per_frame']):
            cpm_module.mcs_step(N_cpm, A_cpm)
            cpm_module.do_mutations()
            cpm_module.kill_exposed_cells(A_cpm)
            cpm_step[0] += 1

        b2d_cpm = cpm_module.get_bacteria_grid()
        lin_cpm = np.zeros((Ny, Nx), dtype=np.int32)
        mic_cpm = np.zeros((Ny, Nx), dtype=np.float32)
        for cid in list(cpm_module.cell_mic.keys()):
            if cid == cpm_module.MEDIUM:
                continue
            mask = cpm_module.cell_id == cid
            lin_cpm[mask] = cpm_module.cell_lineage.get(cid, 0) % 20
            mic_cpm[mask] = cpm_module.cell_mic[cid]

        im_cpm_n1.set_data(N_cpm)
        im_cpm_n2.set_data(N_cpm)
        im_cpm_lin.set_data(np.ma.masked_where(~b2d_cpm, lin_cpm))
        im_cpm_mic_o.set_data(np.ma.masked_where(~b2d_cpm, mic_cpm))
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