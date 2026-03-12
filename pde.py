import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ============================================================
# 1. Grid and Parameter Setup
# ============================================================

# Spatial grid (corresponds to a 120x60cm MEGA-plate, each grid cell = 0.6cm)
Nx = 200
Ny = 100
dx = 1.0
dy = 1.0
dt = 0.05

# Antibiotic parameters
D_A = 0.005         # FIX: lowered 10x (was 0.05) — keeps antibiotic zone boundaries
                    # sharp so wild-type bacteria are truly blocked at each band edge;
                    # with D_A=0.05 the gradient was smoothed out and middle=0 was
                    # indistinguishable from middle=300 in the Fig.2B experiment
delta_A = 0.002     # Degradation rate of antibiotic by bacteria

# Nutrient parameters
D_N = 3           # Nutrient diffusion coefficient (large, for rapid diffusion)
N_init = 10.0       # Initial nutrient concentration
alpha = 0.005       # Maximum consumption rate
K_N = 1.0           # Monod half-saturation constant

# MEGA-plate antibiotic gradient (in units of MIC, symmetric four-step gradient)
# Corresponds to papers like: TMP: 0, 3, 30, 300, 3000, 300, 30, 3, 0 × MIC
MIC_base = 1.0      # Wild-type MIC baseline
antibiotic_levels = np.array([0, 3, 30, 300, 3000, 300, 30, 3, 0]) * MIC_base


# ============================================================
# 2. Initial Conditions
# ============================================================

def init_antibiotic(Nx, Ny, levels=antibiotic_levels):
    """
    Initializes the antibiotic concentration field.
    Divides the x-axis into len(levels) zones, each with a specific concentration.
    This mimics the step-wise gradient design of a MEGA-plate.
    """
    A = np.zeros((Ny, Nx))
    n_zones = len(levels)
    zone_width = Nx // n_zones

    for z, level in enumerate(levels):
        x_start = z * zone_width
        x_end = (z + 1) * zone_width if z < n_zones - 1 else Nx
        A[:, x_start:x_end] = level

    return A


def init_nutrients(Nx, Ny, N_init=N_init):
    """
    Initializes the nutrient concentration field.
    The field is uniform, as the agar contains nutrients evenly at the start.
    """
    return N_init * np.ones((Ny, Nx))


def init_bacteria_density(Nx, Ny):
    """
    Initializes the bacterial density field (continuous approximation).
    Inoculates from the left and right edges, matching the MEGA-plate setup.
    The density is smoothed with a Gaussian to avoid numerical discontinuities.
    """
    rho = np.zeros((Ny, Nx))
    sigma = 2.0    # Initial distribution width (in grid cells)
    x = np.arange(Nx)

    # Left edge
    rho += np.exp(-((x[np.newaxis, :]) ** 2) / (2 * sigma ** 2))
    # Right edge
    rho += np.exp(-((x[np.newaxis, :] - (Nx - 1)) ** 2) / (2 * sigma ** 2))

    rho *= 0.1

    return rho


# ============================================================
# 3. Laplacian Operator Construction
# ============================================================

def build_1d_laplacian(N, bc_left='neumann', bc_right='neumann'):
    """
    Builds a 1D Laplacian matrix with support for two boundary conditions:
    - 'neumann'  : Zero-flux (no-flux), used for nutrients and y-direction.
    - 'dirichlet': Fixed value (constant concentration), for antibiotic x-boundaries.
    """
    diag_main = -2 * np.ones(N)
    diag_off  = np.ones(N - 1)
    L = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='lil')

    # Left boundary
    if bc_left == 'neumann':
        L[0, 1] = 2       # ∂u/∂x=0 -> mirror image
    elif bc_left == 'dirichlet':
        L[0, 0] = -2      # Fixed value, no change (ghost cell handling)

    # Right boundary
    if bc_right == 'neumann':
        L[N-1, N-2] = 2
    elif bc_right == 'dirichlet':
        L[N-1, N-1] = -2

    return L.tocsr() / (dx ** 2)


def build_2d_laplacian(Nx, Ny,
                        bc_x=('neumann', 'neumann'),
                        bc_y=('neumann', 'neumann')):
    """
    Builds a 2D Laplacian using the Kronecker product.
    bc_x: (left_boundary, right_boundary) type
    bc_y: (bottom_boundary, top_boundary) type
    Fields are stored as (Ny, Nx), flattened in row-major (C order).
    """
    Lx = build_1d_laplacian(Nx, bc_left=bc_x[0], bc_right=bc_x[1])
    Ly = build_1d_laplacian(Ny, bc_left=bc_y[0], bc_right=bc_y[1])

    Ix = sp.eye(Nx, format='csr')
    Iy = sp.eye(Ny, format='csr')

    # ∇² = ∂²/∂x² ⊗ I_y + I_x ⊗ ∂²/∂y²
    L2D = sp.kron(Iy, Lx) + sp.kron(Ly, Ix)
    return L2D


# Pre-build Laplacians to avoid reconstruction in each step
# Antibiotic: Dirichlet on x-boundaries (fixed concentration), Neumann on y-boundaries
L2D_A = build_2d_laplacian(Nx, Ny,
                             bc_x=('dirichlet', 'dirichlet'),
                             bc_y=('neumann', 'neumann'))

# Nutrient: Neumann on all boundaries (no-flux)
L2D_N = build_2d_laplacian(Nx, Ny,
                             bc_x=('neumann', 'neumann'),
                             bc_y=('neumann', 'neumann'))


# ============================================================
# 4. Implicit Solver (for stable diffusion)
# ============================================================

def build_implicit_solver(L2D, D, dt, N_total):
    """
    Builds the matrix for an implicit Euler diffusion solver:
    (I - dt·D·L) u_{n+1} = u_n
    Returns a pre-factorized matrix to avoid repeated LU decomposition.
    """
    I = sp.eye(N_total, format='csr')
    M = (I - dt * D * L2D).tocsc()   # FIX: convert to CSC before factorization
    return spla.factorized(M)         # splu requires CSC; pre-convert avoids warning


# ============================================================
# 5. PDE Right-Hand Side (Reaction Terms)
# ============================================================

def rhs_antibiotic_reaction(A_flat, rho_flat):
    """
    Antibiotic reaction term (reaction only, diffusion is handled implicitly):
    reaction = -δ_A · ρ · A
    Bacteria slightly degrade the antibiotic.
    """
    return -delta_A * rho_flat * A_flat


def rhs_nutrients_reaction(N_flat, rho_flat):
    """
    Nutrient reaction term (reaction only, diffusion is handled implicitly):
    reaction = -α · ρ · N/(K_N + N)    (Monod动力学)
    Bacteria consume nutrients; the rate saturates when nutrients are abundant.
    """
    monod = N_flat / (K_N + N_flat)
    return -alpha * rho_flat * monod


# ============================================================
# 6. Time Stepping (Operator Splitting)
# ============================================================
#
# Operator Splitting Method:
#   Step 1 — Implicit diffusion (unconditionally stable)
#   Step 2 — Explicit reaction (stable for small enough dt)
#
# This is much more stable than a purely explicit Euler method, especially
# for coupled fields with very different diffusion coefficients (D).

def step_antibiotic(A_flat, rho_flat, solver_A):
    """
    Advances the antibiotic field by one time step.
    solver_A: Pre-factorized implicit diffusion solver.
    """
    # Step 1: Implicit diffusion
    A_new = solver_A(A_flat)
    # Step 2: Explicit reaction
    A_new += dt * rhs_antibiotic_reaction(A_new, rho_flat)
    # Physical constraint: concentration must be non-negative
    A_new = np.maximum(A_new, 0.0)
    return A_new


def step_nutrients(N_flat, rho_flat, solver_N):
    """
    Advances the nutrient field by one time step.
    solver_N: Pre-factorized implicit diffusion solver.
    """
    # Step 1: Implicit diffusion
    N_new = solver_N(N_flat)
    # Step 2: Explicit reaction (Monod consumption)
    N_new += dt * rhs_nutrients_reaction(N_new, rho_flat)
    N_new = np.maximum(N_new, 0.0)
    return N_new


# ============================================================
# 7. Interface Functions for CA Coupling
# ============================================================

def get_antibiotic_at(A, positions):
    """
    Gets the antibiotic concentration at specified grid points for the CA.
    A        : (Ny, Nx) antibiotic field
    positions: list of (row, col) or flat indices
    Returns an array of antibiotic concentrations for each position.
    """
    if len(positions) == 0:
        return np.array([])

    positions = np.array(positions)

    # 支持两种输入格式
    if positions.ndim == 1:
        # flat index -> 2D
        rows = positions // Nx
        cols = positions % Nx
    else:
        rows = positions[:, 0]
        cols = positions[:, 1]

    rows = np.clip(rows, 0, Ny - 1)
    cols = np.clip(cols, 0, Nx - 1)

    return A[rows, cols]


def get_nutrient_gradient_at(N, positions):
    """
    Gets the nutrient gradient direction at specified grid points for the CA/CPM.
    Returns (grad_y, grad_x) for each position, used for chemotaxis decisions.
    This is vectorized using np.gradient for performance.
    """
    if len(positions) == 0:
        return np.array([])

    positions = np.array(positions)
    if positions.ndim == 1:
        rows, cols = np.unravel_index(positions, (Ny, Nx))
    else:
        rows = positions[:, 0]
        cols = positions[:, 1]

    # Calculate gradient for the entire field efficiently.
    # np.gradient handles boundary conditions correctly (1st order at boundary, 2nd order in interior).
    grad_y_full, grad_x_full = np.gradient(N, dy, dx)

    # Sample the gradients at the specified positions.
    grad_x = grad_x_full[rows, cols]
    grad_y = grad_y_full[rows, cols]

    return np.stack([grad_x, grad_y], axis=1)


def update_rho_from_CA(bacteria_grid, sigma=1.0):
    """
    Converts the discrete CA bacteria distribution to a continuous density field ρ.
    This density field is used in the PDE reaction terms.
    bacteria_grid: (Ny, Nx) boolean array from the CA.
    sigma        : Gaussian smoothing radius (in grid cells), simulating a finite volume effect.
    """
    from scipy.ndimage import gaussian_filter
    rho = bacteria_grid.astype(np.float64)
    if sigma > 0:
        rho = gaussian_filter(rho, sigma=sigma)
    return rho


# ============================================================
# 8. Visualization
# ============================================================

def plot_fields(A, N, rho=None, title_suffix=""):
    """
    Displays the antibiotic, nutrient, and (optionally) bacteria density fields side-by-side.
    """
    n_plots = 3 if rho is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    im0 = axes[0].imshow(A, origin='lower', cmap='Reds',
                          aspect='auto', vmin=0)
    axes[0].set_title(f'Antibiotic A {title_suffix}')
    axes[0].set_xlabel('x (gradient direction)')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='Concentration (×MIC)')

    im1 = axes[1].imshow(N, origin='lower', cmap='Greens',
                          aspect='auto', vmin=0, vmax=N_init)
    axes[1].set_title(f'Nutrient N {title_suffix}')
    axes[1].set_xlabel('x')
    plt.colorbar(im1, ax=axes[1], label='Concentration')

    if rho is not None:
        im2 = axes[2].imshow(rho, origin='lower', cmap='Blues',
                              aspect='auto', vmin=0)
        axes[2].set_title(f'Bacteria density ρ {title_suffix}')
        axes[2].set_xlabel('x')
        plt.colorbar(im2, ax=axes[2], label='Density')

    plt.tight_layout()
    return fig


# ============================================================
# 9. Main Loop Example (for running pde.py standalone)
# ============================================================

if __name__ == "__main__":

    # --- Initialization ---
    A   = init_antibiotic(Nx, Ny)
    N   = init_nutrients(Nx, Ny)
    rho = init_bacteria_density(Nx, Ny)

    # Save initial antibiotic boundary values (needed to restore Dirichlet conditions)
    A_boundary_left  = A[:, 0].copy()
    A_boundary_right = A[:, -1].copy()

    # Pre-build implicit solvers (only needs to be done once)
    N_total  = Nx * Ny
    solver_A = build_implicit_solver(L2D_A, D_A, dt, N_total)
    solver_N = build_implicit_solver(L2D_N, D_N, dt, N_total)

    # Visualize initial state
    plot_fields(A, N, rho, title_suffix="(t=0)")
    plt.show()

    # --- Time Stepping ---
    n_steps = 500
    plot_every = 100

    for step in range(n_steps):

        # Flatten to 1D for sparse matrix operations
        A_flat   = A.ravel()
        N_flat   = N.ravel()
        rho_flat = rho.ravel()

        # Advance each field
        A_flat = step_antibiotic(A_flat, rho_flat, solver_A)
        N_flat = step_nutrients(N_flat, rho_flat, solver_N)

        # Restore Dirichlet boundaries (implicit solver doesn't maintain them automatically)
        A_flat_2d = A_flat.reshape(Ny, Nx)
        A_flat_2d[:, 0]  = A_boundary_left
        A_flat_2d[:, -1] = A_boundary_right
        A_flat = A_flat_2d.ravel()

        # Reshape back to 2D
        A   = A_flat.reshape(Ny, Nx)
        N   = N_flat.reshape(Ny, Nx)

        # Simple update for rho (in real use, this comes from the CA)
        # Example: bacterial density grows with nutrient consumption
        growth = 0.01 * (N / (K_N + N)) * (1 - rho / 1.0)
        rho = np.maximum(rho + dt * growth, 0.0)

        if (step + 1) % plot_every == 0:
            plot_fields(A, N, rho, title_suffix=f"(t={step+1})")
            plt.show()
            print(f"Step {step+1}/{n_steps} | "
                  f"A_max={A.max():.2f} | "
                  f"N_mean={N.mean():.3f} | "
                  f"rho_max={rho.max():.4f}")

    print("Done.")