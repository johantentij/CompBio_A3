import numpy as np
from pde import get_antibiotic_at, update_rho_from_CA, Nx, Ny


def monod(x, K):
    return x / (x + K)


class CASimulation:
    """
    cellular automaton simulation of bacterial spreading and evolution.
    all parameters passed in via a params dict — no hardcoded values.
    no matplotlib imports; visualisation handled by main.py.
    """

    def __init__(self, params, A, N_field, solver_A, solver_N,
                 A_boundary_left, A_boundary_right):
        self.p               = params
        self.A               = A
        self.N_field         = N_field
        self.solver_A        = solver_A
        self.solver_N        = solver_N
        self.A_boundary_left  = A_boundary_left
        self.A_boundary_right = A_boundary_right

        # bacteria state
        self.bacteria         = np.zeros(Nx * Ny, dtype=np.bool_)
        self.bacteria_indices = []
        self.bacteria_mic     = []
        self.bacteria_lineage = []
        self.next_lineage_id  = 0

        # visualisation grids
        self.mic_grid     = np.zeros(Nx * Ny, dtype=np.float32)
        self.lineage_grid = np.full(Nx * Ny, -1, dtype=np.int32)

        self.neighbourhood = np.array(
            [0, -Nx, Nx, -1, -Nx - 1, Nx - 1, 1, -Nx + 1, Nx + 1]
        )

        self.step_count = 0
        self._seed()

    def _seed(self):
        """inoculate from left and right edges."""
        for flat_pos in [
            (Ny // 2) * Nx,           # left edge centre
            (Ny // 2) * Nx + Nx - 1,  # right edge centre
        ]:
            self._add_bacterium(flat_pos, self.p['MIC_base'],
                                self.next_lineage_id)
            self.next_lineage_id += 1

    def _add_bacterium(self, pos, mic, lineage):
        self.bacteria[pos]     = True
        self.mic_grid[pos]     = mic
        self.lineage_grid[pos] = lineage
        self.bacteria_indices.append(pos)
        self.bacteria_mic.append(mic)
        self.bacteria_lineage.append(lineage)

    def _remove_bacterium(self, pos):
        self.bacteria[pos]     = False
        self.mic_grid[pos]     = 0.0
        self.lineage_grid[pos] = -1

    def _mutate_mic(self, parent_mic):
        fold = np.random.lognormal(
            mean=np.log(self.p['mic_fold_mean']),
            sigma=self.p['mic_fold_sigma']
        )
        return parent_mic * fold

    def step(self):
        p = self.p

        # advance PDE fields
        rho_flat = update_rho_from_CA(self.bacteria_2d).ravel()

        # --- Update Antibiotic A (Diffusion-Reaction) ---
        A_diffused = self.solver_A(self.A.ravel())
        A_flat = A_diffused - p['dt'] * p['delta_A'] * rho_flat * A_diffused
        A_flat = np.maximum(A_flat, 0.0)

        # --- Update Nutrient N (Diffusion-Reaction) ---
        N_diffused = self.solver_N(self.N_field.ravel())
        N_flat = N_diffused - p['dt'] * p['alpha'] * rho_flat * (N_diffused / (p['K_N'] + N_diffused))
        N_flat = np.maximum(N_flat, 0.0)

        # restore Dirichlet antibiotic boundaries
        A_2d = A_flat.reshape(Ny, Nx)
        A_2d[:, 0]  = self.A_boundary_left
        A_2d[:, -1] = self.A_boundary_right
        self.A       = A_2d
        self.N_field[:] = N_flat.reshape(Ny, Nx)

        antibiotic_vals = get_antibiotic_at(self.A, self.bacteria_indices)

        death = []
        for i in range(len(self.bacteria_indices)):
            b_pos     = self.bacteria_indices[i]
            b_mic     = self.bacteria_mic[i]
            b_lineage = self.bacteria_lineage[i]

            # antibiotic kill
            if antibiotic_vals[i] > b_mic:
                self._remove_bacterium(b_pos)
                death.append(i)
                continue

            # starvation
            if N_flat[b_pos] <= 0:
                self._remove_bacterium(b_pos)
                death.append(i)
                continue

            # neighbourhood
            region = self.neighbourhood + b_pos
            region = region[(region >= 0) & (region < Nx * Ny)]
            col, row = b_pos % Nx, b_pos // Nx
            region = region[
                (np.abs(region % Nx  - col) <= 1) &
                (np.abs(region // Nx - row) <= 1)
            ]

            free = region[~self.bacteria[region]]
            if free.shape[0] == 0:
                continue

            crowding       = free.shape[0] / len(region)
            p_repro        = (p['p_repro_base']
                              * monod(N_flat[b_pos], p['nutrient_consumption'])
                              * crowding)

            if np.random.random() > p_repro:
                continue

            # directional bias toward antibiotic gradient (inward)
            a_vals  = get_antibiotic_at(self.A, free.tolist())
            weights = np.where(a_vals <= b_mic, a_vals + 0.1, 0.0)
            if weights.sum() == 0:
                weights = np.ones(len(free))
            weights /= weights.sum()

            new_pos = int(np.random.choice(free, 1, p=weights))

            # mutation
            if np.random.random() < p['mu_base']:
                new_mic     = self._mutate_mic(b_mic)
                new_lineage = self.next_lineage_id
                self.next_lineage_id += 1
            else:
                new_mic     = b_mic
                new_lineage = b_lineage

            self._add_bacterium(new_pos, new_mic, new_lineage)

        for i in reversed(death):
            self.bacteria_indices.pop(i)
            self.bacteria_mic.pop(i)
            self.bacteria_lineage.pop(i)

        self.step_count += 1

    # ── read-only properties for main.py ──────────────────────────────────────

    @property
    def bacteria_2d(self):
        return self.bacteria.reshape(Ny, Nx, order='C')

    @property
    def mic_2d(self):
        return self.mic_grid.reshape(Ny, Nx, order='C')

    @property
    def lineage_2d(self):
        return self.lineage_grid.reshape(Ny, Nx, order='C')

    @property
    def n_bacteria(self):
        return len(self.bacteria_indices)

    @property
    def n_mutations(self):
        return max(0, self.next_lineage_id - 2)