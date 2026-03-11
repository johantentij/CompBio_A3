import numpy as np
from pde import get_antibiotic_at, Nx, Ny, MIC_base

# ── CPM parameters ─────────────────────────────────────────────────────────────

J_medium = 8.0    # interface energy between cell and empty space
J_diff   = 16.0   # interface energy between different lineages
J_same   = 2.0    # interface energy between same lineage

lambda_V = 0.5    # volume constraint strength
V_target = 4      # FIX: lowered from 25 → single-pixel mutants no longer crushed
                  # by volume penalty; small clusters can survive and expand

chi      = 3.0    # chemotaxis sensitivity (toward nutrient gradient)
T        = 5.0    # CPM temperature

MEDIUM = 0

# ── state arrays ───────────────────────────────────────────────────────────────

cell_id      = np.zeros((Ny, Nx), dtype=np.int32)
cell_mic     = {0: 0.0}
cell_lineage = {0: -1}
cell_volume  = {0: Nx * Ny}
next_cell_id = 1


def add_cell(pixels, mic, lineage):
    """register a new cell occupying a list of (row, col) pixel positions."""
    global next_cell_id
    cid = next_cell_id
    next_cell_id += 1
    cell_mic[cid]     = mic
    cell_lineage[cid] = lineage
    cell_volume[cid]  = len(pixels)
    for (r, c) in pixels:
        cell_id[r, c] = cid
        cell_volume[0] -= 1
    return cid


def init_cpm_from_CA(bacteria, bacteria_indices, bacteria_mic_list, bacteria_lineage_list):
    """
    seed CPM from an existing CA state.
    FIX: original function ignored bacteria_mic_list / bacteria_lineage_list
    and hard-coded MIC_base / lineage 0 for every pixel.
    Now correctly reads the parallel lists from the CA simulation.
    """
    global cell_id, next_cell_id
    cell_id[:] = MEDIUM
    cell_volume[0] = Nx * Ny
    for k in list(cell_mic.keys()):
        if k != MEDIUM:
            cell_mic.pop(k)
            cell_lineage.pop(k, None)
            cell_volume.pop(k, None)
    next_cell_id = 1

    visited = np.zeros((Ny, Nx), dtype=bool)
    for idx, flat_pos in enumerate(bacteria_indices):
        r, c = flat_pos // Nx, flat_pos % Nx
        if visited[r, c]:
            continue
        visited[r, c] = True
        add_cell([(r, c)], bacteria_mic_list[idx], bacteria_lineage_list[idx])


# ── neighbourhood ──────────────────────────────────────────────────────────────

_nb4 = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def _neighbours4(r, c):
    nb = []
    for dr, dc in _nb4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < Ny and 0 <= nc < Nx:
            nb.append((nr, nc))
    return nb


# ── Hamiltonian terms ──────────────────────────────────────────────────────────

def _interface_energy(r, c, cid):
    energy  = 0.0
    lin_src = cell_lineage.get(cid, -1)
    for nr, nc in _neighbours4(r, c):
        nb_cid = cell_id[nr, nc]
        if nb_cid == cid:
            continue
        if nb_cid == MEDIUM:
            energy += J_medium
        else:
            lin_nb = cell_lineage.get(nb_cid, -1)
            energy += J_same if lin_src == lin_nb else J_diff
    return energy


def _volume_energy(cid, delta):
    """
    FIX: medium is exempt from volume constraint.
    Previously medium's huge pixel count (~Nx*Ny) made dH enormous whenever
    a cell tried to expand into empty space, blocking all colony growth.
    """
    if cid == MEDIUM:
        return 0.0
    v  = cell_volume.get(cid, 0)
    vt = V_target
    return lambda_V * ((v + delta - vt) ** 2 - (v - vt) ** 2)


def _chemotaxis_energy(r, c, N_field):
    return -chi * N_field[r, c]


# ── MCS step ──────────────────────────────────────────────────────────────────

def mcs_step(N_field, A, n_attempts=None):
    """one Monte Carlo sweep of Nx*Ny copy attempts."""
    if n_attempts is None:
        n_attempts = Nx * Ny

    for _ in range(n_attempts):
        r = np.random.randint(0, Ny)
        c = np.random.randint(0, Nx)

        nb = _neighbours4(r, c)
        if not nb:
            continue
        nr, nc = nb[np.random.randint(len(nb))]

        src_cid = cell_id[r, c]
        nb_cid  = cell_id[nr, nc]

        if src_cid == nb_cid:
            continue

        if src_cid != MEDIUM and cell_volume.get(src_cid, 0) <= 1:
            continue

        dH  = 0.0
        dH -= _interface_energy(r, c, src_cid)
        dH += _interface_energy(r, c, nb_cid)
        dH += _volume_energy(src_cid, -1)
        dH += _volume_energy(nb_cid,  +1)

        if nb_cid != MEDIUM and src_cid == MEDIUM:
            dH += _chemotaxis_energy(r, c, N_field)

        if dH <= 0 or np.random.random() < np.exp(-dH / T):
            cell_id[r, c] = nb_cid
            cell_volume[src_cid] = cell_volume.get(src_cid, 0) - 1
            cell_volume[nb_cid]  = cell_volume.get(nb_cid,  0) + 1


# ── mutation ──────────────────────────────────────────────────────────────────

mu_base        = 0.003
mic_fold_mean  = 3.0
mic_fold_sigma = 0.5


def do_mutations():
    """
    attempt one mutation check per live cell, called once per MCS step.
    FIX: previously main.py called try_mutate() inside the steps_per_frame
    loop, multiplying the effective mutation rate by steps_per_frame.
    Centralising here decouples mutation rate from the animation frame rate.
    """
    for cid in list(cell_mic.keys()):
        if cid != MEDIUM:
            try_mutate(cid)


def try_mutate(cid):
    """
    attempt a mutation for cell cid.
    FIX: mutant is now spawned into an empty (medium) neighbour pixel at the
    frontier, rather than replacing a pixel inside the parent cell.  This
    gives the mutant an immediate foothold in open space so it is not
    instantly crushed by the parent's volume pressure.
    """
    global next_cell_id
    if np.random.random() > mu_base:
        return None

    positions = np.argwhere(cell_id == cid)
    if len(positions) == 0:
        return None

    # collect empty neighbour pixels just outside the cell boundary
    spawn_candidates = []
    for r, c in positions:
        for nr, nc in _neighbours4(r, c):
            if cell_id[nr, nc] == MEDIUM:
                spawn_candidates.append((nr, nc))

    if not spawn_candidates:
        return None

    idx     = np.random.randint(len(spawn_candidates))
    sr, sc  = spawn_candidates[idx]

    fold    = np.random.lognormal(mean=np.log(mic_fold_mean), sigma=mic_fold_sigma)
    new_mic = cell_mic[cid] * fold
    new_cid = next_cell_id
    next_cell_id += 1

    cell_id[sr, sc]       = new_cid
    cell_mic[new_cid]     = new_mic
    cell_lineage[new_cid] = new_cid   # each mutant is its own lineage
    cell_volume[new_cid]  = 1
    cell_volume[MEDIUM]  -= 1

    return new_cid


# ── kill cells exposed to lethal antibiotic ────────────────────────────────────

def kill_exposed_cells(A):
    dead = set()
    for cid in list(cell_mic.keys()):
        if cid == MEDIUM:
            continue
        mic       = cell_mic[cid]
        positions = np.argwhere(cell_id == cid)
        for r, c in positions:
            if A[r, c] > mic:
                dead.add(cid)
                break

    for cid in dead:
        positions = np.argwhere(cell_id == cid)
        for r, c in positions:
            cell_id[r, c] = MEDIUM
            cell_volume[MEDIUM] = cell_volume.get(MEDIUM, 0) + 1
        cell_volume.pop(cid, None)
        cell_mic.pop(cid, None)
        cell_lineage.pop(cid, None)


# ── helpers for PDE coupling and visualisation ────────────────────────────────

def get_bacteria_grid():
    """bool (Ny, Nx): True where any cell occupies the pixel."""
    return cell_id > MEDIUM


def get_rho_grid():
    """float density grid for PDE nutrient consumption term."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter((cell_id > MEDIUM).astype(np.float64), sigma=1.0)


def get_mic_grid():
    """float (Ny, Nx): MIC value at each occupied pixel, 0 elsewhere."""
    mic_2d = np.zeros((Ny, Nx), dtype=np.float32)
    for cid, mic in cell_mic.items():
        if cid == MEDIUM:
            continue
        mic_2d[cell_id == cid] = mic
    return mic_2d


def get_lineage_grid():
    """int (Ny, Nx): lineage id at each occupied pixel, -1 elsewhere."""
    lin_2d = np.full((Ny, Nx), -1, dtype=np.int32)
    for cid, lin in cell_lineage.items():
        if cid == MEDIUM or lin < 0:
            continue
        lin_2d[cell_id == cid] = lin
    return lin_2d


def n_cells():
    return max(0, len(cell_mic) - 1)


def n_mutations():
    return max(0, next_cell_id - 3)   # subtract medium + 2 founding cells