import numpy as np
from pde import get_antibiotic_at, Nx, Ny, MIC_base

# ── CPM parameters ─────────────────────────────────────────────────────────────

J_medium = 8.0
J_diff   = 16.0
J_same   = 2.0

lambda_V = 0.5
V_target = 25     

chi      = 3.0
T        = 5.0

MEDIUM = 0

# ── state arrays ───────────────────────────────────────────────────────────────

cell_id             = np.zeros((Ny, Nx), dtype=np.int32)
cell_mic            = {0: 0.0}
cell_lineage        = {0: -1}
cell_parent_lineage = {0: -1}   
cell_volume         = {0: Nx * Ny}
next_cell_id        = 1


def add_cell(pixels, mic, lineage):
    global next_cell_id
    cid = next_cell_id
    next_cell_id += 1
    cell_mic[cid]            = mic
    cell_lineage[cid]        = lineage
    cell_parent_lineage[cid] = lineage
    cell_volume[cid]         = len(pixels)
    for (r, c) in pixels:
        cell_id[r, c] = cid
        cell_volume[0] -= 1
    return cid


def init_cpm_from_CA(bacteria, bacteria_indices, bacteria_mic_list, bacteria_lineage_list):
    global cell_id, next_cell_id
    cell_id[:] = MEDIUM
    cell_volume[0] = Nx * Ny
    for k in list(cell_mic.keys()):
        if k != MEDIUM:
            cell_mic.pop(k)
            cell_lineage.pop(k, None)
            cell_parent_lineage.pop(k, None)
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
    if cid == MEDIUM:
        return 0.0
    v  = cell_volume.get(cid, 0)
    vt = V_target
    return lambda_V * ((v + delta - vt) ** 2 - (v - vt) ** 2)


def _chemotaxis_energy(r, c, N_field):
    return -chi * N_field[r, c]


# ── MCS step ──────────────────────────────────────────────────────────────────

def mcs_step(N_field, A, n_attempts=None):
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
            
    # Trigger divisions at the end of the MCS step
    do_divisions()


# ── cell division ─────────────────────────────────────────────────────────────

def do_divisions():
    """
    Splits cells that have grown close to V_target.
    The division plane is chosen randomly to ensure isotropic colony expansion.
    """
    global next_cell_id
    
    # Use list() to avoid dictionary changed size during iteration errors
    for cid in list(cell_mic.keys()):
        if cid == MEDIUM:
            continue

        vol = cell_volume.get(cid, 0)
        # Trigger division if the cell hits 90% of its target volume
        if vol >= V_target * 0.9:
            positions = np.argwhere(cell_id == cid)
            if len(positions) < 2:
                continue

            # Find center of mass
            com = positions.mean(axis=0)

            # Generate a random 2D vector for the splitting plane
            theta = np.random.uniform(0, 2 * np.pi)
            v = np.array([np.cos(theta), np.sin(theta)])

            # Project relative pixel positions to partition them
            rel_pos = positions - com
            mask = (rel_pos @ v) > 0

            # Fallback in case of an extreme geometric split (e.g., all True/False)
            if not np.any(mask) or np.all(mask):
                mask = np.arange(len(positions)) < (len(positions) // 2)

            new_pixels = positions[mask]

            # Create new daughter cell
            new_cid = next_cell_id
            next_cell_id += 1

            for r, c in new_pixels:
                cell_id[r, c] = new_cid

            # Update volumes
            cell_volume[cid]     = len(positions) - len(new_pixels)
            cell_volume[new_cid] = len(new_pixels)

            # Daughter cell inherits parents traits completely
            cell_mic[new_cid]            = cell_mic[cid]
            cell_lineage[new_cid]        = cell_lineage[cid]
            cell_parent_lineage[new_cid] = cell_parent_lineage[cid]


# ── mutation ──────────────────────────────────────────────────────────────────

mu_base        = 0.0003   
mic_fold_mean  = 3.0
mic_fold_sigma = 0.5

MUT_VOL_THRESHOLD = V_target // 2   


def do_mutations():
    """
    One mutation attempt per live cell per MCS step.
    Cells below MUT_VOL_THRESHOLD are silently skipped.
    """
    for cid in list(cell_mic.keys()):
        if cid != MEDIUM:
            try_mutate(cid)


def try_mutate(cid):
    global next_cell_id

    if cell_volume.get(cid, 0) < MUT_VOL_THRESHOLD:
        return None

    if np.random.random() > mu_base:
        return None

    positions = np.argwhere(cell_id == cid)
    if len(positions) == 0:
        return None

    spawn_candidates = []
    for r, c in positions:
        for nr, nc in _neighbours4(r, c):
            if cell_id[nr, nc] == MEDIUM:
                spawn_candidates.append((nr, nc))

    if not spawn_candidates:
        return None

    idx    = np.random.randint(len(spawn_candidates))
    sr, sc = spawn_candidates[idx]

    fold    = np.random.lognormal(mean=np.log(mic_fold_mean), sigma=mic_fold_sigma)
    new_mic = cell_mic[cid] * fold
    new_cid = next_cell_id
    next_cell_id += 1

    cell_id[sr, sc]              = new_cid
    cell_mic[new_cid]            = new_mic
    cell_lineage[new_cid]        = new_cid             
    cell_parent_lineage[new_cid] = cell_lineage[cid]   
    cell_volume[new_cid]         = 1
    cell_volume[MEDIUM]         -= 1

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
        cell_parent_lineage.pop(cid, None)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_bacteria_grid():
    return cell_id > MEDIUM


def get_rho_grid():
    from scipy.ndimage import gaussian_filter
    return gaussian_filter((cell_id > MEDIUM).astype(np.float64), sigma=1.0)


def get_mic_grid():
    mic_2d = np.zeros((Ny, Nx), dtype=np.float32)
    for cid, mic in cell_mic.items():
        if cid == MEDIUM:
            continue
        mic_2d[cell_id == cid] = mic
    return mic_2d


def get_lineage_grid():
    lin_2d = np.full((Ny, Nx), -1, dtype=np.int32)
    for cid, lin in cell_lineage.items():
        if cid == MEDIUM or lin < 0:
            continue
        lin_2d[cell_id == cid] = lin
    return lin_2d


def get_parent_lineage_grid():
    lin_2d = np.full((Ny, Nx), -1, dtype=np.int32)
    for cid, lin in cell_parent_lineage.items():
        if cid == MEDIUM or lin < 0:
            continue
        lin_2d[cell_id == cid] = lin
    return lin_2d


def n_cells():
    return max(0, len(cell_mic) - 1)


def n_mutations():
    return max(0, next_cell_id - 3)