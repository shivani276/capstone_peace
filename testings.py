
''' 
from scipy.stats import gamma
def compute_urgency_index(n_j, lambda_j, mu_j):
    """
    Computes T*_j (safety time upper bound / urgency index)

    Parameters:
    n_j       : number of available ambulances at station j
    lambda_j  : arrival rate of EMS requests (per unit time)
    mu_j      : station threshold (0 < mu_j < 1)

    Returns:
    T_star    : urgency index T*_j
    """
    shape = n_j + 1              # k
    scale = 1.0 / lambda_j       # theta = 1/lambda

    # CDF(T*) = 1 - mu_j
    T_star = gamma.ppf(1 - mu_j, a=shape, scale=scale)
    return T_star

print(compute_urgency_index(n_j=1, lambda_j=2, mu_j=0.8))  # ≈ 0.42
print(compute_urgency_index(n_j=1, lambda_j=2, mu_j=0.4))  # ≈ 1.02
'''
import numpy as np
import math
from scipy.stats import gamma

# -----------------------------
# 1. Normalized arrival rates
# -----------------------------

def compute_normalized_arrival_rates(grids, dt_minutes=8, eps=0.05):
    lambda_dict = {}

    for grid_id, grid in grids.items():
        lambda_dict[grid_id] = len(grid.incidents) / dt_minutes

    lambda_vals = np.array(list(lambda_dict.values()), dtype=np.float32)

    if lambda_vals.max() > lambda_vals.min():
        norm = (lambda_vals - lambda_vals.min()) / (lambda_vals.max() - lambda_vals.min())
    else:
        norm = np.zeros_like(lambda_vals)

    mu_vals = eps + (1 - 2 * eps) * norm
    mu_dict = dict(zip(lambda_dict.keys(), mu_vals))

    return mu_dict, lambda_dict


# -----------------------------
# 2. Urgency index computation
# -----------------------------

def compute_urgency_index(n_j, lambda_j, mu_j):
    if lambda_j <= 0:
        return math.inf

    shape = n_j + 1
    scale = 1.0 / lambda_j

    return gamma.ppf(1 - mu_j, a=shape, scale=scale)


# -----------------------------
# 3. Verification helper
# -----------------------------

def compute_all_urgencies(grids):
    mu_dict, lambda_dict = compute_normalized_arrival_rates(grids)

    urgency = {}

    for grid_id, grid in grids.items():
        n_j = len(grid.idle_evs)
        lambda_j = lambda_dict[grid_id]
        mu_j = mu_dict[grid_id]

        urgency[grid_id] = compute_urgency_index(n_j, lambda_j, mu_j)

        print(
            f"Grid {grid_id} | "
            f"n_j={n_j}, lambda_j={lambda_j:.3f}, mu_j={mu_j:.3f}, "
            f"T*_j={urgency[grid_id]:.3f}"
        )

    return urgency

class DummyGrid:
    def __init__(self, incidents, idle_evs):
        self.incidents = incidents
        self.idle_evs = idle_evs


grids = {
    0: DummyGrid(incidents=[1, 2, 3, 4], idle_evs=[1]),
    1: DummyGrid(incidents=[1], idle_evs=[1, 2]),
    2: DummyGrid(incidents=[1, 2], idle_evs=[]),
}

compute_all_urgencies(grids)
