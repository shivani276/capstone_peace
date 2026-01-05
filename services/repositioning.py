# services/repositioning.py

from typing import Dict
import math
from scipy.stats import gamma
from utils.Helpers import travel_minutes, utility_repositioning, hop_distance
from Entities.ev import EV, EvState
from MAP_env import Grid
from Entities.GRID import Grid
from utils.Helpers import E_MAX, W_MAX
from Entities.Incident import Incident

class RepositioningService:
    """
    Centralized urgency-based redeployment (paper-faithful).

    Logic:
    - Compute urgency T*_j for every grid
    - Lower T*_j = more urgent
    - Iteratively assign nearest idle EVs to most urgent grids
    - One EV per grid per tick (stable & safe)
    """

    def __init__(self):
        self.last_urgencies = {}

    # ------------------ ETA utilities ------------------ #


    def ev_to_grid_eta(self, ev: EV, grid_id: int, grids):
        g = grids[grid_id]
        return travel_minutes(
            ev.location[0], ev.location[1],
            g.center1d[0], g.center1d[1]
        )

    # ------------------ Urgency index ------------------ #

    @staticmethod
    def compute_urgency_index(n_j, lambda_j, mu_j):
        if lambda_j <= 0:
            return math.inf
        return gamma.ppf(1 - mu_j, a=n_j + 1, scale=1.0 / lambda_j)

    # ------------------ Redeployment ------------------ #

    def urgency_based_redeployment(
        self,
        evs: Dict[int, EV],
        grids: Dict[int, Grid],
        lambda_dict: Dict[int, float],
        mu_dict: Dict[int, float],
    ):
        """
        Deterministic, bug-free urgency-based redeployment.
        """

        # ---------- 1. Compute urgencies ---------- #
        urgencies = {}

        for g_idx in grids.keys():
            n_j = sum(
                1 for ev in evs.values()
                if ev.state == EvState.IDLE and ev.gridIndex == g_idx
            )

            lambda_j = lambda_dict.get(g_idx, 0.0)
            mu_j = mu_dict.get(g_idx, 0.5)

            urgencies[g_idx] = self.compute_urgency_index(
                n_j, lambda_j, mu_j
            )

        self.last_urgencies = urgencies

        # ---------- 2. Sort grids by urgency ---------- #
        grids_sorted = sorted(urgencies, key=lambda g: urgencies[g])

        # ---------- 3. Collect idle EVs ---------- #
        idle_evs = [
            ev for ev in evs.values()
            if ev.state == EvState.IDLE and ev.status == "Idle"
        ]

        if not idle_evs:
            return

        # ---------- 4. Assign EVs ---------- #
        for g_idx in grids_sorted:

            if not idle_evs:
                break

            # Recompute n_j (CRITICAL — NO STALE STATE)
            n_j = sum(
                1 for ev in evs.values()
                if ev.state == EvState.IDLE and ev.gridIndex == g_idx
            )

            # One EV per grid per tick (stable behavior)
            if n_j >= 1:
                continue

            # Find closest EV
            idle_evs.sort(
                key=lambda ev: self.ev_to_grid_eta(ev, g_idx, grids)
            )

            ev = idle_evs.pop(0)

            ev.status = "Repositioning"
            ev.nextGrid = g_idx
            ev.sarns["just_redeployed"] = True
            ev.sarns["from_grid"] = ev.gridIndex
            ev.sarns["to_grid"] = g_idx
            #ev.sarns["reward"] = utility_repositioning(ev.aggIdleTime, ev.aggIdleEnergy)
            ev.sarns["reward"] = 0.0
