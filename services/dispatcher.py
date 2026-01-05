# services/dispatcher.py
"""
Dispatch policy:
- FCFS within each grid
- Nearest idle EV is assigned
- Borrowing from neighbouring grids allowed
"""
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from typing import Dict, List, Tuple

from Entities.ev import EV, EvState
from Entities.GRID import Grid
from Entities.Incident import Incident, IncidentStatus
from utils.Helpers import travel_minutes


class DispatcherService:

    def dispatch_gridwise(
        self,
        grids: Dict[int, Grid],
        evs: Dict[int, EV],
        incidents: Dict[int, Incident],
    ) -> List[Tuple[int, int]]:
        """
        Returns:
            List of (ev_id, incident_id) assignments
        """

        assignments: List[Tuple[int, int]] = []

        # Process EACH grid independently (gridwise FCFS)
        for g_idx, g in grids.items():
            print(f"\n[DISPATCH DEBUG] Grid {g_idx}")
            print(f"  Pending incidents: {g.get_pending_incidents(incidents)}")
            print(f"  Eligible idle EVs: {g.get_eligible_idle_evs(evs)}")

            # 1) Pending incidents in THIS grid
            pending_incidents = [
                inc for inc in incidents.values()
                if inc.status == IncidentStatus.UNASSIGNED
                and inc.gridIndex == g_idx
            ]

            # FCFS: earliest arrival first
            pending_incidents.sort(key=lambda inc: inc.timestamp)

            # 2) Handle incidents one by one
            for inc in pending_incidents:

                # 2a) Idle EVs in same grid
                candidate_evs = [
                    ev for ev in evs.values()
                    if ev.state == EvState.IDLE
                    and ev.status == "Idle"
                    and ev.gridIndex == g_idx
                ]

                # 2b) Borrow from neighbours if needed
                if not candidate_evs:
                    for nb_idx in g.neighbours:
                        nb = grids.get(nb_idx)
                        if nb is None:
                            continue
                        for ev in evs.values():
                            if (
                                ev.state == EvState.IDLE
                                and ev.status == "Idle"
                                and ev.gridIndex == nb_idx
                            ):
                                candidate_evs.append(ev)

                # 2c) Still no EV → incident waits
                if not candidate_evs:
                    continue  # FCFS preserved, no reassignment

                # 2d) Choose nearest EV (ETA-based)
                best_ev = min(
                    candidate_evs,
                    key=lambda ev: travel_minutes(
                    ev.location[0], ev.location[1],
                    inc.location[0], inc.location[1]
                    )

                )
                # 2e) Assign EV to incident
                inc.assign_ev(best_ev.id)
                best_ev.assign_incident(inc.id)
                print(f"[DISPATCH] EV {best_ev.id} → Incident {inc.id} " f"(Grid {g_idx}, wait={inc.get_wait_minutes():.2f} min)")

                assignments.append((best_ev.id, inc.id))