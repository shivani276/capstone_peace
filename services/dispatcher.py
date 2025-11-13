# services/dispatcher.py
"""
Dispatcher service: handles gridwise dispatch of EVs to incidents.
Manages Algorithm 2: gridwise dispatch with multi-grid borrowing.
"""
from typing import Dict, List, Tuple
from Entities.ev import EV, EvState
from Entities.GRID import Grid
from Entities.Incident import Incident
from utils.Helpers import (
    utility_dispatch_v,
    utility_dispatch_p,
    utility_dispatch_total,
    W_MIN,
    W_MAX,
    P_MIN,
    P_MAX,
)


class DispatcherService:
    """Service for managing dispatch of EVs to incidents."""
    
    def dispatch_gridwise(
        self,
        grids: Dict[int, Grid],
        evs: Dict[int, EV],
        incidents: Dict[int, Incident],
        beta: float = 0.5,
    ) -> List[Tuple[int, int, float]]:

        assignments: List[Tuple[int, int, float]] = []
        
        for g_idx, g in grids.items():
            # Get eligible idle EVs in this grid (staying, not repositioning)
            I = g.get_eligible_idle_evs(evs)
            # Sort by idle time (longest idle first = highest priority)
            I.sort(key=lambda eid: evs[eid].aggIdleTime, reverse=True)
            
            # Get unassigned incidents in this grid
            K = g.get_pending_incidents(incidents)
            
            for inc_id in list(K):  # Copy to allow modification during iteration
                # If no local EVs, borrow from neighbours
                if not I:
                    borrowed: List[int] = []
                    for nb_idx in g.neighbours:
                        nb = grids[nb_idx]
                        borrowed.extend(nb.get_eligible_idle_evs(evs))
                    
                    # De-duplicate while preserving order
                    seen = set()
                    borrowed = [eid for eid in borrowed 
                               if not (eid in seen or seen.add(eid))]
                    # Sort by idle time
                    borrowed.sort(key=lambda eid: evs[eid].aggIdleTime, reverse=True)
                    I = borrowed
                
                if not I:
                    # No EVs available in 8-neighbourhood; skip this incident
                    continue
                
                inc = incidents[inc_id]
                
                # Patient utility based on wait time
                wait_minutes = inc.get_wait_minutes()
                U_P = utility_dispatch_p(wait_minutes, P_min=P_MIN, P_max=P_MAX)
                
                # Find EV that maximizes combined utility
                best_eid = None
                best_Ud = -1e9
                
                for eid in I:
                    ev = evs[eid]
                    U_V = utility_dispatch_v(ev.aggIdleTime, W_min=W_MIN, W_max=W_MAX)
                    U_D = utility_dispatch_total(
                        W_idle=ev.aggIdleTime,
                        W_kt=wait_minutes,
                        beta=beta,
                        W_min=W_MIN,
                        W_max=W_MAX,
                        P_min=P_MIN,
                        P_max=P_MAX,
                    )
                    
                    if U_D > best_Ud:
                        best_Ud = U_D
                        best_eid = eid
                
                # Assign incident to best EV
                if best_eid is not None:
                    best_ev = evs[best_eid]
                    inc.assignedEvId = best_eid
                    
                    # Record dispatch reward (utility)
                    prev_reward = best_ev.sarns.get("reward")
                    prev_reward = 0.0 if prev_reward is None else float(prev_reward)
                    best_ev.sarns["reward"] = prev_reward + float(best_Ud)
                    best_ev.sarns["utility_dispatch"] = float(best_Ud)
                    best_ev.sarns["assigned_incident"] = inc_id
                    
                    # Remove from available lists per Algorithm 2
                    I.remove(best_eid)
                    K.remove(inc_id)
                    assignments.append((best_eid, inc_id, float(best_Ud)))
        
        return assignments
