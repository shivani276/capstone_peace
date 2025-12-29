# services/dispatcher.py
"""
Dispatcher service: handles gridwise dispatch of EVs to incidents.
Manages Algorithm 2: gridwise dispatch with multi-grid borrowing.
Dispatch prioritizes closest EV with patient urgency weighting.
"""
from typing import Dict, List, Tuple
#from MAP_env import MAP
from Entities.ev import EV, EvState
from Entities.GRID import Grid
from Entities.Incident import Incident, IncidentStatus
from utils.Helpers import (
    travel_minutes,
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
            
            # Get unassigned incidents in this grid
            K = g.get_pending_incidents(incidents)
            # Sort by priority first (ascending: 1 is highest), then by wait time (descending: longer waits first)
            K.sort(key=lambda iid: (incidents[iid].priority, -incidents[iid].waitTime))
            
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
                    I = borrowed
                
                inc = incidents[inc_id]

                if not I:
                    # No EVs available in 8-neighbourhood; skip this incident
                    continue
                    
                
                # Calculate patient priority weighting (reward for dispatch)
                # Priority weighting: 1 (highest) = 3.0, 2 = 2.0, 3 (lowest) = 1.0
                # Only priority matters, not wait time (to avoid incentivizing delays)
                priority_weight = {1: 3.0, 2: 2.0, 3: 1.0}
                dispatch_reward = priority_weight.get(inc.priority, 1.0)
                
                # Find closest EV (minimum ETA/distance)
                best_eid = None
                best_eta = float('inf')
                
                for eid in I:
                    ev = evs[eid]
                    # Calculate ETA from EV location to incident location
                    eta_minutes = travel_minutes(
                        ev.location[0], ev.location[1],
                        inc.location[0], inc.location[1],
                        kmph=40.0
                    )
                    
                    if eta_minutes < best_eta:
                        best_eta = eta_minutes
                        best_eid = eid
                
                # Assign incident to closest EV
                if best_eid is not None:
                    best_ev = evs[best_eid]
                    inc.assign_ev(best_eid)
                    inc.serviceTime = inc.waitTime
                    
                    # Record dispatch with priority-based reward (not wait time dependent)
                    best_ev.assign_incident(inc_id)
                    #best_ev.sarns["reward"] = dispatch_reward
                    best_ev.state = EvState.BUSY
                    #best_ev.status = "navigation"
                    
                    best_ev.assignedPatientPriority = inc.priority
                    if ev.gridIndex != inc.gridIndex:
                        best_ev.nextGrid = g_idx  # Move to incident grid first
                    
                    # Remove from available lists per Algorithm 2
                    I.remove(best_eid)
                    K.remove(inc_id)
                    assignments.append((best_eid, inc_id, float(dispatch_reward)))
                    

        
        return assignments
