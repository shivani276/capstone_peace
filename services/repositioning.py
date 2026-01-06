# services/repositioning.py
"""
Repositioning service: handles EV repositioning offers and acceptance.
Manages Algorithm 1: accept reposition offers.
"""
from typing import Dict, List, Tuple
from collections import defaultdict

from Entities.GRID import Grid
from utils.Helpers import E_MAX, W_MAX

from Entities.ev import EV, EvState
from Entities.GRID import Grid
from Entities.Incident import Incident
from utils.Helpers import utility_repositioning, hop_distance


class RepositioningService:
   
    
    def accept_reposition_offers(
        self,
        evs: Dict[int, EV],
        grids: Dict[int, Grid],
        n_cols: int,
        function) -> None:

        for ev_id, v in evs.items():
            if v.state != EvState.IDLE:
                continue

            dst = v.sarns.get("action", None)
            if dst is None:
                continue

            if int(dst) == int(v.gridIndex):
                v.sarns["reward"] = utility_repositioning(
                    0, 0, v.aggIdleEnergy, v.aggIdleTime, 0.5
                )

        applicable_evs: List[EV] = []
        for g_idx, g in grids.items():
            
            if g.imbalance < 0:
                for ev_id in g.evs:
                    ev_obj = evs[ev_id]
                    if ev_obj.state == EvState.IDLE:
                        applicable_evs.append(ev_obj)
            else:
                continue
        offers_g = []
        for g_idx, g in grids.items():       

            if g.imbalance > 0:     
            # 2) Build offers_g: offers from neighbour EVs that want THIS grid
                   # list of tuples (utility, ev_id, ev_obj)
                for v in applicable_evs:
                    dst = v.sarns.get("action")
                    #u = v.sarns.get("utility")
                    c = v.reposition_cost(0.5,E_MAX,W_MAX)  # cost to go to g_idx
                    if dst is None or c is None:
                            continue
                    if dst == g_idx and v.gridIndex != g_idx:
                        offers_g.append((float(c), v.id, v))
                        #print(offers_g)

                offers_g.sort(key=lambda x: x[0], reverse=False)
                #print(offers_g[0][2])
                # 4) Capacity: how many EVs this grid "needs"
                imbalance = g.imbalance
                cap = int(max(0, imbalance))
                accepted = 0
                y_i_g_t =0
                while cap > 0 and offers_g:
                    c_val, ev_id, v_obj = offers_g.pop(0)
                    #print(v_obj)
                    v_obj.status = "Repositioning"
                    y_i_g_t = 1
                    h_ggp = hop_distance(v_obj.gridIndex, g_idx, n_cols)
                    v_obj.sarns["reward"] = utility_repositioning(
                        y_i_g_t, h_ggp, v_obj.aggIdleEnergy, v_obj.aggIdleTime, 0.5
                    )
                    #print("Reward for EV ", v_obj.id, " to go to grid ", g_idx, " is ", v_obj.sarns["reward"])
                    v_obj.nextGrid = function(v_obj.gridIndex, g_idx)
                    accepted += 1
                    cap -= 1
            else:
                continue
        # 5) For unaccepted offers, set reward to stay       
        for c_val, ev_id, v in offers_g:
            h_ggp = hop_distance(v.gridIndex, g_idx, n_cols)
            v.sarns["reward"] = utility_repositioning(
                0, h_ggp, v.aggIdleEnergy, v.aggIdleTime, 0.5
            )
        

    '''def execute_repositions(
        self,
        evs: Dict[int, EV],
        grids: Dict[int, Grid],
    ) -> None:
        """
        Step repositioning: apply accepted moves and clear pending decisions.
        
        Args:
            evs: Dict mapping EV IDs to EV objects
            grids: Dict mapping grid indices to Grid objects
        """
        for ev in evs.values():            
            if ev.state != EvState.IDLE:
                ev.nextGrid = None
                continue
            
            dst = ev.nextGrid
            if dst is None:
                # Not decided; treat as stay
                ev.nextGrid = None
                continue
            
            if dst != ev.gridIndex:
                # Execute the move (caller should use move_ev_to_grid from MAP)
                # Here we just mark it; MAP will handle the actual grid list updates
                pass
            
            # Clear pending reposition
            ev.execute_reposition()'''
