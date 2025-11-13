# services/repositioning.py
"""
Repositioning service: handles EV repositioning offers and acceptance.
Manages Algorithm 1: accept reposition offers.
"""
from typing import Dict, List, Tuple
from collections import defaultdict
from Entities.ev import EV
from Entities.GRID import Grid


class RepositioningService:
    """Service for managing EV repositioning decisions."""
    
    def accept_reposition_offers(
        self,
        evs: Dict[int, EV],
        grids: Dict[int, Grid],
        incidents: Dict,
    ) -> None:
        """
        Algorithm 1: Accept reposition offers from idle EVs.
        
        Each idle EV provides an offer (destination grid, utility).
        For each grid, accept offers with highest utility up to grid imbalance capacity.
        
        Args:
            evs: Dict mapping EV IDs to EV objects
            grids: Dict mapping grid indices to Grid objects
            incidents: Dict mapping incident IDs to Incident objects
        """
        # Group offers by destination grid
        offers_by_g = defaultdict(list)  # g_idx -> list[(utility, ev_id)]
        
        for ev in evs.values():
            from Entities.ev import EvState
            if ev.state != EvState.IDLE:
                continue
            
            dst = ev.sarns.get("action")
            u = ev.sarns.get("utility")
            if dst is None or u is None:
                continue
            
            offers_by_g[dst].append((float(u), ev.id))
        
        # Process each destination grid
        for g_idx, offers in offers_by_g.items():
            # Sort by utility (highest first)
            offers.sort(key=lambda x: x[0], reverse=True)
            
            # Capacity = how many EVs needed to balance this grid
            grid = grids[g_idx]
            cap = grid.calculate_imbalance(evs, incidents)
            
            # Accept top offers up to capacity
            accepted = 0
            for u, ev_id in offers:
                ev = evs[ev_id]
                if accepted < cap:
                    # Accept the offer
                    ev.accept_reposition_offer(g_idx, float(u))
                    accepted += 1
                else:
                    # Reject the offer (stay in current grid)
                    ev.reject_reposition_offer()
    
    def execute_repositions(
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
            from Entities.ev import EvState
            
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
            ev.execute_reposition()
