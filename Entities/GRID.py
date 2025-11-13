# entities/grid.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

LatLng = Tuple[float, float]


@dataclass
class Grid:
    index: int
    loc: Optional[LatLng] = None   
    center1d: float | None = None     # (lat_center, lng_center)

    incidents: List[int] = field(default_factory=list)  # incident ids
    evs: List[int] = field(default_factory=list)        # ev ids
    hospitals: List[int] = field(default_factory=list)  # hospital ids
    neighbours: List[int] = field(default_factory=list)

    imbalance: float = 0.0

    def set_loc(self, pos: LatLng) -> None:
        self.loc = pos

    def add_neighbour(self, idx: int) -> None:
        if idx not in self.neighbours:
            self.neighbours.append(idx)

    def add_incident(self, inc_id: int) -> None:
        if inc_id not in self.incidents:
            self.incidents.append(inc_id)

    def remove_incident(self, inc_id: int) -> None:
        if inc_id in self.incidents:
            self.incidents.remove(inc_id)

    def add_ev(self, ev_id: int) -> None:
        if ev_id not in self.evs:
            self.evs.append(ev_id)

    def remove_ev(self, ev_id: int) -> None:
        if ev_id in self.evs:
            self.evs.remove(ev_id)

    def add_hospital(self, hosp_id: int) -> None:
        if hosp_id not in self.hospitals:
            self.hospitals.append(hosp_id)

    def set_imbalance(self, val: float) -> None:
        self.imbalance = val

    # ========== Domain logic for querying grid state ==========
    
    def count_idle_available_evs(self, ev_dict: Dict[int, Any]) -> int:
        """Count idle and available EVs in this grid."""
        return sum(1 for ev_id in self.evs 
                  if ev_dict[ev_id].status == "available" and ev_dict[ev_id].state.name == "IDLE")
    
    def count_unassigned_incidents(self, incident_dict: Dict[int, Any]) -> int:
        """Count incidents not yet assigned to any EV."""
        count = 0
        for inc_id in self.incidents:
            inc = incident_dict[inc_id]
            if not getattr(inc, "assignedEvId", None):
                count += 1
        return count
    
    def calculate_imbalance(self, ev_dict: Dict[int, Any], incident_dict: Dict[int, Any]) -> int:
        """
        Calculate grid imbalance: B_{g,t} = max(0, unassigned incidents − idle EVs).
        Represents how many EVs are needed to service pending incidents.
        """
        unassigned = self.count_unassigned_incidents(incident_dict)
        idle_here = self.count_idle_available_evs(ev_dict)
        return max(0, unassigned - idle_here)
    
    def get_eligible_idle_evs(self, ev_dict: Dict[int, Any]) -> List[int]:
        """
        Get idle EVs in this grid that are staying (not accepted for reposition).
        These EVs are eligible for dispatch.
        """
        ids = []
        for ev_id in self.evs:
            ev = ev_dict[ev_id]
            if (ev.state.name == "IDLE" and 
                ev.status == "available" and 
                ev.nextGrid == ev.gridIndex):
                ids.append(ev_id)
        return ids
    
    def get_pending_incidents(self, incident_dict: Dict[int, Any]) -> List[int]:
        """Get incidents in this grid that are not yet assigned to any EV."""
        res = []
        for inc_id in self.incidents:
            inc = incident_dict[inc_id]
            if not getattr(inc, "assignedEvId", None):
                res.append(inc_id)
        return res

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "loc": self.loc,
            "incidents": list(self.incidents),
            "evs": list(self.evs),
            "hospitals": list(self.hospitals),
            "neighbours": list(self.neighbours),
            "imbalance": self.imbalance,
        }
