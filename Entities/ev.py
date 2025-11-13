# entities/ev.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Any

LatLng = Tuple[float, float]

class EvState(Enum):
    IDLE = auto()
    REPOSITION = auto()
    DISPATCH = auto()
    NAVIGATE = auto()
    SERVICE = auto()
    DROP = auto()
    BUSY = auto()

@dataclass
class EV:
    id: int
    gridIndex: int
    location: LatLng
    state: EvState = EvState.IDLE
    nextGrid: Optional[int] = None
    status: str = "available"
    assignedPatientId: Optional[int] = None

    aggIdleTime: float = 0.0
    aggIdleEnergy: float = 0.0
    navTargetHospitalId: int | None = None  # hospital currently chosen for navigation
    navEtaMinutes: float = 0.0              # latest ETA to that hospital
    navUtility: float = 0.0                 

    # sarns now a dict, as requested
    sarns: Dict[str, Any] = field(default_factory=dict)

    def assign_incident(self, patient_id: int) -> None:
        self.assignedPatientId = patient_id
        self.status = "dispatched"
        self.state = EvState.DISPATCH

    def release_incident(self) -> None:
        self.assignedPatientId = None
        self.status = "idle"
        self.state = EvState.IDLE
        self.nextGrid = None

    def move_to(self, grid_index: int, new_loc: LatLng) -> None:
        self.gridIndex = grid_index
        self.location = new_loc

    def set_next(self, grid_index: Optional[int]) -> None:
        self.nextGrid = grid_index
        if grid_index is not None and self.state == EvState.IDLE:
            self.state = EvState.REPOSITION

    def set_state(self, new_state: EvState) -> None:
        self.state = new_state

    def add_idle(self, dt: float, idle_energy: float = 0.0) -> None:
        self.aggIdleTime += dt
        self.aggIdleEnergy += idle_energy

    # ========== Repositioning logic ==========
    
    def accept_reposition_offer(self, dest_grid: int, utility: float) -> None:
        """
        Accept a reposition offer to move to dest_grid.
        Sets the next grid and records the utility as reward.
        """
        self.nextGrid = dest_grid
        prev_reward = self.sarns.get("reward")
        prev_reward = 0.0 if prev_reward is None else float(prev_reward)
        self.sarns["reward"] = prev_reward + utility
        
    def execute_reposition(self) -> None:
        """
        Execute the reposition decision made in this tick.
        This should be called after move_to() has been invoked by MAP.
        """
        self.status = "repositioning"
        self.aggIdleEnergy += 0.12  # Fixed energy cost for repositioning from one grid to another
        self.aggIdleTime += 8.0       # Fixed time cost for repositioning from one grid to another
    
    def is_eligible_for_dispatch(self) -> (bool): #List[Tuple[int, int, float]]:
        """
        Check if this EV is eligible for dispatch assignment.
        Only idle EVs that are staying in their current grid are eligible."""

        return (self.state == EvState.IDLE and 
                self.status == "available" and 
                self.nextGrid == self.gridIndex)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "gridIndex": self.gridIndex,
            "location": self.location,
            "nextGrid": self.nextGrid,
            "state": self.state.name,
            "status": self.status,
            "assignedPatientId": self.assignedPatientId,
            "aggIdleTime": self.aggIdleTime,
            "aggIdleEnergy": self.aggIdleEnergy,
            "sarns": self.sarns,
        }
