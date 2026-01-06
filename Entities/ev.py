# entities/ev.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Any
import random

LatLng = Tuple[float, float]

class EvState(Enum):
    IDLE = auto()
    
    BUSY = auto()

@dataclass
class EV:
    id: int
    gridIndex: int
    location: LatLng
    state: EvState = EvState.IDLE
    nextGrid: Optional[int] = None
    status: str = "Idle"
    assignedPatientId: Optional[int] = None
    assignedPatientPriority: int = 0  # 1 to 3  inclusive for pri
    metric = []
    nav_metric = []
    aggIdleTime: float = 0.0
    aggIdleEnergy: float = 0.0
    aggBusyTime: float = 0.0
    navTargetHospitalId: int | None = None  # hospital currently chosen for navigation
    navdstGrid: int | None = None        # grid index of that hospital
    navEtaMinutes: float = 0.0              # latest ETA to that hospital
    navWaitTime: float = 0.0                 

    # sarns now a dict, as requested
    sarns: Dict[str, Any ] = field(default_factory=dict)

    def assign_incident(self, patient_id: int) -> None:
        self.assignedPatientId = patient_id
        #print("asgined patient",self.assignedPatientId,"to ev",self.id)
        #self.state = EvState.BUSY
        self.aggIdleTime = 0.0
        self.aggIdleEnergy = 0.0
        self.status = "Dispatching"
        
        

    def release_incident(self) -> None:
        self.assignedPatientId = None
        self.status = "Idle"
        self.state = EvState.IDLE
        self.nextGrid = None
        self.navTargetHospitalId = None
        self.navEtaMinutes = 0.0
        self.navdstGrid = None
        self.navWaitTime = 0.0
        self.aggIdleEnergy = 0.0
        self.aggIdleTime = 0.0

    def move_to(self, grid_index: int, new_loc: LatLng) -> None:
        self.gridIndex = grid_index
        self.location = new_loc
    
    def set_state(self, new_state: EvState) -> None:
        self.state = new_state

    def add_idle(self, dt: float) -> None:
        #print("added idle time for staying")
        self.aggIdleTime += dt
        self.aggIdleEnergy += 0.012
        
    def add_busy(self, dt: float) -> None:
        self.aggBusyTime += dt

    def reposition_cost(self, beta, eMax, wMax):
        eDen = float(eMax)
        wDen = float(wMax)

        eNum = float(getattr(self, "aggIdleEnergy", 0.0))
        wNum = float(getattr(self, "aggIdleTime", 0.0))

        eTerm = 0.0 if eDen <= 0.0 else (eNum / eDen)
        wTerm = 0.0 if wDen <= 0.0 else (wNum / wDen)

        b = float(beta)
        return (b * eTerm) + ((1.0 - b) * wTerm)

    # ========== Repositioning logic ==========
   
    def execute_reposition(self) -> None:
        """
        Execute the reposition decision made in this tick.
        This should be called after move_to() has been invoked by MAP.
        """
        print("entered execute rep added 8 and 0.12")
        #print(f"[DBG] execute_reposition called for EV {self.id}")
        self.aggIdleEnergy += 0.12  # Fixed energy cost for repositioning from one grid to another
        self.aggIdleTime += 8.0       # Fixed time cost for repositioning from one grid to another
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "gridIndex": self.gridIndex,
            "location": self.location,
            "nextGrid": self.nextGrid,
            "state": self.state.name,
            "status": self.status,
            "assignedPatientId": self.assignedPatientId,
            "assignedPatientPriority": self.assignedPatientPriority,
            "aggIdleTime": self.aggIdleTime,
            "aggIdleEnergy": self.aggIdleEnergy,
            "sarns": self.sarns,
        }
