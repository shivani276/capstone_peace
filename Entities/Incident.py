# entities/incident.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
from datetime import datetime
import math
import numpy as np

class IncidentStatus(Enum):
    UNASSIGNED = auto()
    ASSIGNED = auto()
    SERVICING = auto()

    RESOLVED = auto()
    CANCELLED = auto()

    
LatLng = Tuple[float, float]


@dataclass
class Incident:
    id: int
    gridIndex: int
    timestamp: datetime
    location: LatLng
    dropLocation: Optional[LatLng] = None
    priority: int = 1
    status: IncidentStatus = IncidentStatus.UNASSIGNED
    waitTime: float = 0.0
    responseTimestamp: Optional[datetime] = None
    hospitalTimestamp: Optional[datetime] = None
    responseToHospitalMinutes: Optional[float] = None
    serviceTime: float = 0.0
    remainingWaitTime: Optional[float] = None
    assignedEvId: Optional[int] = None
    

    def assign_ev(self, ev_id: int) -> None:
        self.assignedEvId = ev_id
        self.status = IncidentStatus.ASSIGNED
        #print("assginment status changed to",self.status)
        

    '''def start_service(self) -> None:
        self.status = IncidentStatus.SERVICING

    def start_drop(self) -> None:
        self.status = IncidentStatus.SERVICING'''

    def mark_resolved(self) -> None:
        self.status = IncidentStatus.RESOLVED
        self.remainingWaitTime = 0.0

    def cancel_incident(self) -> None:
        self.status = IncidentStatus.CANCELLED

    def add_wait(self, dt: float) -> None:
        self.waitTime += dt
        if self.remainingWaitTime is not None:
            self.remainingWaitTime = max(0.0, self.remainingWaitTime - dt)
    
    # ========== Domain logic for incident state ==========
    
    def is_unassigned(self) -> bool:
        """Check if this incident is waiting for assignment."""
        return self.assignedEvId is None or self.status == IncidentStatus.UNASSIGNED
    
    def get_wait_minutes(self) -> float:
        """Get accumulated wait time in minutes."""
        return self.waitTime
    
    def get_urgency_score(self) -> float:
        """
        Return urgency based on priority and wait time.
        Higher priority and longer waits = higher urgency.
        """
        priority_weight = {
            1: 1.0,  # LOW
            2: 2.0,  # MED
            3: 3.0,  # HIGH
        }
        return priority_weight.get(self.priority, 1.0) * (1.0 + self.waitTime / 30.0)
    
    def haversine_distance_km(self, lat2: float, lng2: float) -> float:
        """
        Calculate great-circle distance in km from this hospital to a point (lat2, lng2).
        Uses Haversine formula.
        """
        R = 6371.0  # Earth radius in km
        lat1, lng1 = self.location
        
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlng/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def estimate_eta_minutes(self, lat2: float, lng2: float, kmph : float) -> float:
        """
        Estimate ETA (in minutes) from this hospital to a point (lat2, lng2)
        at a constant average speed.
        """
        kmph = 40 #np.clip(np.random.normal(40.0, 5.0), 20.0, 80.0)
        #print("speed ",kmph)
        km = self.haversine_distance_km(lat2, lng2)
        return 60.0 * km / max(kmph, 1e-6)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "gridIndex": self.gridIndex,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "dropLocation": self.dropLocation,
            "priority": self.priority,
            "status": self.status.name,
            "waitTime": self.waitTime,
            "serviceTime": self.serviceTime,
            "remainingWaitTime": self.remainingWaitTime,
            "assignedEvId": self.assignedEvId,
        }
