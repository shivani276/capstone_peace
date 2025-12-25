# MAP_env.py (Refactored - OOP-first design)
"""
Environment orchestrator for emergency response simulation.

MAP (Map) class serves as a thin orchestrator that:
1. Manages grid topology and geometric conversions
2. Manages collections (EVs, incidents, hospitals, grids)
3. Delegates domain logic to entity methods and services

Domain logic is distributed to:
- Entity classes (EV, Incident, Grid, Hospital) for entity-specific behavior
- Service classes (DispatcherService, RepositioningService, NavigationService) for cross-entity logic
"""
import random
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import math
import numpy as np
from utils.Helpers import (
    point_to_grid_index,
    load_grid_config_2d, P_MAX,H_MIN, H_MAX
)

from Entities.GRID import Grid
from Entities.ev import EV, EvState
from Entities.Incident import Incident, Priority, IncidentStatus
from Entities.Hospitals import Hospital

# Import services
from services.dispatcher import DispatcherService
from services.repositioning import RepositioningService
from services.navigation import NavigationService


class MAP:
    """
    Environment orchestrator for emergency response simulation.
    
    Responsibilities:
    - Initialize and manage grids, EVs, incidents, hospitals
    - Provide geometric/topology operations (grid conversions, neighbors)
    - Coordinate algorithms via service layer
    - Manage simulation state
    """
    
    def __init__(self, grid_config_path: str):
        """Initialize the MAP environment with grid configuration."""
        self.grids: Dict[int, Grid] = {}
        self.evs: Dict[int, EV] = {}
        self.incidents: Dict[int, Incident] = {}
        self.hospitals: Dict[int, Hospital] = {}

        self._incidentCounter = 0
        self._evCounter = 0
        self._hospitalCounter = 0
        self.dispatcher: DispatcherService

        # Load grid configuration
        self.lat_edges, self.lng_edges, _ = load_grid_config_2d(grid_config_path)
        self.nRows = len(self.lat_edges) - 1
        self.nCols = len(self.lng_edges) - 1

        # Initialize services
        self.dispatcher = DispatcherService()
        self.repositioner = RepositioningService()
        self.navigator = NavigationService()

        # Build grid topology
        self.build_grids(self.lat_edges, self.lng_edges)

    # ========== GRID GEOMETRY & TOPOLOGY ==========
    
    def build_grids(self, lat_edges, lng_edges) -> None:
        """Build grid cells and establish 8-connected neighbor relationships."""
        n_rows = len(lat_edges) - 1
        n_cols = len(lng_edges) - 1

        # Create all grid cells
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                self.grids[idx] = Grid(index=idx)

        # Connect 8-neighbors
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                nbs = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n_rows and 0 <= nc < n_cols:
                            nbs.append(nr * n_cols + nc)
                self.grids[idx].neighbours = nbs

    def index_to_rc(self, idx: int) -> Tuple[int, int]:
        """Convert 1D grid index to (row, col) coordinates."""
        return idx // self.nCols, idx % self.nCols

    #def rc_to_index(self, r: int, c: int) -> int:
        #"""Convert (row, col) coordinates to 1D grid index."""
        #return r * self.nCols + c

    def grid_center(self, idx: int) -> Tuple[float, float]:
        """Get the (lat, lng) center of a grid cell."""
        r, c = self.index_to_rc(idx)
        lat = (self.lat_edges[r] + self.lat_edges[r + 1]) / 2.0
        lng = (self.lng_edges[c] + self.lng_edges[c + 1]) / 2.0
        return lat, lng

    # ========== EV MANAGEMENT ==========
    
    def init_evs(self, seed: int = 42) -> None:
        """Initialize EVs with reproducible seeded placement."""
        self.evs.clear()
        self._evCounter = 0
        for g in self.grids.values():
            g.evs.clear()

        rng = random.Random(seed)
        n_evs = 27
        all_idx = list(self.grids.keys())

        for _ in range(n_evs):
            gi = rng.choice(all_idx)
            self.create_ev(gi)

    def create_ev(self, grid_index: int) -> EV:
        """Create a new EV and place it in a grid."""
        self._evCounter += 1
        loc = self.grid_center(grid_index)
        ev = EV(id=self._evCounter, gridIndex=grid_index, location=loc)
        self.evs[ev.id] = ev
        self.grids[grid_index].add_ev(ev.id)
        return ev

    def move_ev_to_grid(self, ev_id: int, new_grid_index: int) -> None:
        """Move an EV from its current grid to a new grid."""
        ev = self.evs[ev_id]
        old_idx = ev.gridIndex
        if old_idx in self.grids:
            self.grids[old_idx].remove_ev(ev_id)
        self.grids[new_grid_index].add_ev(ev_id)
        ev.move_to(new_grid_index, self.grid_center(new_grid_index))

    # ========== INCIDENT MANAGEMENT ==========
    
    def create_incident(self, incident_id: int, grid_index: int, location: Tuple[float, float], timestamp: Optional[datetime] = None, priority: Optional[int] = None) -> Incident:
        """Create a new incident and place it in a grid."""
        self._incidentCounter += 1

        if timestamp is None:
            timestamp = datetime.now()
        
        # Use priority from dataset, default to 1 if None
        pri = int(priority) if priority is not None else 1
        
        inc = Incident(
            id=incident_id,
            gridIndex=grid_index,
            timestamp=timestamp,
            location=location,
            priority=pri
        )
        self.incidents[inc.id] = inc
        self.grids[grid_index].add_incident(inc.id)
        #print("incident id",inc.id)
        return inc

    # ========== HOSPITAL MANAGEMENT ==========
    
    def init_hospitals(
        self,
        csv_path: str,
        *,
        lat_col: str = "Latitude",
        lng_col: str = "Longitude",
        name_col: str = "Name",
    ) -> None:
        """Load hospitals from CSV and place them in the grid."""
        if getattr(self, "_hospitals_initialized", False):
            return

        import pandas as pd

        df = pd.read_csv(csv_path)
        if lat_col not in df.columns or lng_col not in df.columns:
            raise ValueError(f"Missing hospital columns: {lat_col}/{lng_col}")

        # Clear existing hospital links
        for g in self.grids.values():
            g.hospitals.clear()

        # Create and place hospitals
        for _, row in df.iterrows():
            lat = float(row[lat_col])
            lng = float(row[lng_col])
            name = str(row[name_col]) if name_col in df.columns else f"Hospital_{self._hospitalCounter+1}"

            gi = point_to_grid_index(lat, lng, self.lat_edges, self.lng_edges)

            self._hospitalCounter += 1
            hid = self._hospitalCounter
            hc = Hospital(id=hid, loc=(lat, lng), gridIndex=gi, waitTime=0.0, services=[])
            self.hospitals[hid] = hc
            self.grids[gi].hospitals.append(hid)

        self._hospitals_initialized = True
        print(f"[MAP] Hospitals placed: {len(self.hospitals)} fixed locations.")

        print("Hospitals per grid (non-empty):")
        for gi, g in self.grids.items():
            if g.hospitals:
                ids = [self.hospitals[h].id for h in g.hospitals]
                print(f"  Grid {gi}: {ids}")


    def tick_hospital_waits(self, low_min: float = 5.0, high_min: float = 45.0, seed: int | None = None) -> None:
        """Reset hospital wait times to random values in range."""
        rng = random.Random(seed)
        if not getattr(self, "hospitals", None):
            print("[MAP] No hospitals to reset waits for.")
            return
        number = 0
        for hc in self.hospitals.values():
           # hc.waitTime = math.exp(13)
            number += 1
            rng = np.random.default_rng()
            lam = H_MIN + H_MAX / 2.0 # mean
            hc.waitTime = min(40, rng.exponential(13)) #poisson dist with mean
            #print("hc waitime set",hc.waitTime)
            #print(f"[MAP] Hospital waits initialised in [{hc.id}, {hc.waitTime}] minutes.")
        #print("number of hcs",number)
    '''def tick_hospital_waits(self, lam: float = 0.04, wmin: float = 5.0, wmax: float = 90.0, seed: int | None = None) -> None:
        """Update hospital wait times with random exponential drift."""
        if not getattr(self, "hospitals", None):
            return
        rng = random.Random(seed)
        for hc in self.hospitals.values():
            eps = rng.uniform(-lam, lam)
            hc.waitTime = max(wmin, min(wmax, hc.waitTime * math.exp(eps)))'''



    def next_grid_towards(self, from_idx: int, to_idx: int) -> int:

        if from_idx == to_idx:
            return from_idx  # already there

        n_rows = len(self.lat_edges) - 1
        n_cols = len(self.lng_edges) - 1

        # current cell
        row_from = from_idx // n_cols
        col_from = from_idx % n_cols

        # target cell
        row_to = to_idx // n_cols
        col_to = to_idx % n_cols

        # step direction in row/col: -1, 0, or 1
        dr = 0
        if row_to > row_from:
            dr = 1
        elif row_to < row_from:
            dr = -1

        dc = 0
        if col_to > col_from:
            dc = 1
        elif col_to < col_from:
            dc = -1

        # take one step
        new_row = row_from + dr
        new_col = col_from + dc

        # safety clamp (should already be in bounds)
        new_row = max(0, min(n_rows - 1, new_row))
        new_col = max(0, min(n_cols - 1, new_col))

        return new_row * n_cols + new_col


    # ========== ALGORITHMS (delegated to services) ==========
    
    def accept_reposition_offers(self) -> None:
        #print("function called for acepting offers")
        """
        Algorithm 1: Accept or reject repositioning offers from idle EVs.
        
        Delegates to RepositioningService.
        See services.repositioning.RepositioningService.accept_reposition_offers()
        """
        #print("function call into function :(")
        self.repositioner.accept_reposition_offers(self.evs, self.grids, self.incidents)

    '''def step_reposition(self) -> None:
        """
        Apply accepted reposition moves and clear pending decisions.
        
        Delegates to RepositioningService and handles physical grid moves.
        """
        #self.repositioner.execute_repositions(self.evs, self.grids)
        
        # Apply physical grid moves (MAP manages topology)
        for ev in self.evs.values():
            if ev.state != EvState.IDLE:
                continue
            dst = ev.nextGrid
            if dst is None:
                continue
            if dst != ev.gridIndex:
                self.move_ev_to_grid(ev.id, dst)
            ev.nextGrid = None'''

    def dispatch_gridwise(self, beta: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Algorithm 2: Gridwise dispatch of EVs to incidents.
        
        For each grid:
        1. Collect idle EVs that stayed in this grid
        2. For each unassigned incident:
           - If no local EVs, borrow from neighbors
           - Select EV with highest dispatch utility
           - Assign and remove from available lists
        
        Delegates to DispatcherService.
        
        Args:
            beta: Weight for vehicle idle time utility (vs patient wait time utility)
            
        Returns:
            List of (ev_id, incident_id, utility) assignments
        """
        return self.dispatcher.dispatch_gridwise(
            self.grids,
            self.evs,
            self.incidents,
            beta=beta,
        )

    def calculate_eta_plus_wait(self, ev, hospital: Hospital) -> float:
        """
        Calculate ETA + total wait time for an EV going to a hospital.
        
        Total wait includes:
        1. Hospital's base wait time (updated each tick)
        2. Service time of higher priority EVs already being served
        3. Service time of same priority EVs already being served
        
        Delegates to NavigationService.
        
        Args:
            ev: The EV (for location and priority)
            hospital: The target hospital
            
        Returns:
            ETA + total wait time in minutes
        """
        return self.navigator.calculate_eta_plus_wait(ev, hospital)

    def select_hospital(self, ev: EV, hospitals_in_grid, calculate_wait_func) -> Hospital:
        """
        Select the best (nearest) hospital for a patient incident.
        
        Delegates to NavigationService.
        
        Args:
            ev_id: EV ID (for context; not used in selection)
            inc_id: Incident ID
            
        Returns:
            Tuple of (hospital_id, eta_minutes)
        """
        return self.navigator.select_hospital(ev, hospitals_in_grid, calculate_wait_func=self.calculate_eta_plus_wait)

    '''def get_nav_candidates(self, inc_id: int, max_k: int = 8) -> Tuple[List[int], List[float], List[float]]:
        """
        Get top K candidate hospitals for an incident (sorted by proximity).
        
        Useful for decision-making systems (e.g., RL agents) that need multiple options.
        
        Delegates to NavigationService.
        
        Args:
            inc_id: Incident ID
            max_k: Maximum number of hospitals to return
            
        Returns:
            Tuple of (hospital_ids, etas_minutes, wait_times)
        """
        inc = self.incidents[inc_id]
        return self.navigator.get_candidate_hospitals(inc, self.hospitals, max_k=max_k)'''

    def update_after_tick(self, dt_minutes: float = 8.0) -> None:
        # EV updates
        #print("updtae function call")
        for ev in self.evs.values():
            if ev.nextGrid and ev.id is not None:
                if ev.state == EvState.BUSY: #and ev.gridIndex == ev.navdstGrid: #and ev.assignedPatientId is not None:
                    #
                    if ev.status == "Dispatching" and ev.assignedPatientId is not None:
                        inc = self.incidents.get(ev.assignedPatientId)
                        #print("attricbutes of inc ev id",inc.id,"assgined patient id",ev.assignedPatientId)
                        
                        #print("enviromnet lsit incs disp",len(self.incidents))
                        
                        #print("ev assgiend patient id",ev.assignedPatientId)
                        #print("incident in dispatch",inc)
                        if ev.gridIndex != ev.nextGrid:
                            #print("ev",ev.id,"in grid",ev.gridIndex,"with destination grid",ev.navdstGrid,"is moving to grid",ev.nextGrid,"\n")
                            self.move_ev_to_grid(ev.id, ev.nextGrid)
                            #print("ev moved to grid check?",ev.id,"in grid",ev.gridIndex,"moving to grid",ev.nextGrid,"destination grid",ev.navdstGrid,"\n")
                            if inc is not None:
                                inc.waitTime += 8.0
                        else:
                            if inc is not None:
                                #print("inc not none")
                                #print("ev got service",ev.id,"nav waittime",ev.navWaitTime,"busy time",ev.aggBusyTime,"state of ev",ev.state,"assigned patient",ev.assignedPatientId)
                                inc.waitTime += inc.estimate_eta_minutes(inc.location[0], inc.location[1], 40.0)
                                ev.status = "Navigation"
                                #print("ev",ev.id,"status",ev.status)
                                ev.nextGrid = None
                                inc.status = IncidentStatus.SERVICING

                    if ev.status == "Navigation" and ev.nextGrid is not None:
                        ev.navEtaMinutes -= 8
                        ev.aggBusyTime += 8
                        

                        #print("ev",ev.id,"is navigating with remaining time",ev.navEtaMinutes,"total busy time",ev.aggBusyTime,"\n")
                        if ev.nextGrid != ev.gridIndex and ev.nextGrid is not None:
                            #print("nav eta",ev.navEtaMinutes)
                            self.move_ev_to_grid(ev.id, ev.nextGrid)                            
                        
                             
                            #print("ev ",ev.id,"reached grid",ev.gridIndex,"total navigating time",ev.aggBusyTime)
                              
                        #elif max(0.0,ev.navEtaMinutes) == 0.0 and ev.navTargetHospitalId is not None:
                        elif ev.nextGrid == ev.gridIndex:
                            ev.status = "reached" #reached dst grid, now has to goto hc
                            
                            #print("ev ",ev.id,"nav time",ev.navEtaMinutes,"targethc",ev.navTargetHospitalId,"patient",ev.assignedPatientId)
                            if ev.navTargetHospitalId is not None:
                                h = self.hospitals[ev.navTargetHospitalId]  # Get the Hospital object

                                if ev.assignedPatientPriority == 1:
                                    h.evs_serving_priority_1.append(ev.id)
                                elif ev.assignedPatientPriority == 2:
                                    h.evs_serving_priority_2.append(ev.id)
                                else:
                                    h.evs_serving_priority_3.append(ev.id)
                                #print("ev ",ev.id,"reached dst hc and got in queue to",h.id,"remaining wait time of ev",ev.navWaitTime)
                                #print("ev",ev.id,"navtime before",ev.navWaitTime)
                                #ev.navWaitTime -= 8.0 
                                print("Map:ev",ev.id,"navtime after update ",ev.navWaitTime)
                            #ev.aggBusyTime = 0.0    #is this service time?
                            if ev.navWaitTime <= 0.0:
                                #print("wait time is negative")
                                if ev.assignedPatientId is not None:
                                    #print("enviromnet lsit incs nav",len(self.incidents))
                                    
                                    inc_n = self.incidents.get(int(ev.assignedPatientId))
                                    ev.release_incident()
                                    print("ev finished drop off with nav time", ev.navEtaMinutes,ev.status,ev.state)
                                    #since navwaittime is less than zero, it means serviced
                                    ev.aggBusyTime = 0.0
                               
                                    #print("should be dict",self.incidents,"\n")
                                    #print("keys",self.incidents.keys(),"\n","assgined id",ev.assignedPatientId)
                                    #if ev.assignedPatientId not in  self.incidents.keys():
                                        #print("keys",self.incidents.keys())
                                    #print("incidnents",inc_n)
                                    #print("assigned patient id",ev.assignedPatientId)
                                    #print("assgined incidnet id",inc.id)
                                    #print("incident")
                                    
                                    if inc_n is not None:
                                        inc_n.mark_resolved()
                                        
                                        
                                        
                                        g = self.grids.get(inc_n.gridIndex)
                                        if g is not None:
                                            g.remove_incident(inc_n.id)
                                            #del inc_n
                    '''inc = self.incidents.get(ev.assignedPatientId)
                       if inc is not None:
                        dest_grid_idx = ev.navdstGrid
                        hospitals_in_grid = [
                            h for h in self.hospitals.values()
                            if h.gridIndex == dest_grid_idx
                        ]
                        
                        # Use navigation service to select best hospital
                        best_hospital = self.navigator.select_hospital(
                            ev=ev,
                            hospitals_in_grid=hospitals_in_grid,
                            calculate_wait_func=self.calculate_eta_plus_wait
                        )
                        
                        if best_hospital is not None:
                            # Clear ETA (no longer traveling, now servicing)
                            ev.navEtaMinutes = 0.0
                        
                        # Mark incident as resolved and clean up
                        
                       
                        ev.release_incident()'''

                

            # 1) EV staying idle in its chosen grid
            if ev.state == EvState.IDLE:
                if ev.gridIndex == ev.sarns.get("action"):
                    #ev.add_idle(8)
                    #print("beforee",ev.aggIdleTime,"evid",ev.id)
                    ev.aggIdleTime += 8

                    #print("after",ev.aggIdleTime,"evid",ev.id)
                    #print("sucessful test for idle and stayed,dint add energy")
                elif ev.status == "Repositioning" or ev.sarns.get("reward") is not None:
                    #ev.execute_reposition()

                    self.move_ev_to_grid(ev.id, ev.nextGrid)  
                    ev.aggIdleEnergy += 0.12  # Fixed energy cost for repositioning from one grid to another
                    ev.aggIdleTime += 8.0  
                    #print("i guess its done?")
                    #print("sucessful test for idle and repositioning")
                elif ev.status == "Dispatching" and ev.assignedPatientId is not None:
                    ev.state = EvState.BUSY
                    #print("sucessful test for idle and dispatching, changed state")
            

                
                #print("busy time after",ev.aggBusyTime,"evid",ev.id)
                #print("sucessful test for navigating")

                '''
                hc_id = ev.navTargetHospitalId
                if hc_id is not None:
                    hospital = self.hospitals.get(hc_id)
                    if hospital is not None and getattr(hospital, "gridIndex", None) is not None:
                        ev.nextGrid = self.next_grid_towards(ev.gridIndex, hospital.gridIndex)
                        '''


        # Incident updates
        to_delete = []
        for inc_id, inc in list(self.incidents.items()):
            if inc.status != IncidentStatus.UNASSIGNED:
                continue

            if inc.waitTime < P_MAX:
                inc.add_wait(dt_minutes)
            else:
                inc.status = IncidentStatus.CANCELLED
                g = self.grids.get(inc.gridIndex)
                if g is not None and inc_id in g.incidents:
                    g.incidents.remove(inc_id)
                to_delete.append(inc_id)

        for inc_id in to_delete:
            del self.incidents[inc_id]


        # Recompute grid imbalances
        for g in self.grids.values():
            g.imbalance = g.calculate_imbalance(self.evs, self.incidents)
        
    '''def update_Navigation(self, dt_minutes: float = 8.0) -> None:
        for ev in self.evs.values():
            if ev.state == EvState.BUSY:
                #ev.add_busy(8)
                hc_id = ev.navTargetHospitalId
                if hc_id is not None:
                    hospital = self.hospitals.get(hc_id)
                    if hospital is not None and getattr(hospital, "gridIndex", None) is not None:
                        ev.nextGrid = self.next_grid_towards(ev.gridIndex, hospital.gridIndex)
                        
        '''
    #def update_after_timeslot(self, dt_minutes: float = 8.0) -> None:



                    
                


