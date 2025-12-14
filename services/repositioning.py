''' 
# services/repositioning.py
#
from typing import Dict, List
from Entities.ev import EV, EvState
from Entities.GRID import Grid
from Entities.Incident import Incident
from utils.Helpers import utility_repositioning
class RepositioningService:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def accept_reposition_offers(self, evs, grids, incidents, mean_demand):
        for g_idx, g in grids.items():
            grid_demand=len(g.incidents)
            if grid_demand<=mean_demand:
                continue
            offers = []
            for nb in g.neighbours:
                if nb in grids:
                    for ev_id in grids[nb].evs:
                        v = evs[ev_id]
                        if v.state == EvState.IDLE:
                            u = utility_repositioning(v.aggIdleTime, v.aggIdleEnergy)
                            if u is None:
                                continue
                            offers.append((u, v))

            offers.sort(key=lambda x: x[0], reverse=True)

            cap = max(0, g.imbalance)
            for u, v in offers[:cap]:
                print(f"[Redeployment] ACCEPT EV {v.id} → Grid {g_idx} | u={u:.3f}")
                v.status     = "Repositioning"
                v.nextGrid   = g_idx
                print(f"[Check] EV {v.id} repositioning → Grid {g_idx} | demand={grid_demand} | mean={mean_demand}")
                v.sarns["reward"] = u
''' 
# services/repositioning.py
"""
This paper does urgency-based redeployment for ambulance repositioning.
What the below code should do ->  
    - Historical demand estimation per grid <why>
    - Grid urgency calculation
    - Redeployment of idle EVs to urgent grids using distance matching
"""
#all imports
from typing import Dict, List, Tuple
import math

from Entities.ev import EV, EvState
from utils.Helpers import utility_repositioning, point_to_grid_index #our utility fn (not using it tho), mapping (lat,lon) to grid index
from MAP_env import Grid # Need grid cuz AS are grids


class RepositioningService: #centralisaed repostioining 'controller' - paper says the same
    def __init__(self, env):

        self.env = env
        self.estimated_demand = {}      # for historical calls per grid - paper considers it, but in a different way < so chk this out
        #need to find future demand nd kambda vakues ssomehow 
        self.last_urgencies = {}        # for debugging part

    # HISTORICAL DEMAND CALCULATION (from dataset)
    def build_predicted_demand(self, df, env):
        #Build historical demand for each grid using the dataset 
        #Stores result in self.estimated_demand.
        self.estimated_demand = {g: 0 for g in env.grids.keys()}

        for _, row in df.iterrows(): #iterate thru full datatset 
            lat = row.get("Latitude")
            lng = row.get("Longitude")
            if lat is None or lng is None:
                continue

            gidx = point_to_grid_index(lat, lng, env.lat_edges, env.lng_edges) #map and get the grid indices 
            if gidx is not None and gidx >= 0:
                self.estimated_demand[gidx] += 1 #counter incrementation 
                #Bascally total historical calls in grid gidx

            #print("Predicted demand per grid after loading historical data:")
            #print(self.estimated_demand)

    # GRID URGENCY CALCULATION
    def calculate_grid_urgency(self, grid_id: int, grids, df=None) -> float:
        #Urgency index for a grid. calculated as UI = (predicted_demand * geo_factor) / (num_evs + 1)
        g = grids[grid_id]

        predicted = self.estimated_demand.get(grid_id, 0)
        geo_factor = 1.0                   # extend later if needed
        current_evs = len(g.evs) #evs in grid (=> AS)

        ui = (predicted * geo_factor) / (current_evs + 1) #Urgency index calc

        return float(ui)

    # DISTANCE FROM EV TO GRID
    def ev_distance_to_grid(self, ev: EV, target_grid: int, grids) -> float:
        #Euclidean distance from EV's current grid to target grid center
        try:
            gx, gy = grids[target_grid].center
        except:
            return 999999.0

        ex, ey = grids[ev.gridIndex].center
        return math.sqrt((gx - ex)**2 + (gy - ey)**2)

    # URGENCY-BASED REDEPLOYMENT
    def urgency_based_redeployment(
        self,
        evs: Dict[int, EV],
        grids: Dict[int, "Grid"],
        incidents: Dict,
        mean_demand: float
    ):
        """
        Stage-1 + Stage-2 redeployment based on the method from the paper:
            - Identify AS grids where demand > mean_demand
            - Compute urgency for AS grids
            - Sort by urgency
            - Assign nearest idle EVs to most urgent grids
        """
        # Identify AS grids as grids whose demand > mean_Demand
        as_grids = [g_idx for g_idx, g in grids.items()
                    if len(g.incidents) > mean_demand]

        if not as_grids:
            return

        # Compute urgency for each AS grid
        urgencies = {}
        for g_idx in as_grids:
            urg = self.calculate_grid_urgency(g_idx, grids)
            urgencies[g_idx] = urg

        # Save for debug
        self.last_urgencies = urgencies

        # Sort by urgency (descending since highest need first)
        urgent_sorted = sorted(as_grids, key=lambda x: urgencies[x], reverse=True)

        # Stage-1: Determine demand (how many EVs they need)
        grid_capacities = {}
        for g_idx in urgent_sorted:
            cap = max(0, grids[g_idx].imbalance) #imbalance used to calculate capacity but paper used Ambulance redeployment score  
            grid_capacities[g_idx] = cap

        # Collect IDLE EVs
        idle_evs = [ev for ev in evs.values()
                    if ev.state == EvState.IDLE and ev.status == "Idle"]

        if not idle_evs:
            return


        # Stage-2: For each AS grid, pick nearest idle EVs - this is a greedy approach tho, paper does something else 
        for g_id in urgent_sorted:
            cap = grid_capacities[g_id]
            if cap == 0:
                continue

            # Sort idle EVs by distance to this AS grid
            idle_evs.sort(key=lambda ev: self.ev_distance_to_grid(ev, g_id, grids))

            # Assign closest EVs
            assigned = idle_evs[:cap]
#NOT RL BASED REWARD SHOULDN'T BE THEREEEEEE
            for ev in assigned:
                u = utility_repositioning(ev.aggIdleTime, ev.aggIdleEnergy)

                # mark redeployment
                ev.status = "Repositioning"
                ev.nextGrid = g_id
                ev.sarns["reward"] = u
                ev.sarns["just_redeployed"] = False   # updated in Controller after move

            # Remove assigned EVs from pool
            idle_evs = idle_evs[cap:]
