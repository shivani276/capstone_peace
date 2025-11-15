#==============Checking Entities=================
#--------------Checking EV-----------------------
'''
from Entities.ev import EV, EvState

ev = EV(id=1, gridIndex=5, location=(10.0, 20.0))
#print(ev)

#ev.assign_incident(99)
#print(ev.assignedPatientId, ev.status, ev.aggIdleTime, ev.aggIdleEnergy)
#ev.release_incident()
#print(ev)
ev.move_to(3, (1, 1))
#print(ev.gridIndex, ev.location)    

ev.add_idle(8)
#print(ev.aggIdleTime)

before_e = ev.aggIdleEnergy
before_t = ev.aggIdleTime
ev.execute_reposition()
print(ev.aggIdleEnergy - before_e, ev.aggIdleTime - before_t)

print(ev.to_dict())
'''

#---------------Checking Grid-----------------
'''
from Entities.GRID import Grid
from Entities.ev import EV, EvState
from Entities.Incident import Incident, IncidentStatus
from datetime import datetime

g = Grid(index=0)
evs = {
    1: EV(1, 0, (0,0)),
    2: EV(2, 0, (0,0))
}
incs = {
    10: Incident(10, 0, datetime.now(), (0,0)),
    11: Incident(11, 0, datetime.now(), (0,0))
}

g.evs = [1,2]
g.incidents = [10,11]

evs[1].state = EvState.IDLE
evs[1].status = "available"
evs[1].sarns["action"] = 0

evs[2].state = EvState.BUSY
evs[2].status = "Navigation"
#print(g.count_idle_available_evs(evs))
incs[10].assignedEvId = 1
#print(g.count_unassigned_incidents(incs))
#print(g.calculate_imbalance(evs, incs))
#print(g.get_eligible_idle_evs(evs))
#print(g.get_pending_incidents(incs))

#g = Grid(index=0)

# Add neighbour 1
#g.add_neighbour(1)
#print(g.neighbours)   # Expected: [1]

# Add same neighbour again
#g.add_neighbour(1)
#print(g.neighbours)   # Expected: still [1]

#g = Grid(index=0)

#g.add_incident(100)
#print(g.incidents)     # Expected: [100]

#g.add_incident(100)
#print(g.incidents)     # Expected: still [100]
print(g.to_dict)
'''
#-----------------Check Incident--------------
'''
from Entities.Incident import Incident, Priority
from datetime import datetime

i = Incident(1, 5, datetime.now(), (10,10))

#print(i)
i.assign_ev(3)
#print(i.status, i.assignedEvId)
i.add_wait(10)
#print(i.waitTime)

print(i.get_urgency_score())
i.waitTime = 100
print(i.get_urgency_score())
'''
#=======================SERVICES===========================
#-----------------------Repositioning----------------------
'''
from Entities.ev import EV, EvState
from Entities.GRID import Grid
from services.repositioning import RepositioningService

# Grids
g0 = Grid(index=0)
g1 = Grid(index=1)
g0.add_neighbour(1)
g1.add_neighbour(0)

grids = {0: g0, 1: g1}

# EV
ev = EV(id=1, gridIndex=0, location=(0.0, 0.0))
ev.state = EvState.IDLE
ev.status = "available"
ev.aggIdleTime = 30.0
ev.aggIdleEnergy = 5.0
ev.sarns["action"] = 1  # wants to go to grid 1

g0.add_ev(ev.id)
evs = {ev.id: ev}

incidents = {}  # <- no incidents, so imbalance = 0

rep = RepositioningService()

print("before:", ev.nextGrid, ev.sarns.get("reward"), ev.status)
rep.accept_reposition_offers(evs, grids, incidents)
print("after:", ev.nextGrid, ev.sarns.get("reward"), ev.status)

from datetime import datetime
from Entities.Incident import Incident, Priority

# 1) Create one incident in grid 1
inc = Incident(
    id=1,
    gridIndex=1,
    timestamp=datetime.now(),
    location=(0.0, 0.0),
    priority=Priority.MED,
)
incidents = {1: inc}
g1.add_incident(inc.id)

# 2) Now run accept_reposition_offers again
rep.accept_reposition_offers(evs, grids, incidents)

print("after:", ev.nextGrid, ev.sarns.get("reward"), ev.status)
'''
#-------------------Dispatching------------------------------
'''
from services.dispatcher import DispatcherService
from Entities.GRID import Grid
from Entities.ev import EV, EvState
from Entities.Incident import Incident
from datetime import datetime

g = Grid(0)
ev = EV(1, 0, (0,0))
ev.sarns["action"] = 0
ev.status = "available"
ev.state = EvState.IDLE

inc = Incident(10,0,datetime.now(),(1,1),waitTime=8)

g.evs=[1]
g.incidents=[10]

evs={1:ev}
incs={10:inc}
grids={0:g}

d = DispatcherService()
print(d.dispatch_gridwise(grids,evs,incs))
print(ev)
'''
#===================TESTING MAP_env=============================
from MAP_env import MAP
from Entities.ev import EvState
from Entities.Incident import IncidentStatus
from utils.Helpers import P_MAX

env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()
#env.create_incident(0, (10,10))
ev = env.create_ev(0)
'''
#print(env.grids[0].evs)
#print(env.grids[0].incidents)
print("nRows, nCols:", env.nRows, env.nCols)
print("num grids:", len(env.grids))

mid = list(env.grids.keys())[len(env.grids)//2]
g = env.grids[mid]
print("Grid", mid, "neighbours:", sorted(g.neighbours))


all_evs = env.evs
print("EV count:", len(all_evs))

# Check each EV exists in its grid’s ev list
for eid, ev in all_evs.items():
    assert eid in env.grids[ev.gridIndex].evs, f"EV {eid} not in its grid list!"
print("✅ EVs correctly placed in grids.")


ev = env.create_ev(0)
print("New EV:", ev.id, ev.gridIndex, ev.location)
print("Grid[0] evs:", env.grids[0].evs)

env.move_ev_to_grid(ev.id, 1)
print("After move:", ev.id, ev.gridIndex)
print("Grid[0] evs:", env.grids[0].evs)
print("Grid[1] evs:", env.grids[1].evs)



inc = env.create_incident(grid_index=0, location=(10.0, 20.0))
print("Incident:", inc.id, inc.gridIndex, inc.location)
print("Grid[0] incidents:", env.grids[0].incidents)

#------------Micro test - Idle EV in place---------------

ev.state = EvState.IDLE
ev.status = "available"
ev.sarns["action"] = ev.gridIndex  # <- add this
print("Before:", ev.aggIdleTime)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", ev.aggIdleTime)  # Expect +8
'''
#------------------Micro Test - Dispatching EV---------------
'''
ev.status = "Dispatching"
ev.state = EvState.IDLE

# Incident in grid 5
inc = env.create_incident(grid_index=5, location=(0.0,0.0))
ev.assignedPatientId = inc.id
ev.sarns["reward"] = None  # no reward yet

print("Before:", ev.gridIndex, ev.state, ev.status)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", ev.gridIndex, ev.state, ev.status)
'''
#------------------Repositiong-------------------
'''
ev.state = EvState.IDLE
ev.status = "Repositioning"
ev.nextGrid = 3
ev.sarns["reward"] = 0.8
ev.aggIdleTime = 0.0
ev.aggIdleEnergy = 0.0

print("Before:", ev.gridIndex, ev.aggIdleTime, ev.aggIdleEnergy)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", ev.gridIndex, ev.aggIdleTime, ev.aggIdleEnergy)
'''
#-------------------Incident cancellation---------------------
inc = env.create_incident(grid_index=0, location=(0.0,0.0))
inc.waitTime = P_MAX + 1  # Force over threshold

print("Before:", env.incidents.keys(), env.grids[0].incidents)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", env.incidents.keys(), env.grids[0].incidents)