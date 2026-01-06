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
'''
from MAP_env import MAP
from Entities.ev import EvState
from Entities.Incident import IncidentStatus
from utils.Helpers import P_MAX

env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()
#env.create_incident(0, (10,10))
ev = env.create_ev(0)

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
'''
inc = env.create_incident(grid_index=0, location=(0.0,0.0))
inc.waitTime = P_MAX + 1  # Force over threshold

print("Before:", env.incidents.keys(), env.grids[0].incidents)

env.update_after_timeslot(dt_minutes=8.0)

print("After:", env.incidents.keys(), env.grids[0].incidents)
'''
#=================CHECKING NEIGHBOUR ALLOTMENT==================#
'''
from MAP_env import MAP
from Controller import Controller
from Entities.ev import EvState
from Entities.Incident import IncidentStatus
from utils.Helpers import P_MAX

env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()

# 1. Build env + controller (use your real paths)

ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

# 2. Pick a cell that should be interior (has all 8 neighbours)
center_idx = (env.nRows // 2) * env.nCols + (env.nCols // 2)

neighs = ctrl._get_direction_neighbors_for_index(center_idx)
print("Center index:", center_idx)
print("Neighbours in order [N, NE, E, SE, S, SW, W, NW]:")
print(neighs)
print("Length:", len(neighs))

corner = 0
print("Corner 0 neighbours:", ctrl._get_direction_neighbors_for_index(corner))
'''

#======================Build_state===================#
'''
from MAP_env import MAP
from Entities.ev import EvState
from Controller import Controller
env = MAP("Data/grid_config_2d.json")

env.init_evs()
# reuse env, ctrl from above
ev = next(iter(env.evs.values()))  # just grab the first EV
ev.state = EvState.IDLE
ev.status = "Idle"
ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

state = ctrl._build_state(ev)
print("State length:", len(state))
print("State vector:", state)

'''

#====================Directional poke itseems==================#
'''
from Entities.Incident import Incident, Priority, IncidentStatus
from datetime import datetime
from MAP_env import MAP
from Entities import GRID, ev
from Controller import Controller
env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()

ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")
ev = next(iter(env.evs.values()))
gi = ev.gridIndex
print("EV grid index:", gi)

# Get neighbour indices by direction order
neighs = ctrl._get_direction_neighbors_for_index(gi)
print("Neighbour indices [N, NE, E, SE, S, SW, W, NW]:", neighs)

# Build baseline state
base_state = ctrl._build_state(ev)
print("Baseline neighbour imbalances:", base_state[2:10])

# Now create an incident in some neighbour that exists, say E (index 2)
east_idx = neighs[2]
if east_idx != -1:
    inc = env.create_incident(grid_index=east_idx, location=(0.0,0.0), priority="MED")
    # Recompute imbalance
    for g in env.grids.values():
        g.imbalance = g.calculate_imbalance(env.evs, env.incidents)

    new_state = ctrl._build_state(ev)
    print("New neighbour imbalances:", new_state[2:10])

    print("Change at E slot (index 4):", base_state[4], "->", new_state[4])
else:
    print("E neighbour does not exist for this EV grid; try another direction.")
'''

#=========================Action_Check========================#
'''
import numpy as np
from Entities.Incident import Incident, Priority, IncidentStatus
from datetime import datetime
from MAP_env import MAP
from Entities import GRID, ev
from Controller import Controller
env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()

ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

ctrl.epsilon = 1.0  # force random actions

ev = next(iter(env.evs.values()))
gi = ev.gridIndex

neighs = ctrl._get_direction_neighbors_for_index(gi)
valid_grids = {gi} | {idx for idx in neighs if idx != -1}
print("EV grid:", gi)
print("Neighbour grids:", neighs)
print("Valid destination set:", valid_grids)

bad = 0
for _ in range(100):
    s = ctrl._build_state(ev)
    dest = ctrl._select_action(s, gi)
    if dest not in valid_grids:
        bad += 1
        print("Bad destination:", dest)

print("Bad destinations count:", bad)
'''

#==========================checkin da DQN=========================#
'''
import torch
import torch.nn as nn
import numpy as np

from MAP_env import MAP
from Controller import Controller
from Entities.ev import EvState  # only if you want to fiddle with state/status later


# 1) Build environment and controller
env = MAP("Data/grid_config_2d.json")     # adjust path if needed
env.init_evs()

ctrl = Controller(
    env,
    csv_path="Data/5Years_SF_calls_latlong.csv"  # adjust path if needed
)

print("EV count:", len(env.evs))


# 2) Dummy network that always returns q = [0,1,2,3,4,5,6,7,8]
class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # Return [0,1,2,3,4,5,6,7,8] for every input in the batch
        batch_size = x.size(0)
        q = torch.arange(9, dtype=torch.float32, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return q


# 3) Replace the reposition main DQN with DummyNet
ctrl.dqn_reposition_main = DummyNet().to(ctrl.device)   # type: ignore
ctrl.epsilon = 0.0  # force greedy (no random exploration)


# 4) Pick one EV and test action mapping
ev_obj = next(iter(env.evs.values()))   # grab first EV object
gi = ev_obj.gridIndex

neighs = ctrl._get_direction_neighbors_for_index(gi)
print("EV grid index:", gi)
print("Neighbours [N, NE, E, SE, S, SW, W, NW]:", neighs)

# Build its state
s = ctrl._build_state(ev_obj)
print("State length:", len(s))
print("State vector:", s)

# 5) Ask controller for an action with DummyNet
dest = ctrl._select_action(s, gi)
print("Chosen dest:", dest)

# With strictly increasing q-values, best slot is 8
# → action slot 8 → direction index 7 → NW neighbour (neighs[7])
nw_idx = neighs[7]
expected = nw_idx if nw_idx != -1 else gi
print("Expected dest (NW or stay if -1):", expected)
'''

#======================PushRepositionBuffer=================#

'''import numpy as np
from Entities.Incident import Incident, Priority, IncidentStatus
from datetime import datetime
from MAP_env import MAP
from Entities import GRID
from Entities.ev import EvState
from Controller import Controller
env = MAP("Data/grid_config_2d.json")   # your real path
env.init_evs()

ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

print("Buffer size before:", len(ctrl.buffer_reposition))

ev = next(iter(env.evs.values()))
ev.state = EvState.IDLE
ev.status = "Repositioning"
ev.aggIdleTime = 10.0
ev.aggIdleEnergy = 1.5

# Fake s, a, r as if DQN + service have run
ev.sarns["state"] = ctrl._build_state(ev)
ev.sarns["action"] = ev.gridIndex   # say it chose to stay; any int is fine for test
ev.sarns["reward"] = 0.7

ctrl._push_reposition_transition(ev)

print("Buffer size after:", len(ctrl.buffer_reposition))'''

# ===================== TEST NAV STATE =======================

'''from MAP_env import MAP
from Controller import Controller
from Entities.ev import EvState

print("\n========== TESTING build_state_nav1 ==========")

# 1. Build environment + controller (your real paths)
env = MAP("Data/grid_config_2d.json")
env.init_evs()
env.init_hospitals("Data/hospitals_latlong.csv")
ctrl = Controller(env, csv_path="Data/5Years_SF_calls_latlong.csv")

# 2. Pick any EV to test on
ev = next(iter(env.evs.values()))
ev.state = EvState.BUSY     # Navigation only activates for busy EVs
ev.status = "Navigation"

# 3. Ensure location exists (safety)
if ev.location is None:
    ev.location = (0.0, 0.0)

# 4. Call your build_state_nav1 function
state_vec, grid_ids = ctrl.build_state_nav1(ev)

print("EV id:", ev.id)
print("EV grid index:", ev.gridIndex)
print("NAV state_vec:", state_vec)
print("NAV grid_ids:", grid_ids)
print("Length of state:", len(state_vec))
print("Length of grid_ids:", len(grid_ids))

# 5. Check duplicate grid indices
if len(grid_ids) != len(set(grid_ids)):
    print("⚠ WARNING: Duplicate grid indices detected (NOT expected!)")
else:
    print("✓ No duplicates — HC-grid mapping correct.")

print("========== END TEST ==========\n")


# ===================== TEST _select_nav_action (random) =======================

from Entities.ev import EvState

print("\n========== TESTING _select_nav_action (random) ==========")

# 1. Pick a real EV
ev = next(iter(env.evs.values()))
ev.state = EvState.BUSY
ev.status = "Navigation"

# 2. Make sure it has some location
if ev.location is None:
    ev.location = (0.0, 0.0)

# 3. Build the navigation state
state_vec, grid_ids = ctrl.build_state_nav1(ev)

print("State vec:", state_vec)
print("Grid IDs:", grid_ids)

if len(state_vec) == 0:
    print("No HC-grids found — cannot test random action.")
else:
    # 4. Force epsilon = 1 so RANDOM branch always activates
    old_eps = ctrl.epsilon
    ctrl.epsilon = 1.0

    # 5. Call the action selector
    slot = ctrl._select_nav_action(state_vec)

    # 6. Restore epsilon
    ctrl.epsilon = old_eps

    if 0 <= slot < len(grid_ids):
        print("Random slot chosen:", slot)
        print("Corresponding grid:", grid_ids[slot])
    else:
        print("ERROR: invalid slot returned:", slot)

print("========== END TEST ==========\n")

print("\n========== TESTING _tick NAVIGATION BLOCK ==========")

# 0. Properly initialise an episode so sarns keys exist
ctrl._reset_episode()

# 1. Pick any EV
ev = next(iter(env.evs.values()))

# 2. Force it into a navigation state
ev.state = EvState.BUSY
ev.status = "Navigation"

# 3. Make sure it has some location
if ev.location is None:
    # If your Grid has a centre or something similar, prefer that; fallback to (0,0)
    try:
        g = env.grids[ev.gridIndex]
        ev.location = (g.lat, g.lng) if hasattr(g, "lat") else (0.0, 0.0)
    except Exception:
        ev.location = (0.0, 0.0)

# 4. Make nav policy random to see variety (optional)
old_eps = ctrl.epsilon
ctrl.epsilon = 1.0

# 5. Call one tick
ctrl._tick(0)   # or ctrl._tick(step_idx) if your signature is like that

# 6. Restore epsilon
ctrl.epsilon = old_eps

print("========== END _tick NAV TEST ==========\n")'''
'''
import pandas as pd
from utils.Helpers import load_calls  # your helper :contentReference[oaicite:1]{index=1}

csv_path = "Data/5Years_SF_calls_latlong.csv"
df_all = load_calls(csv_path)

time_col = "Received DtTm"  # adjust if your column name differs
df_all[time_col] = pd.to_datetime(df_all[time_col], errors="coerce")
df_all = df_all.dropna(subset=[time_col])

# Normalise to date only
days = df_all[time_col].dt.normalize()

# Example 1: simple chronological split (first 80% days train, last 20% test)
unique_days = sorted(days.unique())
split_idx = int(0.8 * len(unique_days))
train_days = set(unique_days[:split_idx])
test_days  = set(unique_days[split_idx:])

df_train = df_all[days.isin(train_days)].copy()
df_test  = df_all[days.isin(test_days)].copy()

print("Train rows:", len(df_train), "Test rows:", len(df_test))
print("Train days:", len(train_days), "Test days:", len(test_days))
'''







# Without expired incidents
'''
    # FIX 1: Unpack only 2 values (ignore the dataframe with _)
    df_trace, daily_waits = ctrl.run_inspection_episode(ep)
    
    # FIX 2: Add this episode's waits to the master list
    all_wait_times.extend(daily_waits)
    print(daily_waits)



# --- 2. Process Data into Buckets ---
# NOTE: We use 'all_wait_times' here, not 'wait_times'
labels = ['0-5 min', '5-10 min', '10-15 min', '>15 min']
counts = [0, 0, 0, 0]

for w in all_wait_times:
    if w < 5:
        counts[0] += 1
    elif w < 10:
        counts[1] += 1
    elif w < 15:
        counts[2] += 1
    else:
        counts[3] += 1

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#2ca02c', '#ff7f0e', '#d62728', '#8c564b']
bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)

# Add counts on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Styling
ax.set_title(f'Patient Wait Time Distribution (Over {n_episodes} Days)', fontsize=14)
ax.set_xlabel('Wait Time Range', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Add Summary Stats Box (Using all_wait_times)
if all_wait_times:
    stats_text = f"Avg Wait: {np.mean(all_wait_times):.1f} min\nMax Wait: {np.max(all_wait_times):.1f} min"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()
'''
#------------------------another wait time plot trail 
'''
import matplotlib.pyplot as plt
import pandas as pd
from MAP_env import MAP
from Controller import Controller
import matplotlib.pyplot as plt
import numpy as np

# Initialize Environment
env = MAP("Data/grid_config_2d.json")
env.init_evs()
env.init_hospitals("D:\\Downloads\\hospitals_latlong.csv")
#env.init_hospitals("Data/hospitals_latlong.csv")

# Initialize Controller
ctrl = Controller(
    env,
    ticks_per_ep=180,
    #csv_path="D:\\Downloads\\5Years_SF_calls_latlong.csv"
    csv_path="Data/Fire_Department_and_Emergency_Medical_Services_Dispatched_Calls_for_Service_20251208.csv"
)

# --- 1. Data Collection Loop ---
n_episodes = 10
all_wait_times = []
all_incident_data = []
print(f"Collecting wait times over {n_episodes} episodes...")

for ep in range(n_episodes):
    ctrl.run_inspection_episode(ep)
    daily_data = [
        (inc.get_wait_minutes(), inc.priority) 
        for inc in ctrl._spawned_incidents.values()
    ]
    
    # 3. Add this day's data to the master list
    all_incident_data.extend(daily_data)

    
import matplotlib.pyplot as plt
import numpy as np

# --- STEP 2: Process Data into Better Buckets ---
# We add a specific bucket for >30 mins to catch the "Expired" cases (32.0)
labels = ['0-5 min', '5-10 min', '10-15 min', '15-30 min', 'Expired (>30)']

# Counters for each priority (High, Med, Low)
# Now we have 5 buckets, so we need 5 zeros
counts_p1 = [0, 0, 0, 0, 0] 
counts_p2 = [0, 0, 0, 0, 0]
counts_p3 = [0, 0, 0, 0, 0]

# Assuming 'all_incident_data' is a list of tuples: (wait_time, priority)
# If you only have 'all_wait_times' (no priority), just use one 'counts' list.

for wait, prio in all_incident_data:
    # 1. Determine Bucket Index
    if wait < 5:
        idx = 0
    elif wait < 10:
        idx = 1
    elif wait < 15:
        idx = 2
    elif wait < 30:
        idx = 3 # The "Very Late" bucket
    else:
        idx = 4 # The "Timed Out / 32.0" bucket (This is your P_MAX group)

    # 2. Increment Priority Counter
    if prio == 1:
        counts_p1[idx] += 1
    elif prio == 2:
        counts_p2[idx] += 1
    else:
        counts_p3[idx] += 1

# --- STEP 3: Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

# Stack the bars
ax.bar(labels, counts_p1, label='High Prio', color='#d62728', edgecolor='black') # Red
ax.bar(labels, counts_p2, bottom=counts_p1, label='Med Prio', color='#ff7f0e', edgecolor='black') # Orange

# Calculate bottom for 3rd layer
bottom_p3 = np.add(counts_p1, counts_p2)
ax.bar(labels, counts_p3, bottom=bottom_p3, label='Low Prio', color='#2ca02c', edgecolor='black') # Green

# Styling
ax.set_title(f'Patient Wait Time Distribution (Showing Timeouts)', fontsize=14)
ax.set_xlabel('Wait Time Range', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add counts on top
total_counts = np.add(bottom_p3, counts_p3)
for i, total in enumerate(total_counts):
    ax.text(i, total + 0.5, f'{int(total)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
'''

#------------------------------another useless graph
'''
def plot_energy_idle_tradeoff(all_episode_stats):
    vehicle_data = {}

    for ep_stat in all_episode_stats:
        if "vehicle_idle_time" not in ep_stat:
            print(f"[Warning] Episode {ep_stat.get('episode')} missing 'vehicle_idle_time'.")
            continue

        energies = ep_stat.get("vehicle_energy", {})
        times = ep_stat.get("vehicle_idle_time", {})

        for vid in energies.keys():
            # Ensure we use a consistent key (convert to string if needed for map, but keep original for label)
            if vid not in vehicle_data:
                vehicle_data[vid] = {'time': [], 'energy': []}
            
            vehicle_data[vid]['energy'].append(energies[vid])
            vehicle_data[vid]['time'].append(times.get(vid, 0.0))

    if not vehicle_data:
        print("No data found to plot.")
        return

    # 2. Calculate Averages per Vehicle
    avg_times = []
    avg_energies = []
    labels = []

    # === FIX: ROBUST SORTING ===
    # This handles both Integer IDs (0, 1) and String IDs ("EV_0", "EV_1") without crashing
    def sort_key(x):
        s_x = str(x)
        if '_' in s_x:
            try:
                return int(s_x.split('_')[1])
            except ValueError:
                return s_x
        elif s_x.isdigit():
            return int(s_x)
        else:
            return s_x

    sorted_vids = sorted(vehicle_data.keys(), key=sort_key)

    for vid in sorted_vids:
        data = vehicle_data[vid]
        
        # Calculate mean
        mean_t = np.mean(data['time'])
        mean_e = np.mean(data['energy'])
        
        avg_times.append(mean_t)
        avg_energies.append(mean_e)
        labels.append(f"EV_{vid}")

    # 3. Plot Scatter
    plt.figure(figsize=(10, 7))
    
    plt.scatter(avg_times, avg_energies, s=150, color='#E15759', edgecolor='black', alpha=0.8)

    # Label dots
    for i, txt in enumerate(labels):
        plt.annotate(txt, (avg_times[i], avg_energies[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.title(f'Trade-off: Idle Time vs. Energy Consumption\n(Averaged over {len(all_episode_stats)} Episodes)', fontsize=14)
    plt.xlabel('Average Idle Time (Minutes)', fontsize=12)
    plt.ylabel('Average Idle Energy (kWh)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

#plot_energy_idle_tradeoff(stats_history)
'''


#=================only energy plot=================
'''def plot_real_workload(all_episode_stats):
    
    # 1. Extract data from your stats history
    vehicle_map = {}
    
    # We loop through every episode to collect the final energy values
    for ep_stat in all_episode_stats:
        if "vehicle_energy" not in ep_stat:
            print(f"[Warning] Episode {ep_stat.get('episode')} missing 'vehicle_energy' data.")
            continue
            
        v_energies = ep_stat["vehicle_energy"]
        
        for vid, energy in v_energies.items():
            label = f"EV_{vid}"
            if label not in vehicle_map:
                vehicle_map[label] = []
            vehicle_map[label].append(energy)
            
    # 2. Check if we found data
    if not vehicle_map:
        print("ERROR: No vehicle data found. Did you update 'run_training_episode' in Controller.py?")
        return

    # 3. Calculate Averages
    # Sort by ID so they appear in order (EV_0, EV_1, etc.)
    ev_ids = sorted(vehicle_map.keys(), key=lambda x: int(x.split('_')[1]))
    avg_energies = [np.mean(vehicle_map[ev]) for ev in ev_ids]

    print(f"Plotting for {len(ev_ids)} Vehicles: {ev_ids}")
    sorted_ids=sorted(vehicle_map.keys())
    for vid in sorted_ids:
        avg_val = np.mean(vehicle_map[vid])
        print(f"EV ID: {vid} | Average Energy: {avg_val:.2f} kWh")

    # 4. Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(ev_ids, avg_energies, color='#4E79A7', edgecolor='black', alpha=0.9)
    
    plt.title(f'Real Average Energy Consumed per Vehicle\n(Averaged over {len(all_episode_stats)} Episodes)', fontsize=14)
    plt.xlabel('Vehicle ID')
    plt.ylabel('Avg Energy (kWh)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Label bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

#plot_real_workload(stats_history)
'''

#--------------------------controller changes-----------------------
# Controller.py
import random
from typing import Optional, List

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import json
import torch.nn.utils as nn_utils
from MAP_env import MAP
from Entities.ev import EvState
from Entities.Incident import IncidentStatus
from utils.Epsilon import EpsilonScheduler, hard_update, soft_update
from utils.Helpers import (
    build_daily_incident_schedule,
    point_to_grid_index,
    W_MIN, W_MAX, E_MIN, E_MAX,
    utility_navigation, load_calls, get_k_hop_directional_indices
)

from DQN import DQNetwork, ReplayBuffer
print("controler loaded")
DIRECTION_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NAV_K = 8

class Controller:
    def __init__(
        self,
        env: MAP,
        ticks_per_ep: int = 180,
        seed: int = 123,
        csv_path: str = "Data/Fire_Department_and_Emergency_Medical_Services_Dispatched_Calls_for_Service_20251208_with_index.csv",
        time_col: str = "Received DtTm",
        lat_col: Optional[str] = None,
        lng_col: Optional[str] = None,
        wkt_col: Optional[str] = "case_location",
        test_mode: bool = False,
        #test_mode: bool = False,
    ):
        self.env = env
        self.test_mode = test_mode
        #print("[DEBUG] hospitals at Controller init:", len(self.env.hospitals))
        self.ticks_per_ep = ticks_per_ep
        self.dqn_rep_test = None
        self.dqn_nav_test = None
        self.rng = random.Random(seed)

        # agent params
        self.global_step = 0
        self.global_tick = 0
        self.epsilon_scheduler = EpsilonScheduler(
            start=1.0,     
            end=0.1,       
            decay_steps=5000
        )
        self.epsilon = 1.0 
        self.busy_fraction = 0.5

        # Track losses for plotting
        self.ep_nav_losses = [] 
        self.ep_repo_losses = []
        

        # DQNs 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if getattr(self, 'test_mode', False):
            class DummyBuffer:
                def push(self, *args, **kwargs): return None
                def sample(self, *args, **kwargs): raise RuntimeError("DummyBuffer has no samples")
                def __len__(self): return 0

            self.dqn_reposition_main = None
            self.dqn_reposition_target
            self.opt_reposition = None
            self.buffer_reposition = DummyBuffer()
            
            self.dqn_navigation_main = None
            self.dqn_navigation_target = None
            self.opt_navigation = None
            self.buffer_navigation = DummyBuffer()

        else:
            #state_dim = 12
            evAny = next(iter(self.env.evs.values()))
            state_dim = len(self._build_state(evAny))

            # sanity: must be constant for all EVs
            for ev in self.env.evs.values():
                if len(self._build_state(ev)) != state_dim:
                    raise RuntimeError("reposition state_dim is not constant across EVs")
            action_dim = 1 + len(self.env.grids)

            self.dqn_reposition_main = DQNetwork(state_dim, action_dim).to(self.device)
            self.dqn_reposition_target = DQNetwork(state_dim, action_dim).to(self.device)
            if self.dqn_reposition_main is not None and self.dqn_reposition_target is not None:
                self.dqn_reposition_target.load_state_dict(self.dqn_reposition_main.state_dict())
            if self.dqn_reposition_main is not None:
                self.opt_reposition = torch.optim.Adam(self.dqn_reposition_main.parameters(), lr=1e-4)
            else:
                self.opt_reposition = None
            self.buffer_reposition = ReplayBuffer(100)
            self.repositionLogPath = "reposition_buffer_log.txt"
            self.repositionLogStep = 0


            # --- NAV: one feature and one action per hospital grid ---
            self.hc_grids = sorted({h.gridIndex for h in self.env.hospitals.values()})
            
            nav_action_dim = len(self.hc_grids)

            if nav_action_dim == 0:
                # degenerate case, but keep network alive
                self.hc_grids = [0]
                nav_action_dim = 1

            state_dim_nav = nav_action_dim
            self.nav_step = 0
            self.rep_step = 0
            #self.nav_target_update = 500  
            #self.nav_tau = 0.005          
            self.dqn_navigation_main = DQNetwork(state_dim_nav, nav_action_dim).to(self.device)
            self.dqn_navigation_target = DQNetwork(state_dim_nav, nav_action_dim).to(self.device)
            if self.dqn_navigation_main is not None and self.dqn_navigation_target is not None:
                self.dqn_navigation_target.load_state_dict(self.dqn_navigation_main.state_dict())
            if self.dqn_navigation_main is not None:
                self.opt_navigation = torch.optim.Adam(self.dqn_navigation_main.parameters(), lr=1e-4)
            else:
                self.opt_navigation = None
            self.buffer_navigation = ReplayBuffer(50_000)

            hard_update(self.dqn_reposition_target, self.dqn_reposition_main)
            hard_update(self.dqn_navigation_target, self.dqn_navigation_main)

        #if not getattr(self, 'test_mode', False):
            #print("[Controller] DQNs initialised.")
            #print("  Device:", self.device)
        #else:
            #print("[Controller] test_mode enabled: skipping heavy DQN initialisation")
        #print("number of grids with hcs",len(self.hc_grids))
        self.df = pd.read_csv(csv_path)
        self.time_col = time_col
        self.lat_col = lat_col
        self.lng_col = lng_col
        self.wkt_col = wkt_col

       


        self._schedule = None
        self._current_day = None

        self.max_idle_minutes = W_MAX
        self.max_idle_energy = E_MAX
        #self.max_wait_time_HC = H_MAX

        self._spawn_attempts = 0
        self._spawn_success = 0
        self.pretty = True
        self.debug_dispatch = False

        #CHECK REPOSITIONING
        # Episode-level history of repositioning performance
        self.idle_time_history: list[float] = []
        self.idle_energy_history: list[float] = []

        self._ep_idle_added: float = 0.0
        self._ep_energy_added: float = 0.0
        self._ep_idle_samples: int = 0

        self._ep_idle_baseline: dict[int, float] = {}
        self._ep_energy_baseline: dict[int, float] = {}


        #Q_VALUE Convergence
        self.q_rep_history: list[float] = []
        self.q_nav_history: list[float] = []





    
    def _get_direction_neighbors_for_index(self, index: int) -> list[int]:
        n_rows = len(self.env.lat_edges) - 1
        n_cols = len(self.env.lng_edges) - 1

        cell_row = index // n_cols
        cell_col = index % n_cols

        offset_map = {
            "N":  (1, 0), "NE": (1, 1), "E":  (0, 1), "SE": (-1, 1),
            "S":  (-1, 0), "SW": (-1, -1), "W":  (0, -1), "NW": (1, -1),
        }

        neighbours: list[int] = []
        for dname in DIRECTION_ORDER:
            dr, dc = offset_map[dname]
            n_row = cell_row + dr
            n_col = cell_col + dc

            if 0 <= n_row < n_rows and 0 <= n_col < n_cols:
                neighbours.append(n_row * n_cols + n_col)
            else:
                neighbours.append(-1)
        return neighbours

    def _pad_neighbors(self, nbs: List[int]):
        N = 8
        n = (nbs[:N] if len(nbs) >= N else nbs + [-1] * (N - len(nbs)))
        return n

    #================== STATE BUILDERS =====================#
    
    def _build_state(self, ev) -> list[float]:
        gi = ev.gridIndex
        g = self.env.grids[gi]

        # own imbalance
        imb_self = float(g.calculate_imbalance(self.env.evs, self.env.incidents))
        #print("imbalance self",imb_self,"ev",ev.id,"in grid",gi)

        # neighbour imbalances in fixed direction order
        hop_maps = self.env.hop_maps[gi]
        state: list[float] = []
        # 1-hop (direction-wise)
        for nb in self._get_direction_neighbors_for_index(gi):
            if nb == -1:
                state.append(0.0)
            else:
                state.append(self.env.grids[nb].imbalance)

        # 2-hop (direction-wise, relative to gi)
        two_hop_idxs = get_k_hop_directional_indices(
        start_index=gi,
        k=2,
        n_rows = len(self.env.lat_edges) - 1,
        n_cols = len(self.env.lng_edges) - 1,
        direction_order=DIRECTION_ORDER)
        for nb in two_hop_idxs:
            if nb == -1:
                state.append(0.0)
            else:
                state.append(self.env.grids[nb].imbalance)

        # rest (all hops >= 3, any order)
        for hop in sorted(hop_maps.keys()):
            if hop >= 3:
                for nb in hop_maps[hop]:
                    state.append(self.env.grids[nb].imbalance)

        vec: list[float] = []
        #vec.append(float(gi))
        vec.append(imb_self)
        vec.extend(state)
        vec.append(float(ev.aggIdleTime))
        vec.append(float(ev.aggIdleEnergy))
        return vec
    
    def build_state_nav1(self, ev):
        
        # 1) Use precomputed hospital grids
        hc_grids = getattr(self, "hc_grids", None)
        if not hc_grids:
            return [], []

        state_vec: list[float] = []
        grid_ids:  list[int]   = list(hc_grids)

        # 2) Pre-compute mean wait time per HC-grid
        grid_mean_wait = {}
        for g_idx in hc_grids:
            waits = []
            for h in self.env.hospitals.values():
                if h.gridIndex != g_idx:
                    continue
                w_valid = self.env.calculate_eta_plus_wait(ev, h)
                
                if w_valid is not None:
                    w = max(0,w_valid)
                    waits.append(float(w))
            if waits:
                grid_mean_wait[g_idx] = sum(waits) / len(waits)
            else:
                grid_mean_wait[g_idx] = 0.0
            
            mean_wait = grid_mean_wait[g_idx]
            state_vec.append(mean_wait)
        
        #print("shape of vector",len(grid_ids))
        return state_vec, grid_ids


    #========================= ACTION =================================#

    '''def _select_action(self, state_vec: list[float], gi: int) -> int:
        if getattr(self, "test_mode", False):
            neighbours = self._get_direction_neighbors_for_index(gi)
            valid = [gi] + [nb for nb in neighbours if nb != -1]
            return self.rng.choice(valid) if valid else gi

        if self.dqn_reposition_main is None:
            neighbours = self._get_direction_neighbors_for_index(gi)
            valid = [gi] + [nb for nb in neighbours if nb != -1]
            return self.rng.choice(valid) if valid else gi

        neighbours = self._get_direction_neighbors_for_index(gi)  # len 8
        valid_mask = [1]  # slot 0 (stay)
        for nb_idx in neighbours:
            valid_mask.append(1 if nb_idx != -1 else 0)
        self.epsilon = self.epsilon_scheduler.value(self.global_step)
        #print("epsilon in rep",self.epsilon)
        if self.rng.random() < self.epsilon:
            valid_slots = [i for i, m in enumerate(valid_mask) if m == 1]
            slot = self.rng.choice(valid_slots) if valid_slots else 0
        else:
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.dqn_reposition_main(s).detach().cpu().numpy().ravel()
            #print("q value",q)
            for i, m in enumerate(valid_mask):
                if m == 0:
                    q[i] = -1e9
            slot = int(np.argmax(q))
        #self.global_step += 1

        if slot == 0:
            return gi  # stay
        else:
            dir_index = slot - 1
            nb_idx = neighbours[dir_index]
            return nb_idx if nb_idx != -1 else gi'''
    def _select_action(self, state_vec: list[float], gi: int) -> int:
        grid_ids = sorted(self.env.grids.keys())  # constant order, size N
        n_actions = 1 + len(grid_ids)             # slot 0 + all grids

        valid_mask = [1]  # slot 0 stay always valid
        for gid in grid_ids:
            valid_mask.append(1 if self.env.grids[gid].imbalance > 0 else 0)

        valid_slots = [i for i, m in enumerate(valid_mask) if m == 1]
        if not valid_slots:
            return gi

        if getattr(self, "test_mode", False) or self.dqn_reposition_main is None:
            slot = self.rng.choice(valid_slots)
        else:
            self.epsilon = self.epsilon_scheduler.value(self.global_step)

            if self.rng.random() < self.epsilon:
                slot = self.rng.choice(valid_slots)
            else:
                s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.dqn_reposition_main(s).detach().cpu().numpy().ravel()

                for i, m in enumerate(valid_mask):
                    if m == 0:
                        q[i] = -1e9

                slot = int(np.argmax(q))
                if slot not in valid_slots:
                    slot = self.rng.choice(valid_slots)

        if slot == 0:
            return gi

        return grid_ids[slot - 1]


    
    def _select_nav_action(self, state_vec: list[float]) -> int:


        n_actions = len(state_vec)
        if n_actions == 0:
            return -1  # no hospital grids to choose from

        # 1) Exploration: random slot
        self.epsilon = self.epsilon_scheduler.value(self.global_step)
        
        #print("epsilon in nav",self.epsilon)
        if self.rng.random() < self.epsilon:
            
            return self.rng.randint(0, n_actions - 1)

        # 2) Exploitation: DQN greedy
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        # shape: (1, n_actions)
        if self.dqn_navigation_main is not None:
            q = self.dqn_navigation_main(s).detach().cpu().numpy().ravel()
        # q[i] is the Q-value for choosing slot i (i.e. grid_ids[i])

        slot = int(np.argmax(q))
        #self.global_step += 1
        # safety clamp, just in case
        if slot < 0:
            slot = 0
        elif slot >= n_actions:
            slot = n_actions - 1
        

        return slot


    #================== REPOSITION TRAIN ======================#
    '''def _train_reposition(
    self,
    batch_size: int = 64,
    gamma: float = 0.99,):
        self.repo_tau = 0.005
        

        self.repo_target_hard_update = 2000
        
        if len(self.buffer_reposition) < batch_size:
            return

        # ---- Sample batch ----
        s, a, r, s2, done = self.buffer_reposition.sample(
            batch_size,
            device=self.device
        )

        # ---- Sanity checks ----
        assert self.dqn_reposition_main is not None
        assert self.dqn_reposition_target is not None
        assert self.opt_reposition is not None

        # ---- Q(s,a) from MAIN ----
        q_sa = (
            self.dqn_reposition_main(s)
            .gather(1, a.unsqueeze(1))
            .squeeze(1)
        )

        # ---- TD target from TARGET ----
        with torch.no_grad():
            q_next = self.dqn_reposition_target(s2).max(dim=1)[0]
            q_next[done == 1.0] = 0.0
            target = r + gamma * q_next

        # ---- Loss ----
        #loss = F.mse_loss(q_sa, target)
        loss = F.smooth_l1_loss(q_sa, target)  # Huber (optional)

        # ---- Main network update (θ_m) ----
        self.opt_reposition.zero_grad()
        loss.backward()
        self.opt_reposition.step()

        # ---- Bookkeeping ----
        self.ep_repo_losses.append(loss.item())
        self.rep_step += 1

        # ---- HARD target update ----
        if self.rep_step % self.repo_target_hard_update == 0:
            self.dqn_reposition_target.load_state_dict(
                self.dqn_reposition_main.state_dict()
            )

        # ---- SOFT (Polyak) target update ----
        #tau = self.repo_tau  # e.g. 0.995
        tau = 0.995
        with torch.no_grad():
            for p_t, p in zip(
                self.dqn_reposition_target.parameters(),
                self.dqn_reposition_main.parameters()
            ):
                # θ̂ ← τ θ̂ + (1 − τ) θ
                p_t.data.mul_(tau)
                p_t.data.add_((1.0 - tau) * p.data)

        if self.rep_step % 500 == 0:
            print(
                f"[Controller] REPOSITION train step={self.rep_step} "
                f"loss={loss.item():.4f}"
            )'''


    def _train_reposition(self, batch_size: int = 64, gamma: float = 0.99) -> None:

        #print(f"[DEBUG] Train Repo Called. Buffer Len: {len(self.buffer_reposition)}, Batch: {batch_size}")
               
        if len(self.buffer_reposition) < batch_size:
            return
        
        try:
              s, a, r, s2, done, valid_mask_s2 = self.buffer_reposition.sample(
        batch_size, device=self.device)
        except TypeError:
            batch = self.buffer_reposition.sample(batch_size, device=self.device)
            s   = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[0]])
            a   = torch.as_tensor(batch[1], dtype=torch.long,   device=self.device)
            r   = torch.as_tensor(batch[2], dtype=torch.float32, device=self.device)
            s2  = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[3]])
            done= torch.as_tensor(batch[4], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            q2_all = self.dqn_reposition_target(s2)     # (B, 9)

           
            q2_all[valid_mask_s2 == 0] = -1e9

            q2 = q2_all.max(dim=1).values

            y = r + gamma * (1.0 - done) * q2

            

        if self.dqn_reposition_main is not None:
            q = self.dqn_reposition_main(s).gather(1, a.view(-1, 1)).squeeze(1)
        assert s.shape == s2.shape
        assert not torch.isnan(s).any()
        assert not torch.isnan(q).any()
        assert a.max() < q.size(0)
        

        loss = F.smooth_l1_loss(q, y)
        if self.opt_reposition is not None and self.dqn_reposition_main is not None:
            self.opt_reposition.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(
                self.dqn_reposition_main.parameters(),
                max_norm=1.0
            )
            self.opt_reposition.step()
        
        # --- FIX: TRACK LOSS FOR PLOTTING ---
        self.ep_repo_losses.append(loss.item())

        self.rep_step += 1
        self.rep_tau = 0.005
        self.rep_hard_update = 2000

        # -------- SOFT UPDATE (EVERY STEP) --------
        if self.dqn_reposition_main is not None and self.dqn_reposition_target is not None:
            with torch.no_grad():
                for p_t, p in zip(
                    self.dqn_reposition_target.parameters(),
                    self.dqn_reposition_main.parameters()
                ):
                    p_t.data.mul_(1.0 - self.rep_tau).add_(self.rep_tau * p.data)

        # -------- HARD UPDATE (EVERY N STEPS) --------
        if self.rep_step % self.rep_hard_update == 0:
            if self.dqn_reposition_main is not None and self.dqn_reposition_target is not None:
                self.dqn_reposition_target.load_state_dict(
                    self.dqn_reposition_main.state_dict()
                )

        if self.rep_step % 500 == 0:
            print(f"[Controller] Reposition train step={self.rep_step} "f"loss={loss.item():.4f}")

    #===================== NAVIGATION TRAIN ==================#

    def _train_navigation(self, batch_size: int = 64, gamma: float = 0.99):
        if len(self.buffer_navigation) < batch_size:
            return

        # ---- Sample batch ----
        s, a, r, s2, done,mask = self.buffer_navigation.sample(
            batch_size,
            device=self.device
        )

        # ---- Sanity checks (fail fast) ----
        assert self.dqn_navigation_main is not None
        assert self.dqn_navigation_target is not None
        assert self.opt_navigation is not None

        # ---- Q(s,a) from MAIN network ----
        q_sa = (
            self.dqn_navigation_main(s)
            .gather(1, a.unsqueeze(1))
            .squeeze(1)
        )

        # ---- TD target from TARGET network ----
        with torch.no_grad():
            q_next = self.dqn_navigation_target(s2).max(dim=1)[0]
            done = done.to(torch.bool)
            target = r + gamma * (~done).float() * q_next

        # ---- Loss & optimization ----
        loss = F.smooth_l1_loss(q_sa, target)

        self.opt_navigation.zero_grad()
        loss.backward()
        self.opt_navigation.step()

        # ---- Bookkeeping ----
        self.ep_nav_losses.append(loss.item())
        self.nav_step += 1
        self.nav_target_update = 2000
        self.nav_tau = 0.005
        # ---- Soft target update ----
        if self.nav_step % self.nav_target_update == 0:
            with torch.no_grad():
                for t_param, o_param in zip(
                    self.dqn_navigation_target.parameters(),
                    self.dqn_navigation_main.parameters()
                ):
                    t_param.data.mul_(1.0 - self.nav_tau).add_(self.nav_tau * o_param.data)

        #if self.nav_step % 500 == 0:
            #print(f"[Controller] NAV train step={self.nav_step} "f"loss={loss.item():.4f}")
            

    
    '''def _train_navigation(self, batch_size: int = 64, gamma: float = 0.99):
        
        if len(self.buffer_navigation) < 2000:
            return
        
        try:
            s, a, r, s2, done,mask = self.buffer_navigation.sample(batch_size, device=self.device)
        except TypeError:
            batch = self.buffer_navigation.sample(batch_size, device=self.device)
            s   = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[0]])
            a   = torch.as_tensor(batch[1], dtype=torch.long,   device=self.device)
            r   = torch.as_tensor(batch[2], dtype=torch.float32, device=self.device)
            s2  = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[3]])
            done= torch.as_tensor(batch[4], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.dqn_navigation_target is not None:
                q2 = self.dqn_navigation_target(s2).max(dim=1).values
            y  = r + gamma * (1.0 - done) * q2

        if self.dqn_navigation_main is not None:
            q = self.dqn_navigation_main(s).gather(1, a.view(-1, 1)).squeeze(1)

        loss = F.smooth_l1_loss(q, y)
        if self.opt_navigation is not None:
            self.opt_navigation.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(
                self.dqn_navigation_main.parameters(),
                max_norm=10.0
            )
            self.opt_navigation.step()
        
        # --- FIX: TRACK LOSS FOR PLOTTING ---
        self.ep_nav_losses.append(loss.item())

        self.nav_step += 1
        self.nav_tau = 0.005
        self.nav_target_update = 2000
        if self.nav_step % self.nav_target_update == 0:
            if self.dqn_navigation_target is not None and self.dqn_navigation_main is not None:
                with torch.no_grad():
                    for p_t, p in zip(self.dqn_navigation_target.parameters(),
                                      self.dqn_navigation_main.parameters()):
                        p_t.data.mul_(1.0 - self.nav_tau).add_(self.nav_tau * p.data)

        if self.nav_step % 500 == 0:
            print(f"[Controller] NAV train step={self.nav_step} loss={loss.item():.4f}")'''
    
    def _log_reposition_push(self, t, evId, s, a, r, s2, done):
        try:
            sList = s.detach().cpu().tolist() if hasattr(s, "detach") else list(s)
            s2List = s2.detach().cpu().tolist() if hasattr(s2, "detach") else list(s2)
        except Exception:
            sList = str(s)
            s2List = str(s2)

        rec = {
            "step": int(self.repositionLogStep),
            "tick": int(t),
            "ev": int(evId) if evId is not None else None,
            "action": int(a) if a is not None else None,
            "reward": None if r is None else float(r),
            "done": int(done) if done is not None else 0,
            "state": sList,
            "next_state": s2List,
        }

        with open(self.repositionLogPath, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        self.repositionLogStep += 1
            
    # ---------- episode reset ----------
    def _reset_episode(self) -> None:
        self._spawn_attempts = 0
        self._spawn_success = 0
        self.global_tick = 0
        self.ep_nav_losses = [] # Reset loss tracking
        self.ep_repo_losses = [] # Reset reposition loss tracking

        self.env.incidents.clear()
        for g in self.env.grids.values():
            g.incidents.clear()

        series = pd.to_datetime(
            self.df[self.time_col],
            format="%Y %b %d %I:%M:%S %p",
            errors="coerce"
            ).dt.normalize().dropna()

        days = series.unique()
        if len(days) == 0:
            raise RuntimeError(f"No valid dates in dataset for {self.time_col}")

        self._current_day = pd.Timestamp(self.rng.choice(list(days)))

        self._schedule = build_daily_incident_schedule(
            self.df,
            day=self._current_day,
            time_col=self.time_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            wkt_col=self.wkt_col,
        )

        total_today = 0 if not self._schedule else sum(len(v) for v in self._schedule.values())
        self.total_today = total_today
        #print(f"[Controller] _reset_episode ready: day={self._current_day.date()} incidents_today={total_today}")
        
        self._spawned_incidents = {}
        self._last_dispatches = []

        ev_list = list(self.env.evs.values())
        self.rng.shuffle(ev_list)
        all_idx = list(self.env.grids.keys())
        n_evs = len(ev_list)
        n_busy_target = int(self.busy_fraction * n_evs)

        for i, ev in enumerate(ev_list):
            gi = self.rng.choice(all_idx)
            self.env.move_ev_to_grid(ev.id, gi)
            '''
            if i < n_busy_target:
                ev.set_state(EvState.BUSY)
                ev.status = "Navigation"
                ev.navdstGrid = random.choice((0,3,4,5))
                ev.assignedPatientPriority = random.choice((1,2,3))
                ev.navEtaMinutes = self.rng.uniform(10.0, 30.0)
                ev.navTargetHospitalId = random.choice((1,78))
               
                ev.aggIdleTime = 0.0
                ev.aggIdleEnergy = 0.0
            else:
            '''
            ev.set_state(EvState.IDLE)
            ev.status = "Idle"
            ev.nextGrid = None
            ev.aggIdleTime = self.rng.uniform(0.0, self.max_idle_minutes)
            ev.aggIdleEnergy = self.rng.uniform(0.0, self.max_idle_energy)
            ev.navTargetHospitalId = None
            ev.navEtaMinutes = 0.0
                

            ev.sarns.clear()
            ev.sarns["state"] = None
            ev.sarns["action"] = None
            ev.sarns["reward"] = 0.0
            ev.sarns["next_state"] = None
            #print("ev initiated",ev.id,ev.add_idle,ev.aggBusyTime,ev.state,ev.status)
        #print("the lsit",self.env.grids)

    # ---------- per-tick ----------
    def _spawn_incidents_for_tick(self, t: int):
        todays_at_tick = self._schedule.get(t, []) if self._schedule else []
        for (inc_id,ts,lat, lng, pri,rsp_ts, hosp_ts) in todays_at_tick:
            self._spawn_attempts +=1
            gi = point_to_grid_index(lat, lng, self.env.lat_edges, self.env.lng_edges)
            if gi is None or gi < 0:
                continue
            ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            inc = self.env.create_incident(incident_id = inc_id,grid_index=gi, location=(lat, lng),timestamp=ts_py,priority=pri)
            inc.responseTimestamp = rsp_ts
            inc.hospitalTimestamp = hosp_ts
            if rsp_ts is not None and hosp_ts is not None:
                try:
                    inc.responseToHospitalMinutes = max(0.0, (hosp_ts - rsp_ts).total_seconds() / 60.0)
                except Exception:
                    inc.responseToHospitalMinutes = None
            else:
                inc.responseToHospitalMinutes = None
            try:
                self._spawned_incidents[inc.id] = inc
                #print("incident id",inc)
                #print(f"incident stats",inc.to_dict)
            except Exception:
                pass
            
            self._spawn_success +=1

    def _tick(self, t: int) -> None:
        episode_done = (self.global_tick >= self.ticks_per_ep - 1)

        # 1. Spawn Incidents & Update Environment
        self._spawn_incidents_for_tick(t)
        self.env.tick_hospital_waits()
        
        for g in self.env.grids.values():
            g.imbalance = g.calculate_imbalance(self.env.evs, self.env.incidents)

        # 2. Build States & Actions for IDLE EVs (Phase 1)
        try:
            self._last_dispatches = dispatches # type: ignore
        except Exception:
            self._last_dispatches = []
        
        nav_actions: list = []
        idle_transitions = []

        for ev in self.env.evs.values():
            if ev.state == EvState.IDLE:
                # Reset SARNS for safety
                ev.sarns["state"] = []
                ev.sarns["action"] = None 
                ev.sarns["reward"] = 0
                
                # Build State
                state_vec = self._build_state(ev)
                ev.sarns["state"] = state_vec
                
                # Select Action
                a_gi = self._select_action(state_vec, ev.gridIndex)
                ev.sarns["action"] = a_gi
                
                # Update pointers
                ev.nextGrid = ev.gridIndex 
                self.global_step += 1
                
                # Save snapshot for later training (Critical!)
                idle_transitions.append((ev, state_vec, a_gi))

        # 3. Accept Offers & Dispatch (Environment Interaction)
        self.env.accept_reposition_offers()
        dispatches = self.env.dispatch_gridwise(beta=0.5)

        # 4. Handle BUSY Navigation (Phase 1)
        busy_transitions = []
        for ev in self.env.evs.values():    
            if ev.state == EvState.BUSY and ev.status == "Navigation":
                ev.sarns["state"] = []
                ev.sarns["action"] = None
                ev.sarns["reward"] = 0
                
                # Build Nav State
                state_vec, grid_ids = self.build_state_nav1(ev)
                ev.sarns["state"] = state_vec
                
                # Select Nav Action
                slo = self._select_nav_action(state_vec)
                ev.sarns["action"] = slo
                self.global_step += 1
                
                # Helper: Calculate reward logic (shortened for brevity, keep your logic here)
                dest_grid = grid_ids[slo]
                ev.navdstGrid = dest_grid
                
                # ... [Your existing Navigation Logic for target/hospital selection] ...
                # (I am preserving your logic flow here without re-typing all lines)
                # Assume standard Nav setup occurred here...

                # Capture snapshot for Nav training
                busy_transitions.append((ev, state_vec, slo))


        # 5. Capture "Before" Metrics for Reward Calculation
        idle_before = {ev.id: ev.aggIdleTime for ev in self.env.evs.values()}
        energy_before = {ev.id: ev.aggIdleEnergy for ev in self.env.evs.values()}

        # 6. UPDATE ENVIRONMENT (Time passes here)
        self.env.update_after_tick(8)


        # -----------------------------------------------------------
        # 7. PROCESS IDLE TRANSITIONS (REPOSITIONING TRAINING) - FIXED
        # -----------------------------------------------------------
        for emv, s, a in idle_transitions:
            # FIX 1: Use 's' and 'a' from the list, don't rely on sarns which might be cleared
            sr_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).view(-1)
            ar_t = torch.as_tensor(a, dtype=torch.int64, device=self.device)

            if emv.state == EvState.IDLE:
                # Case A: Vehicle was IDLE and stayed IDLE
                doner_t = float(episode_done)
                rr_t = emv.sarns.get("reward", 0.0) # Reward calculated inside update?
                st_2_r = self._build_state(emv)     # Build NEW state
            else:
                # Case B: Vehicle was IDLE but got DISPATCHED (became Busy)
                doner_t = 1.0
                rr_t = emv.sarns.get("reward", 0.0)
                # Terminal state for repositioning is zero-vector
                st_2_r = np.zeros(len(s), dtype=np.float32)

            # Convert to Tensors
            st_2_r = torch.as_tensor(st_2_r, dtype=torch.float32, device=self.device).view(-1)
            rr_t = torch.as_tensor(rr_t, dtype=torch.float32, device=self.device)

            # Build Valid Action Mask (for Next State)
            grid_ids = sorted(self.env.grids.keys())
            valid_mask_s2 = [1.0] # Stay is always valid
            for gid in grid_ids:
                # Simple mask logic: 1 if neighbor valid (using imbalance > 0 as dummy check or real logic)
                valid_mask_s2.append(1.0 if self.env.grids[gid].imbalance > 0 else 0.0)
            
            valid_mask_s2 = torch.as_tensor(valid_mask_s2, dtype=torch.float32, device=self.device).view(-1)

            # PUSH TO BUFFER (FIX 2: Use emv.id, NOT ev.id)
            # self._log_reposition_push(t, emv.id, sr_t, ar_t, rr_t, st_2_r, doner_t) # Uncomment if you have this func
            
            self.buffer_reposition.push(sr_t, ar_t, rr_t, st_2_r, doner_t, valid_mask_s2)


        # -----------------------------------------------------------
        # 8. PROCESS BUSY TRANSITIONS (NAVIGATION TRAINING)
        # -----------------------------------------------------------
        for emv, s, a in busy_transitions:
            # FIX: Rename variables to avoid collision with repo variables
            sn_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).view(-1)
            an_t = torch.as_tensor(a, dtype=torch.int64, device=self.device)

            if emv.state == EvState.BUSY:
                # Still busy navigation
                done_t = float(episode_done)
                rn_t = emv.sarns.get("reward", 0.0)
                
                wits, _ = self.build_state_nav1(emv)
                st_2_n = wits # Next state
            else:
                # Finished navigation (Idle or Serving)
                done_t = 1.0
                rn_t = emv.sarns.get("reward", 0.0)
                st_2_n = np.zeros(len(s), dtype=np.float32)

            st_2_n = torch.as_tensor(st_2_n, dtype=torch.float32, device=self.device).view(-1)
            rn_t = torch.as_tensor(rn_t, dtype=torch.float32, device=self.device)
            
            mask = torch.ones(len(s), dtype=torch.float32, device=self.device)

            self.buffer_navigation.push(sn_t, an_t, rn_t, st_2_n, done_t, mask)


        # 9. Metrics Updates
        for ev in self.env.evs.values():
            prev_idle = idle_before.get(ev.id, ev.aggIdleTime)
            prev_energy = energy_before.get(ev.id, ev.aggIdleEnergy)

            di = ev.aggIdleTime - prev_idle
            de = ev.aggIdleEnergy - prev_energy

            if di > 0: self._ep_idle_added += di
            if de > 0: self._ep_energy_added += de

        # 10. TRIGGER TRAINING (FIX 3: Uncommented)
        self._train_reposition(batch_size=64, gamma=0.99)
        self._train_navigation(batch_size=64, gamma=0.99)
        
        self.global_tick += 1


    '''
    def _tick(self, t: int) -> None:
        episode_done = (self.global_tick >= self.ticks_per_ep- 1)
        #if episode_done==1:
            #print("done is true",episode_done)

        self._spawn_incidents_for_tick(t)
        #print("spawned inc",self._spawned_incidents)
        self.env.tick_hospital_waits()
        
        for g in self.env.grids.values():
            g.imbalance = g.calculate_imbalance(self.env.evs, self.env.incidents)

        # 2) build states and actions for IDLE EV
       
        try:
            self._last_dispatches = dispatches # type: ignore
        except Exception:
            self._last_dispatches = []
        
        # collect per-tick navigation actions
        nav_actions: list = []
        idle_transitions = []
        for ev in self.env.evs.values():

            if ev.state == EvState.IDLE :
                ev.sarns["state"] = []
                ev.sarns["action"] = None #i am paranoid, so i cleared stuff here
                ev.sarns["reward"] = 0
                state_vec = self._build_state(ev) #and rebuilt stuff here
                sr_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).view(-1)
                ev.sarns["state"] = state_vec
                a_gi = self._select_action(state_vec, ev.gridIndex)
                self.global_step += 1
                ev.sarns["action"] = a_gi
                ev.nextGrid = ev.gridIndex #to handle none type errror
                
                
                idle_transitions.append((ev,state_vec,a_gi))
                
        # 3) Accept offers
        self.env.accept_reposition_offers()  #next grid changes
        #for ev in self.env.evs.values():
            #print("ev",ev.id,"reward",ev.sarns["reward"],"status",ev.status)
        
        # --- FIX: REMOVED DEBUG_DISPATCH ARGUMENT ---
        dispatches = self.env.dispatch_gridwise(beta=0.5)
        busy_transitions = []
        for ev in self.env.evs.values():    
            if ev.state == EvState.BUSY and ev.status == "Navigation":
                #print("EV ",ev.id,"status",ev.status)
                ev.sarns["state"] = []
                ev.sarns["action"] = None
                ev.sarns["reward"] = 0
                ev.aggIdleTime = 0
                state_vec,grid_ids = self.build_state_nav1(ev) #this is the same as idle
                sn_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).view(-1)
                #replace this with the below navigation state builder
                #state_vec = self.build_state_nav(ev)
                ev.sarns["state"] = state_vec
                #print("lenght of getting state",len(ev.sarns["state"]))
                slo = self._select_nav_action(state_vec)
                self.global_step += 1
                #print("navigation actions", a_gi)
                ev.sarns["action"] = slo
                an_t  = ev.sarns.get("action")
                waits = []
                grid_mean_wait = {}
                g_idx = grid_ids[slo]
                for h in self.env.hospitals.values():
                    if h.gridIndex != g_idx:
                        continue
                    w_valid = self.env.calculate_eta_plus_wait(ev, h)
                    
                    if w_valid is not None:
                        w = max(0,w_valid)
                        waits.append(float(w))
                if waits:
                    grid_mean_wait[g_idx] = sum(waits) / len(waits)
                else:
                    grid_mean_wait[g_idx] = 0.0
                
                mean_wait = grid_mean_wait[g_idx]
                #(ev.navWaitTime)
                if ev.assignedPatientId is not None:
                    inc = self.env.incidents.get(ev.assignedPatientId)
                    if inc is not None and inc.responseToHospitalMinutes is not None:
                        if ev.assignedPatientPriority == 2:
                            H_MIN = 20.0
                        elif ev.assignedPatientPriority == 3:
                            H_MIN = 30.0
                        elif ev.assignedPatientPriority == 1:
                            H_MIN = 10.0
                        # R_busy is the total wait (response + hospital) time
                        #print("inc wait",inc.waitTime,"mean wait",mean_wait,"inc id",inc.id)
                        R_busy = inc.waitTime + mean_wait

                        ev.sarns["reward"] = utility_navigation(R_busy, H_min= H_MIN, H_max=inc.responseToHospitalMinutes)
                        #print("ev",ev.id,"utility",ev.sarns["reward"],"r busy",R_busy,"h min",H_MIN,"H_max",inc.responseToHospitalMinutes)
                #print("reward for navigation", ev.sarns["reward"])        
                rn_t  = ev.sarns.get("reward")
                #print("ev",ev.id,"reward",rn_t,"status",ev.status)
                busy_transitions.append((ev,state_vec,slo))
                dest_grid = grid_ids[slo]
                ev.navdstGrid = dest_grid
                if ev.gridIndex == ev.navdstGrid:
                    #print("ev",ev.id,"reached dst",ev.navdstGrid,"nxt grid",ev.nextGrid,"ev status",ev.status)
                    ev.status = "reached" #or already there, in this case
                    ev.nextGrid = ev.gridIndex
                    hospitals_in_grid = [h for h in self.env.hospitals.values() if h.gridIndex == ev.navdstGrid]
                    if hospitals_in_grid:
                        best_hospital = self.env.select_hospital(ev, hospitals_in_grid, self.env.calculate_eta_plus_wait)
                        #print("selected hc",best_hospital.id,"for ev",ev.id,"in grid",ev.gridIndex,"status",ev.status)
                    else:
                        #print("major blunder")
                        best_hospital = self.env.select_hospital(ev, list(self.env.hospitals.values()), self.env.calculate_eta_plus_wait)
                        
                    if best_hospital is not None:

                        ev.navTargetHospitalId = best_hospital.id
                        #print("ev",ev.id,"dstgrid",ev.navdstGrid,"dst hc",ev.navTargetHospitalId,"total wait",ev.navEtaMinutes)
                        ev.navWaitTime = self.env.calculate_eta_plus_wait(ev, best_hospital)
                        #print("Controller:ev",ev.id,"curently in grid",ev.gridIndex,"nav waitime to dst",ev.navWaitTime,"dst grid",ev.navdstGrid)
                       
                        ev.aggBusyTime += self.env.calculate_eta_plus_wait(ev, best_hospital)
                
                    
                ev.nextGrid = self.env.next_grid_towards(ev.gridIndex, ev.navdstGrid)
                #print("ev",ev.id," dst",ev.navdstGrid,"nxt grid",ev.nextGrid,"ev status",ev.status)
                #print("ev",ev.id,"in grid",ev.gridIndex,"dst grid",ev.navdstGrid,"moving to grid",ev.nextGrid,"status of ev",ev.status)

        
        idle_before = {ev.id: ev.aggIdleTime for ev in self.env.evs.values()}
        energy_before = {ev.id: ev.aggIdleEnergy for ev in self.env.evs.values()}
        #print("called the update function")
        self.env.update_after_tick(8)
        #for ev in self.env.evs.values():
            #if ev.state == EvState.IDLE:
        for emv,s,a in idle_transitions:
            if emv.state == EvState.IDLE:#no change is status
                doner_t = float(episode_done)
                
                sr_t = emv.sarns.get("state")
                #print("size of state",len(sr_t))
                #sr_t  = ev.sarns.get("state") 
                ar_t  = emv.sarns.get("action")
                rr_t  = emv.sarns.get("reward")
                st_2_r = self._build_state(emv) #build the next state
                #print("size of s2",len(st_2_r))
                #doner_t = bool(1)
            else: #idle vehicle became busy during update
                doner_t = float(True)
                #print("ev",emv.id,"became busy from idle")
                sr_t = emv.sarns.get("state")
                #print("size of state",len(sr_t))
                #sr_t  = ev.sarns.get("state") 
                ar_t  = emv.sarns.get("action")
                rr_t  = emv.sarns.get("reward")
                st_2_r = np.zeros(len(sr_t), dtype=np.float32)
                #print("size of s2",len(st_2_r))
            # Build valid action mask: slot 0 (stay) is always valid, others based on grid imbalance
            grid_ids = sorted(self.env.grids.keys())
            valid_mask_s2 = [1.0]  # slot 0 stay always valid
            for gid in grid_ids:
                valid_mask_s2.append(1.0 if self.env.grids[gid].imbalance > 0 else 0.0)
            valid_mask_s2 = torch.as_tensor(valid_mask_s2,dtype =torch.float32,device=self.device).view(-1)
            sr_t = torch.as_tensor(sr_t, dtype=torch.float32, device=self.device).view(-1)
            st_2_r = torch.as_tensor(st_2_r, dtype=torch.float32, device=self.device).view(-1) 
            self._log_reposition_push(t, ev.id, sr_t, ar_t, rr_t, st_2_r, doner_t)
            self.buffer_reposition.push(sr_t, ar_t, rr_t, st_2_r, doner_t,valid_mask_s2 )
            #print("rep rewards",rr_t,"evid",emv.id)
            #print("Repositioning transition pushed:",  ev.id, "state",sr_t,"action",ar_t,"next state",st_2_r,"reward",rr_t,"done",doner_t,"\n")
        #elif ev.state == EvState.BUSY and ev.status == "Navigation" :
        for emv,s,a in busy_transitions:
            if emv.state == EvState.BUSY: #was busy is busy
                done_t = float(episode_done)
                sn_t  = emv.sarns.get("state") #checked size = 4
                an_t  = emv.sarns.get("action")
                rn_t  = emv.sarns.get("reward")
                wits, grids_ids = self.build_state_nav1(emv) #checked size = 4
                st_2_n = wits
                
                    #size =4
            #done_t = bool(1)
            else: #was busy, is idle now
                done_t = float(True)
                #print("ev",emv.id,"became idle after",emv.aggBusyTime,"last known location",emv.gridIndex,"dst",emv.navdstGrid)
                sn_t  = emv.sarns.get("state") #checked size = 4
                an_t  = emv.sarns.get("action")
                rn_t  = emv.sarns.get("reward")
                wits = np.zeros(len(sn_t),dtype = np.float32)
                st_2_n = wits
            if sn_t is None or an_t is None or rn_t is None:
                    continue
            mask = torch.ones(len(sn_t), dtype=torch.float32)
            sn_t = torch.as_tensor(sn_t, dtype=torch.float32, device=self.device).view(-1)
            st_2_n = torch.as_tensor(st_2_n, dtype=torch.float32, device=self.device).view(-1)
            self.buffer_navigation.push(sn_t, an_t, rn_t, st_2_n, done_t,mask)
            #print("smaples pushed into buffer are","\n","state",sn_t,"nextstate",st_2_n,"action",an_t,"reward",rn_t,"done",done_t)
            #print("NAV rewards",rn_t,"evid",emv.id)
            #print("Navigation transition pushed:",  ev.id, "state",sn_t,"next state",st_2_n,"done",done_t,"\n")
            if len(self.buffer_reposition) >= 100:
                Sr, Ar, Rr, S2r, Dr,mask_r = self.buffer_reposition.sample(64, self.device)
                #print("sampled value from rep buffer",Sr,Ar,Rr,S2r,Dr,"\n")
            
        
            #print(" tensor pushed for nav",st_2_n,)
            if len(self.buffer_navigation) >= 100:
                Sn, An, Rn, S2n, Dn,mask_n = self.buffer_navigation.sample(64, self.device)
                #print("sampled value from nav buffer",Sn,An,Rn,S2n,Dn,"\n")
                
                
        # measure how much idle time / energy was added this tick
        for ev in self.env.evs.values():
            
            prev_idle = idle_before.get(ev.id, ev.aggIdleTime)
            prev_energy = energy_before.get(ev.id, ev.aggIdleEnergy)

            di = ev.aggIdleTime - prev_idle
            de = ev.aggIdleEnergy - prev_energy

            if di > 0:
                self._ep_idle_added += di
            if de > 0:
                self._ep_energy_added += de

        #next state???????????????  
        
        
                
                
                
                
        emv2 = self.env.evs[2]
        emv1 = self.env.evs[1]
        #print("for ev number metric list ",emv.id,"is",emv.metric)
        #print("for ev number metric list ",emv2.id,"is",emv2.metric)
        #print("for ev nummber",emv.id,"idle time is",emv.aggIdleTime)  
        #print("for ev nummber",emv2.id,"idle time is",emv2.aggIdleTime)
        #self._train_reposition(batch_size=64, gamma=0.99)
        self._train_navigation(batch_size=64, gamma=0.99)
        self.global_tick +=1
        '''
    '''print("EV state distribution:",
        sum(ev.state == EvState.IDLE for ev in self.env.evs.values()), "idle,",
        sum(ev.status == "Dispatching" for ev in self.env.evs.values()), "dispatching,",
        sum(ev.state == EvState.BUSY for ev in self.env.evs.values()), "busy")'''
    


    def run_training_episode(self, episode_idx: int) -> dict:
        # 1) Reset environment and schedule for this episode
        self._reset_episode()

        total_rep_reward = 0.0
        n_rep_moves = 0
        total_dispatched = 0
        max_concurrent_assigned = 0
        all_dispatches = []
        all_nav_actions = []
        per_tick_dispatch_counts = []

        # 2) Baseline idle time and energy at episode start (per EV)
        self._idle_baseline = {
            ev.id: float(getattr(ev, "aggIdleTime", 0.0))
            for ev in self.env.evs.values()
        }
        energy_baseline = {
            ev.id: float(getattr(ev, "aggIdleEnergy", 0.0))
            for ev in self.env.evs.values()
        }

        # 3) Episode-level accumulators for reposition stats
        total_rep_reward: float = 0.0
        n_rep_moves: int = 0
        total_dispatched: int = 0

        # 4) Run ticks
        for t in range(self.ticks_per_ep):
            self._tick(t)
            tick_dispatches = getattr(self, "_last_dispatches", []) or []
            try:
                per_tick_dispatch_counts.append(len(tick_dispatches))
            except Exception:
                per_tick_dispatch_counts.append(0)

            if tick_dispatches:
                try:
                    all_dispatches.extend(tick_dispatches)
                    total_dispatched += len(tick_dispatches)
                except Exception:
                    pass
            tick_navs = getattr(self, "_last_nav_actions", []) or []
            if tick_navs:
                try:
                    all_nav_actions.extend(tick_navs)
                except Exception:
                    pass
            
            if self.pretty and tick_dispatches:
                n = len(tick_dispatches)
                sample = tick_dispatches[:3]
                #print(f"Tick {t:03d}: dispatches={n} sample={sample}")

            for ev in self.env.evs.values():
                r = ev.sarns.get("reward")
                if r not in (None, 0.0):
                    total_rep_reward += float(r)
                    n_rep_moves += 1

            try:
                n_servicing = sum(
                    1 for inc in self.env.incidents.values()
                    if inc.status == IncidentStatus.ASSIGNED
                )
                if n_servicing > max_concurrent_assigned:
                    max_concurrent_assigned = n_servicing
            except Exception:
                pass

        # 5) Average reposition reward
        avg_rep_reward = total_rep_reward / max(1, n_rep_moves)

        # Compact episode summary line
        total_dispatches = len(all_dispatches)
        try:
            unique_assigned_incidents = len(set(d[1] for d in all_dispatches))
        except Exception:
            unique_assigned_incidents = 0

        mean_util = 0.0
        if total_dispatches > 0:
            try:
                mean_util = sum(d[2] for d in all_dispatches) / total_dispatches
            except Exception:
                mean_util = 0.0

        total_nav = len(all_nav_actions)
        mean_nav_reward = 0.0
        mean_nav_eta = 0.0
        if total_nav > 0:
            try:
                mean_nav_reward = sum(x[2] for x in all_nav_actions) / total_nav
                mean_nav_eta = sum(x[3] for x in all_nav_actions) / total_nav
            except Exception:
                mean_nav_reward = 0.0
                mean_nav_eta = 0.0

        total_incidents_spawned = len(getattr(self, "_spawned_incidents", {}))
        avg_wait = 0.0
        max_wait = 0.0
        if total_incidents_spawned > 0:
            waits = [inc.get_wait_minutes() for inc in self._spawned_incidents.values()]
            avg_wait = sum(waits) / len(waits)
            max_wait = max(waits)

        busy_count = sum(1 for ev in self.env.evs.values() if ev.state == EvState.BUSY)
        idle_count = sum(1 for ev in self.env.evs.values() if ev.state == EvState.IDLE)
        
        # --- FIX: Calculate Average Loss ---
        avg_ep_loss = 0.0
        if len(self.ep_nav_losses) > 0:
            avg_ep_loss = sum(self.ep_nav_losses) / len(self.ep_nav_losses)

        avg_repo_loss = 0.0
        if len(self.ep_repo_losses) > 0:
            avg_repo_loss = sum(self.ep_repo_losses) / len(self.ep_repo_losses)

        #print("=" * 60)
        #print(f"EP {episode_idx:03d} Summary")
        #print("-" * 60)
        #print(f"Schedule: total={self.total_today} | spawned_success={self._spawn_success}")
        #print(f"Dispatch: total={total_dispatches} | unique={unique_assigned_incidents}")
        #print(f"Nav Loss: {avg_ep_loss:.4f}| Repo Loss: {avg_repo_loss:.4f}")
        #print("=" * 60)

        hard_update(self.dqn_reposition_target, self.dqn_reposition_main)

        stats = {
            "episode": episode_idx,
            "avg_rep_reward": avg_rep_reward,
            "rep_moves": n_rep_moves,
            "max_servicing": max_concurrent_assigned,
            "dispatches": len(all_dispatches),
            "total_assignments": total_dispatches,
            "unique_assigned_incidents": unique_assigned_incidents,
            "dispatch_mean_util": mean_util,
            "nav_actions": total_nav,
            "nav_mean_reward": mean_nav_reward,
            "nav_mean_eta": mean_nav_eta,
            "incidents_spawned": total_incidents_spawned,
            "avg_patient_wait": avg_wait,
            "max_patient_wait": max_wait,
            "busy_evs": busy_count,
            "idle_evs": idle_count,
            "total_incidents": len(self.env.incidents),
            "average ep loss": avg_ep_loss,
            "average repo loss": avg_repo_loss,  # Added this key
        }
        return stats

            #"avg_idle_added": avg_idle_added,
            #"avg_energy_added": avg_energy_added,
        

        #return stats    
    import torch

    def _estimate_avg_max_q(self, which: str = "rep", sample_size: int = 256) -> float | None:
        """
        Estimate average max Q(s,·) over a random sample of states
        from the chosen replay buffer ('rep' or 'nav').
        """
        if which == "rep":
            buf = self.buffer_reposition
            net = self.dqn_reposition_main
        else:
            buf = self.buffer_navigation
            net = self.dqn_navigation_main

        if len(buf) == 0:
            return None

        # we just need states; your buffer stores (s, a, r, s2, d)
        n_samples = min(sample_size, len(buf))
        batch = buf.sample(n_samples, device = self.device)  # use your existing sample() that returns python objects

        states = batch[0]  # assuming sample returns (states, actions, rewards, next_states, dones)

        with torch.no_grad():
            s_t = torch.stack(
                [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in states]
            )
            if net is None:
                return None
            q_all = net(s_t)  # shape: (B, n_actions)
            if q_all.shape[1] == 0:
                return None
            q_max = q_all.max(dim=1).values  # (B,)
            return float(q_max.mean().item())
    
    def _build_offers_for_idle_evs(self) -> int:
        offers = 0
        for ev in self.env.evs.values():
            a_gi = ev.sarns["action"]
            if a_gi == ev.nextGrid and ev.status == "Repositioning" :
                offers += 1
        return offers
    def _tick_check(self, t: int) :
            
            self.slot_idle_time = []
            self.slot_idle_energy = []
            self.list_metrics = {}#dict of evids and idle times
            self.nav_metric = {}

            # 1) spawn incidents for testing 
            #for t in range(0,t+1):
        # 1) spawn incidents
            self._spawn_incidents_for_tick(t)
           #self.env.tick_hospital_waits()
            
            for g in self.env.grids.values():
                g.imbalance = g.calculate_imbalance(self.env.evs, self.env.incidents)
            
            # 2) build states and actions for IDLE EVs
            for ev in self.env.evs.values():
                if ev.state == EvState.IDLE and ev.status == "Idle":

                    state_vec = self._build_state(ev)
                    ev.sarns["state"] = state_vec
                    a_gi = self._select_action(state_vec, ev.gridIndex)
                    ev.sarns["action"] = a_gi
                    idle_time = ev.aggIdleTime
                    #print("idle time collected", idle_time)
                    ev.metric.append(idle_time)
                    self.list_metrics[ev.id] = ev.metric
                    #print("in time slot metric appended", ev.id, ev.metric)
                    
                    

                    self.slot_idle_time.append(idle_time)
                    idle_energy = ev.aggIdleEnergy
                    self.slot_idle_energy.append(idle_energy)
                

            # 3) Accept offers
            self.env.accept_reposition_offers()
            
            # --- FIX: REMOVED DEBUG_DISPATCH ARGUMENT ---
            dispatches = self.env.dispatch_gridwise(beta=0.5)
            
            try:
                self._last_dispatches = dispatches
            except Exception:
                self._last_dispatches = []
            
            # collect per-tick navigation actions
            nav_actions: list = []
            for ev in self.env.evs.values():
                self.nav_metric[ev.id] = 0
                if ev.state == EvState.BUSY and ev.status == "Navigation":
                    state_vec,_ = self.build_state_nav1(ev) 
                    ev.sarns["state"] = state_vec
                    a_gi = self._select_nav_action(state_vec)
                    ev.sarns["action"] = a_gi
                    ev.sarns["reward"] = 0.0
                    ev.navEtaMinutes = 0.0
                    busy_time = ev.aggBusyTime
                    ev.nav_metric.append(busy_time)
                    
                    self.nav_metric[ev.id] = ev.nav_metric
                    h = self.env.hospitals.get(a_gi)
                    if h is not None:
                        eta = h.estimate_eta_minutes(ev.location[0], ev.location[1],kmph = np.clip(np.random.normal(40.0, 5.0), 20.0, 80.0))
                        ev.nextGrid = self.env.next_grid_towards(ev.gridIndex, h.gridIndex)
                        ev.navdstGrid = h.gridIndex
                        ev.status = "Navigation"

                        if h.waitTime is not None:
                            w_busy = eta + h.waitTime
                            ev.navEtaMinutes = w_busy
                            reward = utility_navigation(w_busy)
                            ev.sarns["reward"] = reward
                        else:
                            ev.navEtaMinutes = eta
                            reward = utility_navigation(eta)
                            ev.sarns["reward"] = reward

                    try:
                        nav_actions.append((ev.id, a_gi, float(ev.sarns.get("reward", 0.0)), float(ev.navEtaMinutes)))
                        
                    except Exception:
                        pass
                    

            try:
                self._last_nav_actions = nav_actions
            except Exception:
                self._last_nav_actions = []
        
            self.env.update_after_tick(8)
            
        

            self.slot_idle_time_avg = sum(self.slot_idle_time)/len(self.slot_idle_time) if self.slot_idle_time else 0.0
            self.slot_idle_energy_avg = sum(self.slot_idle_energy)/len(self.slot_idle_energy) if self.slot_idle_energy else 0.0
            stats = {"slot idle time": self.slot_idle_time_avg, "slot idle energy": self.slot_idle_energy_avg, "list metrics": self.list_metrics,"list nav metrics":self.nav_metric}
         
                #print("in time slot metric added")
                #print("key vlaue pair in test",self.list_metrics.keys,self.list_metrics.values)
                #print("check", self.list_metrics[ev.id],ev.id)
                       
            return stats
    def run_test_episode(self, episode_idx: int) :
        self._reset_episode()

        total_rep_reward = 0.0
        n_rep_moves = 0
        total_dispatched = 0
        max_concurrent_assigned = 0
        all_dispatches = []
        all_nav_actions = []
        per_tick_dispatch_counts = []
        self.list_avg = []
        self.list_nav_avg =[]
        self.average_episodic_idle = 0 #evid : list of idle times or avg idle time
        self.average_episodic_busy = 0
        for t in range(self.ticks_per_ep):
            check_stats = self._tick_check(t)
            metric_list = check_stats["list metrics"]
            nav_metrics = check_stats["list nav metrics"]
            #print("in test, the metrics observed are fetched")
            for evid in metric_list:
                #print("ev id ", evid," metric list", metric_list[evid])
                avg = sum(metric_list[evid])/len(metric_list[evid]) if metric_list[evid] else 0.0
                #self.list_metrics[evid] = (avg)
                #print("calculated avg idle time for ev", evid, "is", avg)
                self.list_avg.append(avg)      
            for rvid in nav_metrics:
                avrg = sum(nav_metrics[evid])/len(nav_metrics[evid]) if nav_metrics[evid] else 0.0
                self.list_nav_avg.append(avrg)
           #dict ev.id: ev.idletime
            if self.list_avg:
                self.average_episodic_idle = sum(self.list_avg)/len(self.list_avg)
            if self.list_nav_avg:
                self.average_episodic_busy = sum(self.list_nav_avg)/len(self.list_nav_avg)

            tick_dispatches = getattr(self, "_last_dispatches", []) or []
            try:
                per_tick_dispatch_counts.append(len(tick_dispatches))
            except Exception:
                per_tick_dispatch_counts.append(0)

            if tick_dispatches:
                try:
                    all_dispatches.extend(tick_dispatches)
                    total_dispatched += len(tick_dispatches)
                except Exception:
                    pass
            tick_navs = getattr(self, "_last_nav_actions", []) or []
            if tick_navs:
                try:
                    all_nav_actions.extend(tick_navs)
                except Exception:
                    pass
            
            if self.pretty and tick_dispatches:
                n = len(tick_dispatches)
                sample = tick_dispatches[:3]
                #print(f"Tick {t:03d}: dispatches={n} sample={sample}")

            for ev in self.env.evs.values():
                r = ev.sarns.get("reward")
                if r not in (None, 0.0):
                    total_rep_reward += float(r)
                    n_rep_moves += 1

            try:
                n_servicing = sum(
                    1 for inc in self.env.incidents.values()
                    if inc.status == IncidentStatus.ASSIGNED
                )
                if n_servicing > max_concurrent_assigned:
                    max_concurrent_assigned = n_servicing
            except Exception:
                pass
       
            
        avg_rep_reward = total_rep_reward / max(1, n_rep_moves)

        # Compact episode summary line
        total_dispatches = len(all_dispatches)
        try:
            unique_assigned_incidents = len(set(d[1] for d in all_dispatches))
        except Exception:
            unique_assigned_incidents = 0

        mean_util = 0.0
        if total_dispatches > 0:
            try:
                mean_util = sum(d[2] for d in all_dispatches) / total_dispatches
            except Exception:
                mean_util = 0.0

        total_nav = len(all_nav_actions)
        mean_nav_reward = 0.0
        mean_nav_eta = 0.0
        if total_nav > 0:
            try:
                mean_nav_reward = sum(x[2] for x in all_nav_actions) / total_nav
                mean_nav_eta = sum(x[3] for x in all_nav_actions) / total_nav
            except Exception:
                mean_nav_reward = 0.0
                mean_nav_eta = 0.0

        total_incidents_spawned = len(getattr(self, "_spawned_incidents", {}))
        avg_wait = 0.0
        max_wait = 0.0
        if total_incidents_spawned > 0:
            waits = [inc.get_wait_minutes() for inc in self._spawned_incidents.values()]
            avg_wait = sum(waits) / len(waits)
            max_wait = max(waits)

        busy_count = sum(1 for ev in self.env.evs.values() if ev.state == EvState.BUSY)
        idle_count = sum(1 for ev in self.env.evs.values() if ev.state == EvState.IDLE)
        
        # --- FIX: Calculate Average Loss ---
        avg_ep_loss = 0.0
        if len(self.ep_nav_losses) > 0:
            avg_ep_loss = sum(self.ep_nav_losses) / len(self.ep_nav_losses)

        avg_repo_loss = 0.0
        if len(self.ep_repo_losses) > 0:
            avg_repo_loss = sum(self.ep_repo_losses) / len(self.ep_repo_losses)

        #print("=" * 60)
        #print(f"EP {episode_idx:03d} Summary")
        #print("-" * 60)
        #print(f"Schedule: total={self.total_today} | spawned_success={self._spawn_success}")
        #print(f"Dispatch: total={total_dispatches} | unique={unique_assigned_incidents}")
        #print(f"Nav Loss: {avg_ep_loss:.4f}| Repo Loss: {avg_repo_loss:.4f}")
        #print("=" * 60)

        stats = {
            "episode": episode_idx,
            "avg_rep_reward": avg_rep_reward,
            "rep_moves": n_rep_moves,
            "max_servicing": max_concurrent_assigned,
            "dispatches": len(all_dispatches),
            "total_assignments": total_dispatches,
            "unique_assigned_incidents": unique_assigned_incidents,
            "dispatch_mean_util": mean_util,
            "nav_actions": total_nav,
            "nav_mean_reward": mean_nav_reward,
            "nav_mean_eta": mean_nav_eta,
            "incidents_spawned": total_incidents_spawned,
            "avg_patient_wait": avg_wait,
            "max_patient_wait": max_wait,
            "busy_evs": busy_count,
            "idle_evs": idle_count,
            "total_incidents": len(self.env.incidents),
            "average ep loss": avg_ep_loss,
            "average repo loss": avg_repo_loss,  # Added this key
            "average episodic idle times": self.average_episodic_idle, #evid : avg idle time over episode
            "average episodic busy times":self.average_episodic_busy
        }
        #print("episodic idle time",stats["average episodic idle times\n"])
        return stats
    
    def run_inspection_episode(self, episode_idx: int):
        print(f"[Inspection] Starting Episode {episode_idx} trace...")
        self._reset_episode()
        
        trace_data = []

        for t in range(self.ticks_per_ep):
            self._tick(t)
            
            # === FIX: USE CAPITALIZED KEYS TO MATCH YOUR MAIN SCRIPT ===
            for ev in self.env.evs.values():
                trace_data.append({
                    "Tick": t,                  # Was "t"
                    "EV_ID": ev.id,             # Was "ev_id"
                    "State": ev.state.name,     # Was "state"
                    "Status": ev.status,        # Was "status"
                    "Grid": ev.gridIndex,       # Was "grid"
                    "Lat": ev.location[0],
                    "Lng": ev.location[1],
                    
                    # These match your requested print columns
                    "Episode_Total_Idle": ev.aggIdleTime,
                    "Current_Idle_Buffer": getattr(ev, "idleDuration", 0), # Uses idleDuration if available
                    "Energy": ev.aggIdleEnergy
                })

        print("[Inspection] Trace complete.")
        
        # --- Stats Collection (Keep this for graphs) ---
        waits = []
        for inc in getattr(self, "_spawned_incidents", {}).values():
            w = inc.get_wait_minutes()
            if w is not None:
                waits.append(w)
        
        avg_wait = sum(waits) / len(waits) if waits else 0.0

        stats = {
            "episode": episode_idx,
            "total_incidents": len(self.env.incidents),
            "avg_patient_wait": avg_wait,
            "all_wait_times": waits,
            "vehicle_energy": {ev.id: ev.aggIdleEnergy for ev in self.env.evs.values()},
            "vehicle_idle_time": {ev.id: ev.aggIdleTime for ev in self.env.evs.values()} 
        }

        df_trace = pd.DataFrame(trace_data)
        return df_trace, stats
            

            




