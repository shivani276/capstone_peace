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