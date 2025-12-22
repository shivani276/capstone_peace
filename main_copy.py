
import matplotlib.pyplot as plt
import pandas as pd
from MAP_env import MAP
from Controller import Controller
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

n_episodes = 100
all_wait_times = []
stats_history=[]

#stats_list = []
#for ep in range(1, 100):
    #stats = ctrl.run_training_episode(ep)
    #print(stats)

for ep in range(n_episodes):

    df_trace, stats = ctrl.run_inspection_episode(ep)
    #stats = ctrl.run_training_episode(ep)
    stats_history.append(stats)
'''  
# 2. Filter for a specific EV to see its "Life Story"
ev_id_to_watch = 11  # Change this to any EV ID
ev_data = df_trace[df_trace["EV_ID"] == ev_id_to_watch]

# 3. Print the first 20 ticks to see it accumulating
print(f"\n--- Trace for EV {ev_id_to_watch} ---")
print(ev_data[["Tick", "State", "Status","Grid", "Current_Idle_Buffer", "Episode_Total_Idle"]])

# 4. Find where it got dispatched (The Reset Moment)
# Look for rows where State changes from IDLE to BUSY
resets = ev_data[ev_data["State"] == "BUSY"]
if not resets.empty:
    print("\n--- Dispatch Events (Resets) ---")
    print(resets[["Tick", "State", "Current_Idle_Buffer", "Episode_Total_Idle"]].head())

# Check if it EVER became busy in the whole episode
busy_moments = ev_data[ev_data["State"] == "BUSY"]

if busy_moments.empty:
    print(" EV was lazy! It was never dispatched in this entire episode.")
else:
    print("EV finally got to work at these ticks:")
    print(busy_moments.head(50))
'''
def plot_real_workload(all_episode_stats):
    
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

plot_real_workload(stats_history)


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


