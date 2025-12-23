
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

n_episodes = 1
all_wait_times = []
stats_history=[]
#all_incident_data = []
'''
#stats_list = []
#for ep in range(1, 100):
    #stats = ctrl.run_training_episode(ep)
    #print(stats)

for ep in range(n_episodes):

    #df_trace, stats = ctrl.run_inspection_episode(ep)
    stats = ctrl.run_test_episode(ep)
    stats_history.append(stats)
    daily_data = []
    for inc in ctrl._spawned_incidents.values():
        # Get the wait time (filtering out the 0.0s if needed)
        w = inc.get_wait_minutes()
        
        # Get the priority
        p = inc.priority
            
    # Add to master list    
    all_incident_data.extend(daily_data)
 
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



import matplotlib.pyplot as plt
import numpy as np

def plot_real_workload(all_episode_stats):

    vehicle_energy_map = {}
    vehicle_time_map = {}

    for ep_stat in all_episode_stats:
        # Ensure both required keys exist
        if "vehicle_energy" not in ep_stat or "vehicle_idle_time" not in ep_stat:
            print(f"[Warning] Skipping episode {ep_stat.get('episode')} due to missing data.")
            continue

        v_energies = ep_stat["vehicle_energy"]
        v_times = ep_stat["vehicle_idle_time"]

        # Iterate through EVs present in this episode
        # We use list(v_energies.keys()) to ensure we iterate safely over a copy
        for vid in list(v_energies.keys()):
            # Handle potential string vs int IDs uniformly
            label = f"EV_{vid}"
            
            if label not in vehicle_energy_map:
                vehicle_energy_map[label] = []
                vehicle_time_map[label] = []
                
            vehicle_energy_map[label].append(v_energies.get(vid, 0.0))
            # Use .get() with default 0.0 just in case time is missing for a vid that has energy
            vehicle_time_map[label].append(v_times.get(vid, 0.0))

    if not vehicle_energy_map:
        print("No valid vehicle data found to plot.")
        return

    # 2. Sort and Average
    # Robust sorting key for labels like "EV_1", "EV_10", "EV_2"
    def sort_key(x):
        try:
            return int(x.split('_')[1])
        except:
            return x
            
    ev_ids = sorted(vehicle_energy_map.keys(), key=sort_key)
    
    # Calculate averages across all test episodes
    avg_energies = [np.mean(vehicle_energy_map[ev]) for ev in ev_ids]
    avg_times = [np.mean(vehicle_time_map[ev]) for ev in ev_ids]

    # 3. Setup Plotting Coordinates
    x = np.arange(len(ev_ids))  # The label locations
    width = 0.35  # The width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot Energy bars shifted slightly left
    rects1 = ax.bar(x - width/2, avg_energies, width, label='Avg Energy (kWh)', color='#4E79A7', edgecolor='black')
    # Plot Time bars shifted slightly right
    rects2 = ax.bar(x + width/2, avg_times, width, label='Avg Idle Time (Mins)', color='#F28E2B', edgecolor='black')

    # 4. Formatting
    ax.set_title(f'Per-Vehicle Workload: Energy vs. Idle Time\n(Averaged over {len(all_episode_stats)} Test Episodes)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ev_ids, rotation=45)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 5. Helper function to add labels on top of bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


    


def plot_wait_time_stats(all_episode_stats):
    
    # --- 1. PREPARE DATA ---
    episodes = []
    avg_waits = []
    all_raw_waits = [] # Collect every single wait time from all episodes

    for stat in all_episode_stats:
        ep = stat.get("episode")
        avg = stat.get("avg_patient_wait")
        raw = stat.get("all_wait_times", [])
        
        if ep is not None and avg is not None:
            episodes.append(ep)
            avg_waits.append(avg)
            all_raw_waits.extend(raw)
            
    if not episodes:
        print("No wait time data found. Did you update Controller.py?")
        return

    # --- 3. PLOT WAIT TIME DISTRIBUTION (Histogram) ---
    if all_raw_waits:
        plt.figure(figsize=(10, 5))
        plt.hist(all_raw_waits, bins=20, color="#bbf8ae", edgecolor='black', alpha=0.8)
        
        # Add a vertical line for the global average
        global_avg = np.mean(all_raw_waits)
        plt.axvline(global_avg, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {global_avg:.1f} min')
        counts, edges, bars = plt.hist(all_raw_waits, bins=20, color='#98df8a', edgecolor='black', alpha=0.8)
        plt.title(f'Distribution of Patient Wait Times\n(Across {len(episodes)} Episodes)', fontsize=14)
        plt.xlabel('Wait Time (Minutes)', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label bars that actually have patients
                plt.text(
                    bar.get_x() + bar.get_width() / 2, 
                    height + 0.5, 
                    str(int(height)), 
                    ha='center', va='bottom', fontsize=9
                )
        plt.tight_layout()
        plt.show()
        #print(all_raw_waits)
    else:
        print("No individual wait times found for histogram.")

#plot_wait_time_stats(stats_history)
#plot_real_workload(stats_history)



all_incident_data = [] 

# Run 5 episodes to get enough data
# (Make sure 'ctrl' is already initialized in your code before this!)
for ep in range(5):
    
    ctrl.run_test_episode(ep)  # Run the sim
    
    # Extract data from this episode
    daily_data = []
    for inc in ctrl._spawned_incidents.values():
        w = inc.get_wait_minutes()
        p = inc.priority
        # We append a tuple: (wait_time, priority)
        daily_data.append((w, p))
            
    all_incident_data.extend(daily_data)

# DEBUG CHECK: Did we actually get data?
#print(f"[Result] Collected {len(all_incident_data)} simulation incidents.")

if len(all_incident_data) == 0:
    print("\n[CRITICAL ERROR] No incidents were collected!")
    print("Possibilities:")
    print("1. Your simulation logic is not spawning incidents.")
    print("2. 'ctrl._spawned_incidents' is empty.")
    # We stop here to prevent the crash
    exit()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ... (Previous code for loading Real Data) ...

# 1. LOAD REAL DATA (As before)
df = pd.read_csv("Data/Fire_Department_and_Emergency_Medical_Services_Dispatched_Calls_for_Service_20251208.csv")
df['Received DtTm'] = pd.to_datetime(df['Received DtTm'], format="%Y %b %d %I:%M:%S %p", errors='coerce')
df['Hospital DtTm'] = pd.to_datetime(df['Hospital DtTm'], format="%Y %b %d %I:%M:%S %p", errors='coerce')
df = df.dropna(subset=['Received DtTm', 'Hospital DtTm'])
df['calculated_wait'] = (df['Hospital DtTm'] - df['Received DtTm'])
real_waits_list = (df['calculated_wait'].dt.total_seconds() / 60.0).tolist()

# 2. PREPARE SIM DATA (Ensure this variable exists!)
# If 'all_incident_data' was not created in this run, create a dummy one or fail gracefully
if 'all_incident_data' not in locals() or len(all_incident_data) == 0:
    print("\n[ERROR] 'all_incident_data' is EMPTY or undefined.")
    print("Please run the training/inspection loop BEFORE running this plot code.")
    # For testing only, here is dummy data to stop the crash:
    sim_waits_list = [0, 0, 0, 0, 0] 
else:
    sim_waits_list = [w for w, p in all_incident_data]

# 3. SLICE DATA (Safety Check)
N = 10
# Ensure we have at least N items in both lists to avoid index errors
if len(sim_waits_list) < N:
    print(f"\n[WARNING] Not enough simulation data! Wanted {N}, got {len(sim_waits_list)}.")
    N = len(sim_waits_list) # Reduce N to match available data

plot_ids = range(1, N + 1)
plot_real = real_waits_list[:N]
plot_sim = sim_waits_list[:N]

# 4. PLOTTING
if N > 0:
    x = np.arange(len(plot_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, plot_real, width, label='Real Data', color="#e9ed89", edgecolor='black')
    rects2 = ax.bar(x + width/2, plot_sim, width, label='RL Agent', color="#a475dd", edgecolor='black')

    ax.set_xlabel('Incident ID')
    ax.set_ylabel('Wait Time (Minutes)')
    ax.set_title(f'Comparison: First {N} Incidents')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_ids)
    ax.legend()
    
    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()
else:
    print("Cannot plot: No data available to compare.")








