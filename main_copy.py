import matplotlib.pyplot as plt
import pandas as pd
from MAP_env import MAP
from Controller import Controller

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

n_episodes = 50

#stats_list = []

#for ep in range(1, 100):
    #stats = ctrl.run_training_episode(ep)
    #print(stats)

# 1. Run the inspection
df_trace = ctrl.run_inspection_episode(episode_idx=10)

# 2. Filter for a specific EV to see its "Life Story"
ev_id_to_watch = 9  # Change this to any EV ID
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
