'''from MAP_env import MAP
from Controller import Controller



# 1) Load grid environment
env = MAP("Data/grid_config_2d.json")

# 2) Create EVs
env.init_evs()

# 3) Create Controller (with your real CSV path)
controller = Controller(
    env,
    ticks_per_ep=180,
    csv_path="Data/5Years_SF_calls_latlong.csv",  # adjust to your actual path
)

# 4) Run the short debug episode (5 ticks)
controller.run_one_episode()
print("Reposition buffer size:", len(controller.buffer_reposition))
'''

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
    csv_path="D:\\Downloads\\5Years_SF_calls_latlong.csv"
    #csv_path="Data/5Years_SF_calls_latlong.csv"
)
n_episodes = 500


# 1. Run the inspection
df_trace = ctrl.run_inspection_episode(episode_idx=10)

# 2. Filter for a specific EV to see its "Life Story"
ev_id_to_watch = 5  # Change this to any EV ID
ev_data = df_trace[df_trace["EV_ID"] == ev_id_to_watch]

# 3. Print the first 20 ticks to see it accumulating
print(f"\n--- Trace for EV {ev_id_to_watch} ---")
print(ev_data[["Tick", "State", "Status", "Current_Idle_Buffer", "Episode_Total_Idle"]].head(20))

# 4. Find where it got dispatched (The Reset Moment)
# Look for rows where State changes from IDLE to BUSY
resets = ev_data[ev_data["State"] == "BUSY"]
if not resets.empty:
    print("\n--- Dispatch Events (Resets) ---")
    print(resets[["Tick", "State", "Current_Idle_Buffer", "Episode_Total_Idle"]].head())


    

