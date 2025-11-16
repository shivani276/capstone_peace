from MAP_env import MAP
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