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

env = MAP("Data/grid_config_2d.json")
env.init_evs()
env.init_hospitals("Data/hospitals_latlong.csv")
ctrl = Controller(
    env,
    ticks_per_ep=180,
    csv_path="Data/5Years_SF_calls_latlong.csv"
)

n_episodes = 500
all_stats = []
all_loss = []
for ep in range(1, n_episodes + 1):
    #dispatched = 0
    stats = ctrl.run_training_episode(ep)
    episode_loss = stats["average ep loss"]
    all_loss.append(episode_loss)
    all_stats.append(stats)
    
import matplotlib.pyplot as plt

plt.plot(all_loss)
plt.xlabel("Episode")
plt.ylabel("Average Navigation Loss")
plt.title("Navigation Training Loss Curve")
plt.grid(True)
plt.show()

