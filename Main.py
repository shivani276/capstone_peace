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
stats_list = []

for ep in range(1, 100):
    stats = ctrl.run_training_episode(ep)
    print(stats)


import matplotlib.pyplot as plt

episodes = list(range(1, len(ctrl.q_rep_history) + 1))

plt.figure(figsize=(10, 4))
plt.plot(episodes, ctrl.q_rep_history, label="Avg max Q (reposition)")
plt.plot(episodes, ctrl.q_nav_history, label="Avg max Q (navigation)")
plt.xlabel("Episode")
plt.ylabel("Average max Q(s,·)")
plt.title("Q-value convergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
