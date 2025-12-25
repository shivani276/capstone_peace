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
'''
from MAP_env import MAP
from Controller import Controller

env = MAP("Data/grid_config_2d.json")
env.init_evs()
env.init_hospitals("D:\\Downloads\\hospitals_latlong.csv")
ctrl = Controller(
    env,
    ticks_per_ep=180,
    csv_path="D:\\Downloads\\5Years_SF_calls_latlong.csv"
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
'''

import matplotlib.pyplot as plt
import pandas as pd
from MAP_env import MAP
from Controller import Controller
import torch
# Initialize Environment
env = MAP("Data/grid_config_2d.json")
env.init_evs()

#env.init_hospitals("D:\\Downloads\\hospitals_latlong.csv")
env.init_hospitals("Data/hospitals_latlong.csv")

# Initialize Controller
ctrl = Controller(
    env,
    ticks_per_ep=180,
    #csv_path="D:\\Downloads\\5Years_SF_calls_latlong.csv"
    csv_path="Data/Fire_Department_and_Emergency_Medical_Services_Dispatched_Calls_for_Service_20251208_with_index.csv"
)
#print("initialized evs", ctrl.env)
n_episodes = 10
n_tests = 1
all_stats = []
all_nav_loss = []
all_repo_loss = [] # New list for repositioning
test_idlet =[]
test_idlee = []
average_i_veh = {}


for ep in range(1, n_episodes + 1):
    stats = ctrl.run_training_episode(ep)
    
    # Get both losses
    nav_loss = stats["average ep loss"]
    repo_loss = stats["average repo loss"]
    
    all_nav_loss.append(nav_loss)
    all_repo_loss.append(repo_loss)
    all_stats.append(stats)


    
    # Get both losses
    nav_loss = stats["average ep loss"]
    repo_loss = stats["average repo loss"]
    
    all_nav_loss.append(nav_loss)
    all_repo_loss.append(repo_loss)
    all_stats.append(stats)

# === NEW PLOTTING SECTION ===

trained_nav = ctrl.dqn_navigation_main 
trained_rep = ctrl.dqn_reposition_main 
# after 500 training episodes
if trained_nav is not None and trained_rep is not None:
    torch.save(trained_nav.state_dict(), "Entities/pained_nav.pth")
    torch.save(trained_rep.state_dict(), "Entities/pained_rep.pth")
print("---------TRAINED DQNS ARE SAVED IN ENTITIES----------")
for ep in range(0,n_tests):
    print("Test Slot:", ep+1)
    test_stats = ctrl.run_test_episode(ep)
    #slot_itime = test_stats["slot idle time"]
    #slot_ienergy = test_stats["slot idle energy"]
    #list_metrics = test_stats["list metrics"]
    slot_itime = test_stats["average episodic idle times"]
    print("avergae episodic idle time for ep",ep,"is",slot_itime)
    slot_btime = test_stats["average episodic busy times"]
    print("avergae episodic busy time for ep ",ep, "is",slot_btime)
    #ids = list(slot_itime.keys())
    #avg_vals = list(slot_itime.values())
    #print("idle time episodic", slot_itime)
    #for evid in list_metrics:
        #avg_list_metric = sum(list_metrics[evid])/len(list_metrics[evid])
        #average_i_veh[evid] = avg_list_metric

    
    #test_idlet.append(slot_itime)
    #test_idlee.append(slot_ienergy)
    
'''plt.plot(test_idlet)
plt.ylabel("average idle time")
plt.xlabel("test slots")
plt.title("Idle time over test slots")
plt.grid(True)
plt.show()
plt.plot(test_idlee)
plt.ylabel("average idle energy")   
plt.xlabel("test slots")
plt.title("Idle energy over test slots")
plt.grid(True)
plt.show()'''


# Extract keys and values id: avg_val
#ids = list(average_i_veh.keys())
#avg_vals = list(average_i_veh.values())

#print("ids",type(ids[0]))
#print("avg_vals",avg_vals)




plt.figure(figsize=(10, 8)) # Make the figure taller

# Plot 1: Navigation Loss
plt.subplot(2, 1, 1) # 2 rows, 1 column, plot #1
plt.plot(all_nav_loss, color='blue', label='Nav Loss')
plt.ylabel("Navigation Loss")
plt.title("Training Loss Curves")
plt.grid(True)
plt.legend()

# Plot 2: Repositioning Loss
plt.subplot(2, 1, 2) # 2 rows, 1 column, plot #2
plt.plot(all_repo_loss, color='orange', label='Reposition Loss')
plt.xlabel("Episode")
plt.ylabel("Reposition Loss")
plt.grid(True)
plt.legend()

plt.tight_layout() # Prevents overlap
plt.show()
