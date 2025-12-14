import matplotlib.pyplot as plt
import pandas as pd
from MAP_env import MAP
from Controller import Controller
import torch
env = MAP("Data/grid_config_2d.json")
env.init_evs()

#env.init_hospitals("D:\\Downloads\\hospitals_latlong.csv")
env.init_hospitals("Data/hospitals_latlong.csv")

# Initialize Controller
ctrl = Controller(
    env,
    ticks_per_ep=180,
    #csv_path="D:\\Downloads\\5Years_SF_calls_latlong.csv"
    csv_path="Data/5Years_SF_calls_latlong.csv"
)
# LOAD TRAINED MODELS
ctrl.dqn_navigation_main.load_state_dict(torch.load("Entities/pained_nav.pth"))
ctrl.dqn_navigation_main.eval()

ctrl.dqn_reposition_main.load_state_dict(torch.load("Entities/pained_rep.pth"))
ctrl.dqn_reposition_main.eval()




n_tests = 1
all_stats = []
all_nav_loss = []
all_repo_loss = [] # New list for repositioning
test_idlet =[]
test_idlee = []
average_i_veh = {}

for ep in range(0,n_tests):
    print("Test Slot:", ep+1)
    test_stats = ctrl.run_test_episode(ep)
    #slot_itime = test_stats["slot idle time"]
    #slot_ienergy = test_stats["slot idle energy"]
    #list_metrics = test_stats["list metrics"]
    slot_itime = test_stats["average episodic idle times"]
    ids = list(slot_itime.keys())
    avg_vals = list(slot_itime.values())
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