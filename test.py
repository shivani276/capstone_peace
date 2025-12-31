'''import matplotlib.pyplot as plt
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
    '''
import pandas as pd

filePath = "Data/Fire_Department_and_Emergency_Medical_Services_Dispatched_Calls_for_Service_20251208_with_index.csv"
df = pd.read_csv(filePath)

fmt = "%Y %b %d %I:%M:%S %p"

resp = pd.to_datetime(df["Response DtTm"], format=fmt, errors="coerce")
scene = pd.to_datetime(df["On Scene DtTm"], format=fmt, errors="coerce")
tran = pd.to_datetime(df["Transport DtTm"], format=fmt, errors="coerce")
hosp = pd.to_datetime(df["Hospital DtTm"], format=fmt, errors="coerce")

# Response time: Hospital - Response
responseTime = (hosp - resp).dt.total_seconds()

# Scene time: Transport - On Scene
sceneTime = (tran - scene).dt.total_seconds()

# Subtract scene time
df["adjusted_sec"] = responseTime - sceneTime

# Fill missing priority as 1
df["Final Priority"] = df["Final Priority"].fillna(1).astype(int)

# Keep valid rows for priority 2 and 3
use = df[
    df["Final Priority"].isin([2, 3]) &
    df["adjusted_sec"].notna() &
    (df["adjusted_sec"] > 0) &
    responseTime.notna() &
    (responseTime > 0) &
    sceneTime.notna() &
    (sceneTime >= 0)
]

meanP2 = use.loc[use["Final Priority"] == 2, "adjusted_sec"].mean()
meanP3 = use.loc[use["Final Priority"] == 3, "adjusted_sec"].mean()
meanP23 = use["adjusted_sec"].mean()

print(f"Mean adjusted time (Priority 2): {meanP2:.2f} sec ({meanP2/60:.2f} min)")
print(f"Mean adjusted time (Priority 3): {meanP3:.2f} sec ({meanP3/60:.2f} min)")
print(f"Mean adjusted time (Priority 2+3): {meanP23:.2f} sec ({meanP23/60:.2f} min)")
print(f"Rows used: {len(use)}")

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

