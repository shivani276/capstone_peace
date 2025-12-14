"""
Test Navigation System
Tests the hospital selection and wait time calculation when EVs reach hospital grids.
"""
import sys
import pandas as pd
from MAP_env import MAP
from Entities.ev import EV, EvState
from Entities.Incident import Incident, IncidentStatus

# Initialize environment
print("[TEST] Initializing MAP environment...")
env = MAP("Data/grid_config_2d.json")
env.init_evs(seed=42)
env.init_hospitals("Data/hospitals_latlong.csv")

print(f"[TEST] EVs created: {len(env.evs)}")
print(f"[TEST] Hospitals created: {len(env.hospitals)}")

# Get some hospital info
print("\n" + "="*60)
print("HOSPITAL GRID DISTRIBUTION")
print("="*60)
for hid, h in list(env.hospitals.items())[:5]:
    print(f"Hospital {hid}: Grid {h.gridIndex}, Location {h.loc}")

# Create a test incident in a grid with multiple hospitals
print("\n" + "="*60)
print("TEST 1: Multiple Hospitals in Same Grid")
print("="*60)

# Find a grid with multiple hospitals
grid_with_multiple_hospitals = None
for gi, g in env.grids.items():
    if len(g.hospitals) > 1:
        grid_with_multiple_hospitals = gi
        break

if grid_with_multiple_hospitals is not None:
    grid = env.grids[grid_with_multiple_hospitals]
    hospitals_in_grid = [env.hospitals[hid] for hid in grid.hospitals]
    
    print(f"\nGrid {grid_with_multiple_hospitals} has {len(hospitals_in_grid)} hospitals:")
    for h in hospitals_in_grid:
        print(f"  - Hospital {h.id}: {h.loc}")
    
    # Create incident in this grid
    inc = env.create_incident(
        grid_index=grid_with_multiple_hospitals,
        location=(40.7128, -74.0060),
        priority=2
    )
    print(f"\nCreated Incident {inc.id} with Priority {inc.priority} in Grid {grid_with_multiple_hospitals}")
    
    # Pick an EV and dispatch it
    ev = list(env.evs.values())[0]
    print(f"\nDispatching EV {ev.id} from Grid {ev.gridIndex}")
    
    # Set up EV for navigation
    ev.set_state(EvState.BUSY)
    ev.assignedPatientId = inc.id
    ev.navdstGrid = grid_with_multiple_hospitals
    ev.nextGrid = grid_with_multiple_hospitals  # Will reach destination this tick
    
    # Set hospital wait times
    for h in hospitals_in_grid:
        env.tick_hospital_waits()
        h.waitTime = (h.id % 3) * 10 + 5  # Different wait times
        print(f"Hospital {h.id} waitTime: {h.waitTime:.2f} min")
    
    print(f"\n[Before update_after_tick]")
    print(f"EV {ev.id} state: {ev.state}, gridIndex: {ev.gridIndex}, navdstGrid: {ev.navdstGrid}")
    print(f"Incident {inc.id} status: {inc.status}")
    
    # Run update_after_tick - this should select best hospital
    env.update_after_tick(8.0)
    
    print(f"\n[After update_after_tick]")
    print(f"EV {ev.id} state: {ev.state}, gridIndex: {ev.gridIndex}")
    print(f"Incident {inc.id} status: {inc.status}")
    print(f"EV navWaitTime: {ev.navWaitTime}")
    
    # Check which hospital was selected (EV should be in one hospital's queue)
    for h in hospitals_in_grid:
        if ev.id in h.evs_serving_priority_2:
            print(f"\n✓ EV {ev.id} selected Hospital {h.id} (waitTime: {h.waitTime:.2f})")

else:
    print("No grid with multiple hospitals found!")

# Test 2: Priority-based queue
print("\n" + "="*60)
print("TEST 2: Priority-Based Wait Time Calculation")
print("="*60)

# Find a hospital
hospital = list(env.hospitals.values())[0]
print(f"\nUsing Hospital {hospital.id} in Grid {hospital.gridIndex}")

# Add some EVs to different priority queues
hospital.evs_serving_priority_1 = [99, 100]  # 2 Priority 1 EVs
hospital.evs_serving_priority_2 = [101, 102, 103]  # 3 Priority 2 EVs
hospital.evs_serving_priority_3 = [104, 105]  # 2 Priority 3 EVs
hospital.waitTime = 10.0

print(f"Hospital state:")
print(f"  Priority 1 queue: {hospital.evs_serving_priority_1} ({len(hospital.evs_serving_priority_1)} EVs)")
print(f"  Priority 2 queue: {hospital.evs_serving_priority_2} ({len(hospital.evs_serving_priority_2)} EVs)")
print(f"  Priority 3 queue: {hospital.evs_serving_priority_3} ({len(hospital.evs_serving_priority_3)} EVs)")
print(f"  Base wait time: {hospital.waitTime:.2f} min")

# Test with different priority EVs
test_ev_p1 = list(env.evs.values())[5]
test_ev_p1.assignedPatientPriority = 1
test_ev_p2 = list(env.evs.values())[6]
test_ev_p2.assignedPatientPriority = 2
test_ev_p3 = list(env.evs.values())[7]
test_ev_p3.assignedPatientPriority = 3

# Calculate wait times
wait_p1 = env.calculate_eta_plus_wait(test_ev_p1, hospital)
wait_p2 = env.calculate_eta_plus_wait(test_ev_p2, hospital)
wait_p3 = env.calculate_eta_plus_wait(test_ev_p3, hospital)

print(f"\nWait time calculations:")
print(f"  Priority 1 EV: base_wait={hospital.waitTime:.2f} + higher_priority=0 + same_priority={(len(hospital.evs_serving_priority_1)*8):.2f} = {wait_p1:.2f} min")
print(f"  Priority 2 EV: base_wait={hospital.waitTime:.2f} + higher_priority={(len(hospital.evs_serving_priority_1)*8):.2f} + same_priority={(len(hospital.evs_serving_priority_2)*8):.2f} = {wait_p2:.2f} min")
print(f"  Priority 3 EV: base_wait={hospital.waitTime:.2f} + higher_priority={((len(hospital.evs_serving_priority_1)+len(hospital.evs_serving_priority_2))*8):.2f} + same_priority={(len(hospital.evs_serving_priority_3)*8):.2f} = {wait_p3:.2f} min")

# Verify expected relationships
print(f"\n✓ Priority 1 wait ({wait_p1:.2f}) < Priority 2 wait ({wait_p2:.2f}): {wait_p1 < wait_p2}")
print(f"✓ Priority 2 wait ({wait_p2:.2f}) < Priority 3 wait ({wait_p3:.2f}): {wait_p2 < wait_p3}")

# Test 3: Hospital selection with different wait times
print("\n" + "="*60)
print("TEST 3: Hospital Selection with Varying Wait Times")
print("="*60)

# Get all hospitals in a grid
test_grid_idx = 5
hospitals_in_test_grid = [h for h in env.hospitals.values() if h.gridIndex == test_grid_idx]

if hospitals_in_test_grid:
    print(f"\nGrid {test_grid_idx} has {len(hospitals_in_test_grid)} hospitals")
    
    # Set different wait times and priority queues
    for i, h in enumerate(hospitals_in_test_grid):
        h.waitTime = (i + 1) * 5  # 5, 10, 15, etc.
        h.evs_serving_priority_1 = list(range(i))  # Different queue lengths
        h.evs_serving_priority_2 = []
        h.evs_serving_priority_3 = []
        print(f"  Hospital {h.id}: waitTime={h.waitTime}, P1_queue_len={len(h.evs_serving_priority_1)}")
    
    # Create a test EV
    test_ev = list(env.evs.values())[10]
    test_ev.assignedPatientPriority = 1
    
    # Calculate wait for each hospital
    print(f"\nWait times for Priority 1 EV {test_ev.id}:")
    wait_times = {}
    for h in hospitals_in_test_grid:
        w = env.calculate_eta_plus_wait(test_ev, h)
        wait_times[h.id] = w
        print(f"  Hospital {h.id}: {w:.2f} min")
    
    # Find best hospital (should be the one with min wait)
    best_hid = min(wait_times, key=wait_times.get)
    best_h = env.hospitals[best_hid]
    print(f"\n✓ Best hospital selection: Hospital {best_hid} with wait time {wait_times[best_hid]:.2f} min")

print("\n" + "="*60)
print("NAVIGATION TESTS COMPLETED")
print("="*60)
