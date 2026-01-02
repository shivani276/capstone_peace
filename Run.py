'''
from MAP_env import MAP
from game_controller import GameTheoryController

if __name__ == "__main__":
    print("--- STARTING DYNAMIC NASH SIMULATION ---")
    
    # 1. Setup
    env = MAP(grid_config_path="Data/grid_config_2d.json")
    controller = GameTheoryController(env, 
                                      ticks_per_ep=180, 
                                      test_mode=True,
                                      csv_path="Data/Fire_Department_and_Emergency_Medical_Services_Dispatched_Calls_for_Service_20251208.csv")
    
    total_episodes = 10 # Change to 500 later
    
    for day in range(total_episodes):
        print(f"\n>>> STARTING DAY {day}")
        
        # --- MANUAL RESET (Cleans Env for new day) ---
        env.incidents = {} 
        for ev in env.evs.values():
            ev.status = "Idle"
            ev.remaining_service_time = 0.0
            ev.assigned_incident_id = None
            ev.navEtaMinutes = 0.0
        
        # Run the Episode
        controller.run_test_episode(day)
        
        print(f"<<< DAY {day} COMPLETE")
'''


from Controller import Controller
from Entities.ev import EvState
from MAP_env import MAP
import random
import math

class GameTheoryController(Controller):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # --- PAPER METRICS: MEMORY FOR NASH GAME ---
        # Stores N (served) and T (total time) for Equation 1: Score = N / T_total
        # Initialized with small baseline values to avoid division by zero.
        self.hospital_stats = {h_id: {'served': 10, 'total_time': 300.0} for h_id in self.env.hospitals.keys()}
        
        # Paper parameter: "Service Process" (Time spent at hospital)
        self.SERVICE_TIME_MINUTES = 30.0 
        
        # Paper parameter: "Background Noise"
        # Since ~80% of patients are "walk-ins" (not simulated agents), we need to 
        # occasionally update hospital stats to reflect this hidden load.
        self.background_noise_prob = 0.1

    def get_hospital_stats(self, h_id):
        """Safely retrieves N and T for calculation."""
        if h_id not in self.hospital_stats:
            self.hospital_stats[h_id] = {'served': 10, 'total_time': 300.0}
        return self.hospital_stats[h_id]['served'], self.hospital_stats[h_id]['total_time']

    def get_distance(self, loc1, loc2):
        """Helper: Euclidean distance for Nearest EV dispatch."""
        return math.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)

    def resolve_destination(self, ev, incident_id):
        """
        --- ONLINE NON-COOPERATIVE NASH EQUILIBRIUM ---
        Executed ONCE when an EV is ready to navigate.
        [cite_start]Hospitals act 'selfishly' to maximize their own Efficiency Score[cite: 22].
        """
        print(f"\n[Nash Logic] Incident {incident_id} (EV {ev.id}) Requesting Destination...")
        
        hospitals = list(self.env.hospitals.values())
        
        # 1. Identify Candidates (Nearest & Next Nearest)
        # Sort all hospitals by estimated travel time
        sorted_hospitals = sorted(hospitals, key=lambda h: h.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0))
        
        h_nearest = sorted_hospitals[0]
        # Find the best alternative (competitor) that isn't the nearest one
        candidates = [h for h in sorted_hospitals if h.id != h_nearest.id]
        h_next = candidates[0] if candidates else h_nearest

        # 2. H_nearest Evaluates Strategy: ACCEPT vs REDIRECT
        # [cite_start]Formula: Score = N / T_total [cite: 197]
        
        # A. Current Score (Status Quo)
        n_curr, t_curr = self.get_hospital_stats(h_nearest.id)
        score_current = n_curr / t_curr if t_curr > 0 else 0

        # B. Projected Score (If Accepted)
        eta_1 = h_nearest.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0)
        wait_1 = getattr(h_nearest, 'waitTime', 0.0) or 0.0 # Queue time
        cost_new_patient = eta_1 + wait_1 + self.SERVICE_TIME_MINUTES
        
        # New N is (N+1), New T is (T + cost)
        score_if_accept = (n_curr + 1) / (t_curr + cost_new_patient)

        print(f"  > H{h_nearest.id} (Nearest) Self-Check:")
        print(f"    Current Score: {score_current:.6f} | If Accept: {score_if_accept:.6f}")

        # 3. The Decision (Non-Cooperative)
        selected_hospital = None
        
        # STRICT SELFISH CHECK: Only accept if score improves or stays same.
        if score_if_accept >= score_current:
            print(f"  >>> DECISION: H{h_nearest.id} ACCEPTS (Efficiency Maintained).")
            selected_hospital = h_nearest
        else:
            print(f"  >>> DECISION: H{h_nearest.id} REJECTS (Efficiency Drop). Attempting Redirect to H{h_next.id}...")
            
            # 4. Attempt Redirection
            # H_next will now perform the same selfish check.
            n_next, t_next = self.get_hospital_stats(h_next.id)
            score_next_current = n_next / t_next if t_next > 0 else 0
            
            eta_2 = h_next.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0)
            wait_2 = getattr(h_next, 'waitTime', 0.0) or 0.0
            cost_redirected = eta_2 + wait_2 + self.SERVICE_TIME_MINUTES
            
            score_next_accept = (n_next + 1) / (t_next + cost_redirected)
            
            print(f"    > H{h_next.id} (Alternative) Self-Check:")
            print(f"      Current Score: {score_next_current:.6f} | If Accept: {score_next_accept:.6f}")
            
            if score_next_accept >= score_next_current:
                print(f"  >>> REDIRECTION SUCCESS: H{h_next.id} ACCEPTS.")
                selected_hospital = h_next
            else:
                # [cite_start]"If no hospital accepts... the nearest hospital has to accept" [cite: 95]
                print(f"  >>> REDIRECTION FAILED: H{h_next.id} REJECTS. Forcing H{h_nearest.id}.")
                selected_hospital = h_nearest

        return selected_hospital

    def _tick_check(self, t: int) -> dict:
        self.list_metrics = {} 

        # 1. SPAWN INCIDENTS
        self._spawn_incidents_for_tick(t)
        
        # 2. DISPATCH LOGIC: Assign Nearest IDLE EV
        # Get live incidents that need an ambulance
        pending_incidents = [
            inc for inc in self.env.incidents.values() 
            if getattr(inc, 'status', 'PENDING') == 'PENDING'
        ]
        
        # Get Idle EVs
        idle_evs = [ev for ev in self.env.evs.values() if ev.state == EvState.IDLE]
        
        for inc in pending_incidents:
            if not idle_evs:
                # Optional: Print if you want to know incidents are waiting
                # print(f"  [Wait] Incident {inc.id} waiting for EV...")
                break 
            
            # Find nearest EV (Euclidean Distance)
            best_ev = min(idle_evs, key=lambda ev: self.get_distance(ev.location, inc.location))
            
            # ASSIGN
            print(f"[Dispatch] Assigning EV {best_ev.id} to Incident {inc.id}")
            best_ev.state = EvState.BUSY
            best_ev.status = "Navigation"
            best_ev.assigned_incident_id = inc.id
            
            # Mark incident as assigned
            inc.status = "ASSIGNED"
            
            idle_evs.remove(best_ev) # Remove from pool

        # 3. NAVIGATION & DECISION LOGIC
        for ev in self.env.evs.values():
            if ev.status == "Navigation" and ev.assigned_incident_id is not None:
                
                # Check if we have calculated a destination yet
                has_dest = getattr(ev, 'navdstGrid', None) is not None
                
                if not has_dest:
                    # RUN NASH LOGIC
                    target_hospital = self.resolve_destination(ev, ev.assigned_incident_id)
                    
                    # Apply Physics Targets
                    ev.navdstGrid = target_hospital.gridIndex
                    ev.nextGrid = self.env.next_grid_towards(ev.gridIndex, target_hospital.gridIndex)
                    
                    # Set Estimated Time of Arrival (ETA)
                    eta = target_hospital.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0)
                    ev.navEtaMinutes = eta
                    
                    # Link Incident to Hospital (for Stats Later)
                    if ev.assigned_incident_id in self.env.incidents:
                         self.env.incidents[ev.assigned_incident_id].hospital_id = target_hospital.id

                # Boilerplate for RL State Vector (Kept from your environment)
                state_vec, _ = self.build_state_nav1(ev) 
                ev.sarns["state"] = state_vec
                ev.sarns["action"] = ev.navdstGrid if ev.navdstGrid else 0
                ev.sarns["reward"] = 0.0

        # 4. PHYSICS UPDATE
        self._manual_status_update(time_delta_minutes=8) # Assuming 1 tick = 8 mins
        self.env.update_after_tick(8)
        
        # 5. BACKGROUND NOISE (Simulate walk-ins)
        for h_id in self.hospital_stats:
            if random.random() < self.background_noise_prob:
                self.hospital_stats[h_id]['served'] += 1
                self.hospital_stats[h_id]['total_time'] += 45.0
        
        return self.list_metrics

    def _manual_status_update(self, time_delta_minutes):
        """Updates EV status (Nav -> Service -> Idle)"""
        for ev in self.env.evs.values():
            
            # A. Navigation -> Service (Arrival)
            if ev.status == "Navigation":
                if hasattr(ev, 'navEtaMinutes'):
                    ev.navEtaMinutes -= time_delta_minutes
                else:
                    ev.navEtaMinutes = 0

                if ev.navEtaMinutes <= 0:
                    # ARRIVED AT HOSPITAL
                    
                    # --- ADDED PRINT HERE ---
                    hospital_id = "Unknown"
                    if ev.navdstGrid in self.env.hospitals:
                        hospital_id = self.env.hospitals[ev.navdstGrid].id
                    print(f"  [Status] EV {ev.id} ARRIVED at H{hospital_id}. Starting Service.")
                    
                    ev.status = "Service"
                    ev.remaining_service_time = self.SERVICE_TIME_MINUTES
                    
                    # Snap location to Hospital (Drop off patient)
                    if ev.navdstGrid in self.env.hospitals:
                        h_obj = self.env.hospitals[ev.navdstGrid]
                        if hasattr(h_obj, 'location'): ev.location = list(h_obj.location)
                        elif hasattr(h_obj, 'lat'): ev.location = [h_obj.lat, h_obj.lon]
                    
                    # Clear Navigation Flags
                    ev.navdstGrid = None 

            # B. Service -> Idle (Completion)
            elif ev.status == "Service":
                ev.remaining_service_time -= time_delta_minutes
                
                if ev.remaining_service_time <= 0:
                    # FINISHED
                    
                    # 1. Update Game Theory Stats (Learning Loop)
                    inc_id = ev.assigned_incident_id
                    if inc_id in self.env.incidents:
                        inc = self.env.incidents[inc_id]
                        inc.status = "RESOLVED"
                        
                        h_id = getattr(inc, 'hospital_id', None)
                        if h_id is not None:
                            # Update N and T
                            self.hospital_stats[h_id]['served'] += 1
                            # Approx time added (Wait + Service + Travel proxy)
                            t_added = 15.0 + self.SERVICE_TIME_MINUTES 
                            self.hospital_stats[h_id]['total_time'] += t_added
                            print(f"  [Resolved] Incident {inc_id} at H{h_id}. Stats Updated.")

                    # 2. Reset EV to IDLE
                    # EV stays at hospital location, ready for next nearest dispatch
                    ev.status = "Idle"
                    ev.state = EvState.IDLE
                    ev.remaining_service_time = 0
                    ev.assigned_incident_id = None


if __name__ == "__main__":
    print("--- STARTING DYNAMIC NASH SIMULATION ---")
    
    # 1. Setup Environment
    # Ensure config path points to your actual JSON file
    env = MAP(grid_config_path="Data/grid_config_2d.json")
    
    # Initialize Controller
    controller = GameTheoryController(
        env, 
        ticks_per_ep=180, 
        test_mode=True,
        csv_path="Data/Fire_Department_and_Emergency_Medical_Services_Dispatched_Calls_for_Service_20251208.csv"
    )
    
    total_episodes = 5  # Run 5 days for testing
    
    for day in range(total_episodes):
        print(f"\n>>> STARTING DAY {day}")
        
        # --- MANUAL RESET (Clean State) ---
        env.incidents = {} 
        for ev in env.evs.values():
            ev.status = "Idle"
            ev.state = EvState.IDLE
            ev.remaining_service_time = 0.0
            ev.assigned_incident_id = None
            ev.navEtaMinutes = 0.0
            ev.navdstGrid = None
        
        # Run the Episode
        controller.run_test_episode(day)
        
        print(f"<<< DAY {day} COMPLETE")


