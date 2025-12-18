# File: GT_Controller.py

from Controller import Controller  # Import your ORIGINAL class
from Entities.ev import EvState
import random

class GameTheoryController(Controller):
    def __init__(self, env, **kwargs):
        # Initialize the original controller
        super().__init__(env, **kwargs)
        
        # Add new GT-specific variables
        self.hospital_strategies = {h_id: 'A' for h_id in self.env.hospitals.keys()}
        self.hospital_stats = {h_id: {'served': 0, 'total_time': 0.0} for h_id in self.env.hospitals.keys()}
        self.redirect_wait_threshold = 30.0 

    def set_strategies(self, new_strategies: dict):
        self.hospital_strategies = new_strategies.copy()
        self.hospital_stats = {h_id: {'served': 0, 'total_time': 0.0} for h_id in self.env.hospitals.keys()}

    def resolve_destination(self, ev):
        """GT Logic: Choose destination based on Strategies (A/R)"""
        hospitals = list(self.env.hospitals.values())
        
        # 1. Default: Nearest Hospital
        nearest_h = min(hospitals, key=lambda h: h.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0))
        target_h = nearest_h

        # 2. Check Strategy
        strategy = self.hospital_strategies.get(nearest_h.id, 'A')

        if strategy == 'R':
            # If Redirecting and busy, go to next nearest
            current_wait = getattr(nearest_h, 'waitTime', 0.0) or 0.0
            if current_wait > self.redirect_wait_threshold:
                candidates = [h for h in hospitals if h.id != nearest_h.id]
                if candidates:
                    target_h = min(candidates, key=lambda h: h.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0))

        return target_h.gridIndex

    # --- THE OVERRIDE ---
    # We copy-paste your _tick_check here and change ONLY the navigation part.
    def _tick_check(self, t: int) -> dict:
        self.slot_idle_time = []
        self.slot_idle_energy = []
        self.list_metrics = {} 

        # 1) Spawn incidents
        self._spawn_incidents_for_tick(t)
        
        # Update imbalance
        for g in self.env.grids.values():
            g.imbalance = g.calculate_imbalance(self.env.evs, self.env.incidents)
        
        # 2) IDLE EVs (Keep original logic)
        for ev in self.env.evs.values():
            if ev.state == EvState.IDLE and ev.status == "Idle":
                state_vec = self._build_state(ev)
                ev.sarns["state"] = state_vec
                a_gi = self._select_action(state_vec, ev.gridIndex)
                ev.sarns["action"] = a_gi
                
                idle_time = ev.aggIdleTime
                ev.metric.append(idle_time)
                self.list_metrics[ev.id] = ev.metric
                self.slot_idle_time.append(idle_time)
                self.slot_idle_energy.append(ev.aggIdleEnergy)
            
        # 3) Accept offers
        self.env.accept_reposition_offers()
        dispatches = self.env.dispatch_gridwise(beta=0.5)
        self._last_dispatches = dispatches if dispatches else []
        
        # 4) NAVIGATION (This is the ONLY changed section)
        nav_actions: list = []
        for ev in self.env.evs.values():
            if ev.state == EvState.BUSY and ev.status == "Navigation":
                state_vec, _ = self.build_state_nav1(ev) 
                ev.sarns["state"] = state_vec
                
                # --- CHANGED: Use GT logic instead of DQN ---
                a_gi = self.resolve_destination(ev) 
                # --------------------------------------------

                ev.sarns["action"] = a_gi
                ev.sarns["reward"] = 0.0
                ev.navEtaMinutes = 0.0

                h = self.env.hospitals.get(a_gi)
                if h is not None:
                    eta = h.estimate_eta_minutes(ev.location[0], ev.location[1], kmph=40.0)
                    ev.nextGrid = self.env.next_grid_towards(ev.gridIndex, h.gridIndex)
                    ev.navdstGrid = h.gridIndex
                    ev.status = "Navigation"

                    if h.waitTime is not None:
                        w_busy = eta + h.waitTime
                        ev.navEtaMinutes = w_busy
                    else:
                        ev.navEtaMinutes = eta

                try:
                    nav_actions.append((ev.id, a_gi, 0.0, float(ev.navEtaMinutes)))
                except Exception:
                    pass

        self._last_nav_actions = nav_actions if nav_actions else []
        
        # 5) Update Physics
        self.env.update_after_tick(8)
        
        self.slot_idle_time_avg = sum(self.slot_idle_time)/len(self.slot_idle_time) if self.slot_idle_time else 0.0
        self.slot_idle_energy_avg = sum(self.slot_idle_energy)/len(self.slot_idle_energy) if self.slot_idle_energy else 0.0
        
        return self.list_metrics

    # Helper to calculate scores at the end of episodes
    def update_hospital_stats(self):
        for inc in self.env.incidents.values():
             # Assuming status 3 is RESOLVED/DONE
            if inc.status == 3 or inc.status == "RESOLVED":
                h_id = inc.hospital_id 
                # Ensure travel_time/wait_time are valid numbers in your Env
                total_t = inc.travel_time + inc.wait_time + inc.service_time
                
                if h_id in self.hospital_stats:
                    self.hospital_stats[h_id]['served'] += 1
                    self.hospital_stats[h_id]['total_time'] += total_t

        print("--- DEBUGGING STATS ---")
        found_finished = False
        
        for inc in self.env.incidents.values():
            # 1. Print the status of the first few incidents to see what they look like
            # This will tell you if status is an Integer (3, 4) or String ("DONE")
            if inc.id < 5: 
                print(f"Incident {inc.id}: Status={inc.status} (Type: {type(inc.status)})")

            # 2. Add your SPECIFIC status codes here
            # In many MAP envs, Status 3 = Resolved, 4 = Archived.
            # Check your Entities/Incident.py to be sure!
            if str(inc.status) in ["3", "4", "RESOLVED", "DONE", "IncidentStatus.DONE"]:
                finished_count += 1
                # ... (rest of calculation) ...
                found_finished = True
        
        if not found_finished:
            print("WARNING: No finished incidents found! Check the Status ID.")

    def get_final_scores(self):
        scores = {}
        for h_id, data in self.hospital_stats.items():
            if data['total_time'] > 0:
                scores[h_id] = data['served'] / data['total_time']
            else:
                scores[h_id] = 0.0
        return scores