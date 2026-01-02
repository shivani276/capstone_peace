from Controller import Controller
from Entities.ev import EvState
import random

class GameTheoryController(Controller):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # We track live stats here to ensure we have data for the N/T calculation
        # This acts as the "Memory" for the Nash Game.
        self.hospital_stats = {h_id: {'served': 20, 'total_time': 600.0} for h_id in self.env.hospitals.keys()}
        self.redirect_wait_threshold = 30.0 

    def get_hospital_stats(self, h_id):
        """Safely retrieves N (served) and T (total time) for calculation"""
        if h_id not in self.hospital_stats:
            self.hospital_stats[h_id] = {'served': 20, 'total_time': 600.0}
        return self.hospital_stats[h_id]['served'], self.hospital_stats[h_id]['total_time']

    def resolve_destination(self, ev, incident_id):
        """
        --- THE ONLINE NASH LOGIC ---
        Runs ONCE per incident when the EV is ready to go to a hospital.
        """
        print(f"\n[Incident {incident_id}] Running Nash Equilibrium Calculation...")
        
        hospitals = list(self.env.hospitals.values())
        
        # 1. Identify Candidates
        # Option A: The Geographically Nearest (Default)
        h1 = min(hospitals, key=lambda h: h.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0))
        
        # Option B: The Best Alternative (Next Nearest)
        candidates = [h for h in hospitals if h.id != h1.id]
        if not candidates:
            print(f"  -> Only one hospital exists. Going to H{h1.id}.")
            return h1.gridIndex
            
        h2 = min(candidates, key=lambda h: h.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0))

        # 2. Get Current Historical Stats (N and T)
        n1, t1 = self.get_hospital_stats(h1.id)
        n2, t2 = self.get_hospital_stats(h2.id)

        # 3. Calculate Payoffs (Projected Efficiency)
        # Service Time Constant (e.g. 30 mins)
        service_time = 30.0 

        # --- SCENARIO A: Choose H1 ---
        eta_1 = h1.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0)
        # Check if hospital has a queue (waitTime)
        wait_1 = getattr(h1, 'waitTime', 0.0) or 0.0 
        
        trip_cost_1 = eta_1 + wait_1 + service_time
        score_1 = (n1 + 1) / (t1 + trip_cost_1)
        
        # --- SCENARIO B: Choose H2 (Redirect) ---
        eta_2 = h2.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0)
        wait_2 = getattr(h2, 'waitTime', 0.0) or 0.0
        
        trip_cost_2 = eta_2 + wait_2 + service_time
        score_2 = (n2 + 1) / (t2 + trip_cost_2)

        # 4. detailed Print Logs
        print(f"  > Option A (Nearest): H{h1.id} | ETA: {eta_1:.1f}m | Queue: {wait_1:.1f}m | Total Cost: {trip_cost_1:.1f}m")
        print(f"    -> Projected Eff Score: {score_1:.6f}")
        
        print(f"  > Option B (Redirect): H{h2.id} | ETA: {eta_2:.1f}m | Queue: {wait_2:.1f}m | Total Cost: {trip_cost_2:.1f}m")
        print(f"    -> Projected Eff Score: {score_2:.6f}")

        # 5. The Decision
        if score_1 >= score_2:
            print(f"  >>> DECISION: STAY with H{h1.id} (Score {score_1:.6f} >= {score_2:.6f})")
            return h1.gridIndex
        else:
            print(f"  >>> DECISION: REDIRECT to H{h2.id} (Score {score_2:.6f} > {score_1:.6f})")
            return h2.gridIndex

    # --- MANUAL STATUS UPDATER (Physics) ---
    def _manual_status_update(self, time_delta_minutes):
        """Updates EV status (Nav -> Service -> Idle) and Resolves Incidents"""
        for ev in self.env.evs.values():
            
            # 1. Navigation -> Service
            if ev.status == "Navigation":
                if hasattr(ev, 'navEtaMinutes'):
                    ev.navEtaMinutes -= time_delta_minutes
                else:
                    ev.navEtaMinutes = 0

                if ev.navEtaMinutes <= 0:
                    ev.status = "Service"
                    ev.navEtaMinutes = 0
                    ev.remaining_service_time = 30.0 # Fixed Service Time
                    
                    # Snap to location
                    if ev.navdstGrid in self.env.hospitals:
                        h_obj = self.env.hospitals[ev.navdstGrid]
                        if hasattr(h_obj, 'location'): ev.location = h_obj.location
                        elif hasattr(h_obj, 'lat'): ev.location = (h_obj.lat, h_obj.lon)

            # 2. Service -> Idle (Resolved)
            elif ev.status == "Service":
                ev.remaining_service_time -= time_delta_minutes
                
                if ev.remaining_service_time <= 0:
                    # Free the EV
                    ev.status = "Idle"
                    ev.state = EvState.IDLE
                    ev.remaining_service_time = 0
                    
                    # Update Stats & Resolve Incident
                    if hasattr(ev, 'assigned_incident_id') and ev.assigned_incident_id is not None:
                        inc_id = ev.assigned_incident_id
                        if inc_id in self.env.incidents:
                            inc = self.env.incidents[inc_id]
                            inc.status = "RESOLVED"
                            
                            # Update our GT Stats for next time
                            h_id = inc.hospital_id 
                            # If hospital_id wasn't set on incident, we skip stats update
                            if h_id is not None:
                                if h_id not in self.hospital_stats:
                                    self.hospital_stats[h_id] = {'served': 20, 'total_time': 600.0}
                                
                                self.hospital_stats[h_id]['served'] += 1
                                # Approx time calc
                                total_t = getattr(inc, 'travel_time', 15) + 30 
                                self.hospital_stats[h_id]['total_time'] += total_t
                                
                            print(f"  [Status] Incident {inc_id} RESOLVED by EV {ev.id}")
                        
                        ev.assigned_incident_id = None

    def _tick_check(self, t: int) -> dict:
        self.list_metrics = {} 

        # 1. Spawn & Dispatch
        self._spawn_incidents_for_tick(t)
        dispatches = self.env.dispatch_gridwise(beta=0.5)
        
        # Link Incident ID to EV
        if dispatches:
            for ev_id, inc_id in dispatches.items():
                if ev_id in self.env.evs:
                    self.env.evs[ev_id].assigned_incident_id = inc_id

        # 2. RUN NASH LOGIC (Navigation Decisions)
        for ev in self.env.evs.values():
            if ev.state == EvState.BUSY and ev.status == "Navigation":
                
                # Check if we already have a destination. If not (or if we want to re-eval), run logic.
                # Usually dispatch sets 'navdstGrid', but we override it here.
                
                # Only run logic if we haven't locked it in (optional check)
                # For now, we run it every tick the EV is in "Navigation" state 
                # BUT to prevent spamming logs, we should only do it once per trip.
                # A simple way: check if ev.navEtaMinutes is not set or 0, implying start of trip.
                
                is_start_of_trip = (not hasattr(ev, 'navEtaMinutes') or ev.navEtaMinutes == 0.0)
                
                if is_start_of_trip:
                    # Run Nash
                    inc_id = getattr(ev, 'assigned_incident_id', '?')
                    target_grid = self.resolve_destination(ev, inc_id) 
                    
                    # Apply Result
                    h = self.env.hospitals.get(target_grid)
                    if h:
                        eta = h.estimate_eta_minutes(ev.location[0], ev.location[1], 40.0)
                        ev.nextGrid = self.env.next_grid_towards(ev.gridIndex, h.gridIndex)
                        ev.navdstGrid = h.gridIndex
                        ev.navEtaMinutes = eta # Set ETA, which stops us from re-running this block next tick
                        
                        # Set Incident's hospital ID for stats tracking later
                        if inc_id in self.env.incidents:
                            self.env.incidents[inc_id].hospital_id = h.id

                # Boilerplate for logging
                state_vec, _ = self.build_state_nav1(ev) 
                ev.sarns["state"] = state_vec
                ev.sarns["action"] = ev.navdstGrid if ev.navdstGrid else 0
                ev.sarns["reward"] = 0.0

        # 3. Physics Update
        self._manual_status_update(time_delta_minutes=8) 
        self.env.update_after_tick(8)
        
        return self.list_metrics