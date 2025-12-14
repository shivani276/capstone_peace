# Controller.py
import random
from typing import Optional, List

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from MAP_env import MAP
from Entities.ev import EvState
from Entities.Incident import Priority, IncidentStatus
from utils.Epsilon import EpsilonScheduler, hard_update, soft_update
from utils.Helpers import (
    build_daily_incident_schedule,
    point_to_grid_index,
    W_MIN, W_MAX, E_MIN, E_MAX,H_MIN, H_MAX,
    utility_navigation, load_calls
)

from DQN import DQNetwork, ReplayBuffer
print("controler loaded")
DIRECTION_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NAV_K = 8

class Controller:
    def __init__(
        self,
        env: MAP,
        ticks_per_ep: int = 180,
        seed: int = 123,
        csv_path: str = "Data/5Years_SF_calls_latlong.csv",
        time_col: str = "Received DtTm",
        lat_col: Optional[str] = None,
        lng_col: Optional[str] = None,
        wkt_col: Optional[str] = "case_location",
        test_mode: bool = False,
        #test_mode: bool = False,
    ):
        self.env = env
        self.test_mode = test_mode
        #print("[DEBUG] hospitals at Controller init:", len(self.env.hospitals))
        self.ticks_per_ep = ticks_per_ep
        self.dqn_rep_test = None
        self.dqn_nav_test = None
        self.rng = random.Random(seed)

        # agent params
        self.global_step = 0
        self.epsilon_scheduler = EpsilonScheduler(
            start=1.0,     
            end=0.1,       
            decay_steps=5000
        )
        self.epsilon = 1.0 
        self.busy_fraction = 0.5

        # Track losses for plotting
        self.ep_nav_losses = [] 
        self.ep_repo_losses = []

        # DQNs 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if getattr(self, 'test_mode', False):
            class DummyBuffer:
                def push(self, *args, **kwargs): return None
                def sample(self, *args, **kwargs): raise RuntimeError("DummyBuffer has no samples")
                def __len__(self): return 0

            self.dqn_reposition_main = None
            self.dqn_reposition_target = None
            self.opt_reposition = None
            self.buffer_reposition = DummyBuffer()
            
            self.dqn_navigation_main = None
            self.dqn_navigation_target = None
            self.opt_navigation = None
            self.buffer_navigation = DummyBuffer()
        else:
            state_dim = 12
            action_dim = 9
            self.dqn_reposition_main = DQNetwork(state_dim, action_dim).to(self.device)
            self.dqn_reposition_target = DQNetwork(state_dim, action_dim).to(self.device)
            self.dqn_reposition_target.load_state_dict(self.dqn_reposition_main.state_dict())
            self.opt_reposition = torch.optim.Adam(self.dqn_reposition_main.parameters(), lr=1e-3)
            self.buffer_reposition = ReplayBuffer(100)
            #state_dim_nav = max(1, len(self.env.hospitals))
            #action_dim_nav = max(1, action_dim_nav)
            #state_dim_nav = max(1, 78)
            #action_dim_nav = 4

            # --- NAV: one feature and one action per hospital grid ---
            self.hc_grids = sorted({h.gridIndex for h in self.env.hospitals.values()})
            nav_action_dim = len(self.hc_grids)

            if nav_action_dim == 0:
                # degenerate case, but keep network alive
                self.hc_grids = [0]
                nav_action_dim = 1

            state_dim_nav = nav_action_dim
            self.nav_step = 0
            self.nav_target_update = 500  
            self.nav_tau = 0.005          
            self.dqn_navigation_main = DQNetwork(state_dim_nav, nav_action_dim).to(self.device)
            self.dqn_navigation_target = DQNetwork(state_dim_nav, nav_action_dim).to(self.device)
            self.dqn_navigation_target.load_state_dict(self.dqn_navigation_main.state_dict())
            self.opt_navigation = torch.optim.Adam(self.dqn_navigation_main.parameters(), lr=1e-3)
            self.buffer_navigation = ReplayBuffer(100)

            hard_update(self.dqn_reposition_target, self.dqn_reposition_main)
            hard_update(self.dqn_navigation_target, self.dqn_navigation_main)

        if not getattr(self, 'test_mode', False):
            #print("[Controller] DQNs initialised.")
            print("  Device:", self.device)
        else:
            print("[Controller] test_mode enabled: skipping heavy DQN initialisation")

        self.df = pd.read_csv(csv_path)
        self.time_col = time_col
        self.lat_col = lat_col
        self.lng_col = lng_col
        self.wkt_col = wkt_col

       


        self._schedule = None
        self._current_day = None

        self.max_idle_minutes = W_MAX
        self.max_idle_energy = E_MAX
        self.max_wait_time_HC = H_MAX

        self._spawn_attempts = 0
        self._spawn_success = 0
        self.pretty = True
        self.debug_dispatch = False

        #CHECK REPOSITIONING
        # Episode-level history of repositioning performance
        self.idle_time_history: list[float] = []
        self.idle_energy_history: list[float] = []

        self._ep_idle_added: float = 0.0
        self._ep_energy_added: float = 0.0
        self._ep_idle_samples: int = 0

        self._ep_idle_baseline: dict[int, float] = {}
        self._ep_energy_baseline: dict[int, float] = {}


        #Q_VALUE Convergence
        self.q_rep_history: list[float] = []
        self.q_nav_history: list[float] = []





    
    def _get_direction_neighbors_for_index(self, index: int) -> list[int]:
        n_rows = len(self.env.lat_edges) - 1
        n_cols = len(self.env.lng_edges) - 1

        cell_row = index // n_cols
        cell_col = index % n_cols

        offset_map = {
            "N":  (1, 0), "NE": (1, 1), "E":  (0, 1), "SE": (-1, 1),
            "S":  (-1, 0), "SW": (-1, -1), "W":  (0, -1), "NW": (1, -1),
        }

        neighbours: list[int] = []
        for dname in DIRECTION_ORDER:
            dr, dc = offset_map[dname]
            n_row = cell_row + dr
            n_col = cell_col + dc

            if 0 <= n_row < n_rows and 0 <= n_col < n_cols:
                neighbours.append(n_row * n_cols + n_col)
            else:
                neighbours.append(-1)
        return neighbours

    def _pad_neighbors(self, nbs: List[int]):
        N = 8
        n = (nbs[:N] if len(nbs) >= N else nbs + [-1] * (N - len(nbs)))
        return n

    #================== STATE BUILDERS =====================#
    
    def _build_state(self, ev) -> list[float]:
        gi = ev.gridIndex
        g = self.env.grids[gi]

        # own imbalance
        imb_self = float(g.calculate_imbalance(self.env.evs, self.env.incidents))

        # neighbour imbalances in fixed direction order
        neigh_indices = self._get_direction_neighbors_for_index(gi)
        neigh_imbs: list[float] = []
        for nb_idx in neigh_indices:
            if nb_idx == -1:
                neigh_imbs.append(0.0)
            else:
                nb_g = self.env.grids[nb_idx]
                nb_imb = float(nb_g.calculate_imbalance(self.env.evs, self.env.incidents))
                neigh_imbs.append(nb_imb)

        vec: list[float] = []
        vec.append(float(gi))
        vec.append(imb_self)
        vec.extend(neigh_imbs)
        vec.append(float(ev.aggIdleTime))
        vec.append(float(ev.aggIdleEnergy))
        return vec
    
    def build_state_nav1(self, ev):
        # 1) Use precomputed hospital grids
        hc_grids = getattr(self, "hc_grids", None)
        if not hc_grids:
            return [], []

        state_vec: list[float] = []
        grid_ids:  list[int]   = list(hc_grids)

        # 2) Pre-compute mean wait time per HC-grid
        grid_mean_wait = {}
        for g_idx in hc_grids:
            waits = []
            for h in self.env.hospitals.values():
                if h.gridIndex != g_idx:
                    continue
                w = getattr(h, "waitTime", None)
                if w is not None:
                    waits.append(float(w))
            if waits:
                grid_mean_wait[g_idx] = sum(waits) / len(waits)
            else:
                grid_mean_wait[g_idx] = 0.0

        # 3) Build feature per HC-grid: eta(ev → grid) + mean_wait(grid)
        ev_lat, ev_lng = ev.location

        for g_idx in hc_grids:

            hs_in_grid = [h for h in self.env.hospitals.values()
                          if h.gridIndex == g_idx]
            if not hs_in_grid:
                state_vec.append(0.0)
                continue

            h0 = hs_in_grid[0]

            try:
                eta = float(h0.estimate_eta_minutes(ev_lat, ev_lng))
            except Exception:
                eta = 0.0

            mean_wait = grid_mean_wait[g_idx]
            feature = eta + mean_wait

            state_vec.append(feature)


        return state_vec, grid_ids

    '''def build_state_nav(self, ev) -> list[float]:
        gi = ev.gridIndex
        h_list = self.env.hospitals
        vec_n: list[float] = []
        for h in h_list.values():
            if h.gridIndex == gi:
                eta = 0.0
            else:
                eta = h.estimate_eta_minutes(ev.location[0], ev.location[1])
            wait = float(getattr(h, "waitTime", 0.0))
            wg_h = eta + wait
            vec_n.append(wg_h)
        return vec_n

    def build_state_nav1(self, ev):

        gi = ev.gridIndex
        hids = sorted(self.env.hospitals.keys())  # FIXED ORDER
        wgs = []
        for hid in hids:
            h = self.env.hospitals[hid]
            if h.gridIndex == gi:
                eta = 0.0
            else:
                eta = h.estimate_eta_minutes(ev.location[0], ev.location[1])
            wait = float(getattr(h, "waitTime", 0.0))
            wg = eta + wait
            wgs.append(wg)

        # NORMALIZE
        #max_wg = max(wgs) if wgs else 1.0
        #vec_n = [wg / max_wg for wg in wgs]

            return wgs'''



      
   

    #========================= ACTION =================================#

    def _select_action(self, state_vec: list[float], gi: int) -> int:
        if getattr(self, "test_mode", False):
            neighbours = self._get_direction_neighbors_for_index(gi)
            valid = [gi] + [nb for nb in neighbours if nb != -1]
            return self.rng.choice(valid) if valid else gi

        if self.dqn_reposition_main is None:
            neighbours = self._get_direction_neighbors_for_index(gi)
            valid = [gi] + [nb for nb in neighbours if nb != -1]
            return self.rng.choice(valid) if valid else gi

        neighbours = self._get_direction_neighbors_for_index(gi)  # len 8
        valid_mask = [1]  # slot 0 (stay)
        for nb_idx in neighbours:
            valid_mask.append(1 if nb_idx != -1 else 0)

        if self.rng.random() < self.epsilon:
            valid_slots = [i for i, m in enumerate(valid_mask) if m == 1]
            slot = self.rng.choice(valid_slots) if valid_slots else 0
        else:
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.dqn_reposition_main(s).detach().cpu().numpy().ravel()
            for i, m in enumerate(valid_mask):
                if m == 0:
                    q[i] = -1e9
            slot = int(np.argmax(q))

        if slot == 0:
            return gi  # stay
        else:
            dir_index = slot - 1
            nb_idx = neighbours[dir_index]
            return nb_idx if nb_idx != -1 else gi

    '''def _select_nav_action(self, state_vec) -> int:
        """
        Epsilon greedy over all hospitals.

        Returns:
            hospital id (not slot)
        """
        n_actions = len(state_vec)
        hids = sorted(self.env.hospitals.keys()) 

        if n_actions == 0:
            return -1

        if getattr(self, "test_mode", False):
            hid = random.choice(hids)
            hid = max(0, min(hid, n_actions - 1))
            return hid

        if self.rng.random() < self.epsilon:
            hid = random.choice(hids)
            hid = max(0, min(hid, n_actions - 1))
            return hid

        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        s = s.flatten().unsqueeze(0)
        q = self.dqn_navigation_main(s).detach().cpu().numpy().ravel()
        
        hid = int(np.argmax(q))
        hid = max(0, min(hid, n_actions - 1))
        return hid'''
    
    def _select_nav_action(self, state_vec: list[float]) -> int:


        n_actions = len(state_vec)
        if n_actions == 0:
            return -1  # no hospital grids to choose from

        # 1) Exploration: random slot
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, n_actions - 1)

        # 2) Exploitation: DQN greedy
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        # shape: (1, n_actions)
        if self.dqn_navigation_main is not None:
            q = self.dqn_navigation_main(s).detach().cpu().numpy().ravel()
        # q[i] is the Q-value for choosing slot i (i.e. grid_ids[i])

        slot = int(np.argmax(q))
        # safety clamp, just in case
        if slot < 0:
            slot = 0
        elif slot >= n_actions:
            slot = n_actions - 1

        return slot


    #================== REPOSITION TRAIN ======================#

    def _train_reposition(self, batch_size: int = 64, gamma: float = 0.99) -> None:
        if len(self.buffer_reposition) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer_reposition.sample(
            batch_size,
            device=self.device
        )
        if self.dqn_reposition_main is not None:
            q_values = self.dqn_reposition_main(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.dqn_reposition_target is not None:
                q_next = self.dqn_reposition_target(next_states).max(1)[0]
            target = rewards + gamma * (1.0 - dones) * q_next

        loss = F.smooth_l1_loss(q_sa, target)

        if self.opt_reposition is not None:
            self.opt_reposition.zero_grad()
            loss.backward()
            self.opt_reposition.step()

        self.ep_repo_losses.append(loss.item())

        tau = 0.005
        if self.dqn_reposition_target is not None and self.dqn_reposition_main is not None:
            for t_param, o_param in zip(self.dqn_reposition_target.parameters(),
                                        self.dqn_reposition_main.parameters()):
                t_param.data.mul_(1.0 - tau).add_(tau * o_param.data)

    #===================== NAVIGATION TRAIN ==================#
    
    def _train_navigation(self, batch_size: int = 64, gamma: float = 0.99):
        if len(self.buffer_navigation) < batch_size:
            return
        
        try:
            s, a, r, s2, done = self.buffer_navigation.sample(batch_size, device=self.device)
        except TypeError:
            batch = self.buffer_navigation.sample(batch_size, device=self.device)
            s   = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[0]])
            a   = torch.as_tensor(batch[1], dtype=torch.long,   device=self.device)
            r   = torch.as_tensor(batch[2], dtype=torch.float32, device=self.device)
            s2  = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[3]])
            done= torch.as_tensor(batch[4], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.dqn_navigation_target is not None:
                q2 = self.dqn_navigation_target(s2).max(dim=1).values
            y  = r + gamma * (1.0 - done) * q2

        if self.dqn_navigation_main is not None:
            q = self.dqn_navigation_main(s).gather(1, a.view(-1, 1)).squeeze(1)

        loss = torch.nn.functional.smooth_l1_loss(q, y)
        if self.opt_navigation is not None:
            self.opt_navigation.zero_grad()
            loss.backward()
            self.opt_navigation.step()
        
        # --- FIX: TRACK LOSS FOR PLOTTING ---
        self.ep_nav_losses.append(loss.item())

        self.nav_step += 1
        if self.nav_step % self.nav_target_update == 0:
            if self.dqn_navigation_target is not None and self.dqn_navigation_main is not None:
                with torch.no_grad():
                    for p_t, p in zip(self.dqn_navigation_target.parameters(),
                                      self.dqn_navigation_main.parameters()):
                        p_t.data.mul_(1.0 - self.nav_tau).add_(self.nav_tau * p.data)

        if self.nav_step % 500 == 0:
            print(f"[Controller] NAV train step={self.nav_step} loss={loss.item():.4f}")

    # ---------- episode reset ----------
    def _reset_episode(self) -> None:
        self._spawn_attempts = 0
        self._spawn_success = 0
        self.ep_nav_losses = [] # Reset loss tracking
        self.ep_repo_losses = [] # Reset reposition loss tracking

        self.env.incidents.clear()
        for g in self.env.grids.values():
            g.incidents.clear()

        series = pd.to_datetime(
            self.df[self.time_col],
            format="%Y %b %d %I:%M:%S %p",
            errors="coerce"
            ).dt.normalize().dropna()

        days = series.unique()
        if len(days) == 0:
            raise RuntimeError(f"No valid dates in dataset for {self.time_col}")

        self._current_day = pd.Timestamp(self.rng.choice(list(days)))

        self._schedule = build_daily_incident_schedule(
            self.df,
            day=self._current_day,
            time_col=self.time_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            wkt_col=self.wkt_col,
        )

        total_today = 0 if not self._schedule else sum(len(v) for v in self._schedule.values())
        self.total_today = total_today
        #print(f"[Controller] _reset_episode ready: day={self._current_day.date()} incidents_today={total_today}")
        
        self._spawned_incidents = {}
        self._last_dispatches = []

        ev_list = list(self.env.evs.values())
        self.rng.shuffle(ev_list)
        all_idx = list(self.env.grids.keys())
        n_evs = len(ev_list)
        n_busy_target = int(self.busy_fraction * n_evs)

        for i, ev in enumerate(ev_list):
            gi = self.rng.choice(all_idx)
            self.env.move_ev_to_grid(ev.id, gi)

            if i < n_busy_target:
                ev.set_state(EvState.BUSY)
                ev.status = "Navigation"
                ev.nextGrid = None
                ev.navEtaMinutes = self.rng.uniform(0.0, self.max_wait_time_HC)
                ev.aggIdleTime = 0.0
                ev.aggIdleEnergy = 0.0
            else:
                ev.set_state(EvState.IDLE)
                ev.status = "Idle"
                ev.nextGrid = None
                ev.aggIdleTime = self.rng.uniform(0.0, self.max_idle_minutes)
                ev.aggIdleEnergy = self.rng.uniform(0.0, self.max_idle_energy)
                ev.navTargetHospitalId = None
                ev.navEtaMinutes = 0.0
                ev.navUtility = 0.0

            ev.sarns.clear()
            ev.sarns["state"] = None
            ev.sarns["action"] = None
            ev.sarns["reward"] = 0.0
            ev.sarns["next_state"] = None
            #print("ev initiated",ev.id,ev.add_idle,ev.aggBusyTime,ev.state,ev.status)
        #print("the lsit",self.env.grids)

    # ---------- per-tick ----------
    def _spawn_incidents_for_tick(self, t: int) -> None:
        todays_at_tick = self._schedule.get(t, []) if self._schedule else []
        for (lat, lng) in todays_at_tick:
            self._spawn_attempts +=1
            gi = point_to_grid_index(lat, lng, self.env.lat_edges, self.env.lng_edges)
            if gi is None or gi < 0:
                continue
            inc = self.env.create_incident(grid_index=gi, location=(lat, lng), priority="MED")
            try:
                self._spawned_incidents[inc.id] = inc
            except Exception:
                pass
            self._spawn_success +=1

    def _tick(self, t: int) -> None:
        #print("called tick")
        hard_update(self.dqn_reposition_target, self.dqn_reposition_main)

        # 1) spawn incidents
        self._spawn_incidents_for_tick(t)
        self.env.tick_hospital_waits()
        
        for g in self.env.grids.values():
            g.imbalance = g.calculate_imbalance(self.env.evs, self.env.incidents)

        # 2) build states and actions for IDLE EVs
        for ev in self.env.evs.values():
            if ev.state == EvState.IDLE and ev.status == "Idle":
                state_vec = self._build_state(ev)
                ev.sarns["state"] = state_vec
                a_gi = self._select_action(state_vec, ev.gridIndex)
                ev.sarns["action"] = a_gi

        # 3) Accept offers
        self.env.accept_reposition_offers()
        
        # --- FIX: REMOVED DEBUG_DISPATCH ARGUMENT ---
        dispatches = self.env.dispatch_gridwise(beta=0.5)
        
        try:
            self._last_dispatches = dispatches
        except Exception:
            self._last_dispatches = []
        
        # collect per-tick navigation actions
        nav_actions: list = []
        for ev in self.env.evs.values():
            if ev.state == EvState.BUSY and ev.status == "Navigation":
                state_vec,grid_ids = self.build_state_nav1(ev) #this is the same as idle
                #replace this with the below navigation state builder
                #state_vec = self.build_state_nav(ev)
                ev.sarns["state"] = state_vec
                slo = self._select_nav_action(state_vec)
                #print("navigation actions", a_gi)
                ev.sarns["action"] = slo
                dest_grid = grid_ids[slo]
                candidate_hs = [
                h for h in self.env.hospitals.values()
                if h.gridIndex == dest_grid
                ]
                #h=self.env.hospitals.get(a_gi)
                # 5. Pick a concrete hospital (best = minimum wait time)
                h = min(candidate_hs, key=lambda hh: hh.waitTime or 0.0)

                if h is not None:
                    eta = float(h.estimate_eta_minutes(ev.location[0], ev.location[1]))
                    ev.nextGrid = self.env.next_grid_towards(ev.gridIndex, h.gridIndex)
                    ev.navTargetHospitalId = h.id
                    ev.navdstGrid = h.gridIndex
                    
                    ev.status = "Navigation"

                    if h.waitTime is not None:
                        w_busy = eta + h.waitTime
                        ev.navEtaMinutes = w_busy
                        ev.sarns["reward"] = utility_navigation(w_busy)
                        #print("reward for navigation", ev.sarns["reward"])
                        '''print(
                        f"[NAV-DEBUG] ev={ev.id} "
                        f"slot={slo} dest_grid={dest_grid} "
                        f"navTargetHospitalId={ev.navTargetHospitalId} "
                        f"nextGrid={ev.nextGrid} navdstGrid={ev.navdstGrid} "
                        f"w_busy={w_busy:.2f}"
                        )'''


            # snapshot idle/energy before env update
        
        
        idle_before = {ev.id: ev.aggIdleTime for ev in self.env.evs.values()}
        energy_before = {ev.id: ev.aggIdleEnergy for ev in self.env.evs.values()}
        #print("called the update function")
        self.env.update_after_tick(8)

        # measure how much idle time / energy was added this tick
        for ev in self.env.evs.values():
            prev_idle = idle_before.get(ev.id, ev.aggIdleTime)
            prev_energy = energy_before.get(ev.id, ev.aggIdleEnergy)

            di = ev.aggIdleTime - prev_idle
            de = ev.aggIdleEnergy - prev_energy

            if di > 0:
                self._ep_idle_added += di
            if de > 0:
                self._ep_energy_added += de

        #next state???????????????  
        
        for ev in self.env.evs.values():
            if ev.state == EvState.IDLE or ev.status == "Dispatching":
                #s2 = self._build_state(ev)
                #append this into the push rep trans, remove s2 from there
                #self._push_reposition_transition(ev)

                sr_t  = ev.sarns.get("state")
                ar_t  = ev.sarns.get("action")
                rr_t  = ev.sarns.get("reward")
                st_2_r = self._build_state(ev)
                doner_t = bool(1)
                sr_t = torch.as_tensor(sr_t, dtype=torch.float32, device=self.device).view(-1)
                st_2_r = torch.as_tensor(st_2_r, dtype=torch.float32, device=self.device).view(-1) 
                self.buffer_reposition.push(sr_t, ar_t, rr_t, st_2_r, doner_t)
                
                
                if len(self.buffer_reposition) >= 1000:
                    Sr, Ar, Rr, S2r, Dr = self.buffer_reposition.sample(64, self.device)
                
            elif ev.state == EvState.BUSY and ev.status == "Navigation" :
                s_t  = ev.sarns.get("state")
                a_t  = ev.sarns.get("action")
                r_t  = ev.sarns.get("reward")
                wits,_ = self.build_state_nav1(ev)
                if s_t is None or a_t is None or r_t is None:
                    continue
                st_2_n = wits
                done_t = bool(1)
                s_t = torch.as_tensor(s_t, dtype=torch.float32, device=self.device).view(-1)
                st_2_n = torch.as_tensor(st_2_n, dtype=torch.float32, device=self.device).view(-1)
               
                self.buffer_navigation.push(s_t, a_t, r_t, st_2_n, done_t)
                
                
                if len(self.buffer_navigation) >= 1000:
                    Sn, An, Rn, S2n, Dn = self.buffer_navigation.sample(64, self.device)
        emv = self.env.evs[1]
        emv2 = self.env.evs[2]
        #print("for ev number metric list ",emv.id,"is",emv.metric)
        #print("for ev number metric list ",emv2.id,"is",emv2.metric)
        #print("for ev nummber",emv.id,"idle time is",emv.aggIdleTime)  
        #print("for ev nummber",emv2.id,"idle time is",emv2.aggIdleTime)
        self._train_reposition(batch_size=64, gamma=0.99)
        self._train_navigation(batch_size=64, gamma=0.99)
        
        '''print("EV state distribution:",
        sum(ev.state == EvState.IDLE for ev in self.env.evs.values()), "idle,",
        sum(ev.status == "Dispatching" for ev in self.env.evs.values()), "dispatching,",
        sum(ev.state == EvState.BUSY for ev in self.env.evs.values()), "busy")'''



    def run_training_episode(self, episode_idx: int) -> dict:
        # 1) Reset environment and schedule for this episode
        self._reset_episode()

        total_rep_reward = 0.0
        n_rep_moves = 0
        total_dispatched = 0
        max_concurrent_assigned = 0
        all_dispatches = []
        all_nav_actions = []
        per_tick_dispatch_counts = []

        # 2) Baseline idle time and energy at episode start (per EV)
        self._idle_baseline = {
            ev.id: float(getattr(ev, "aggIdleTime", 0.0))
            for ev in self.env.evs.values()
        }
        energy_baseline = {
            ev.id: float(getattr(ev, "aggIdleEnergy", 0.0))
            for ev in self.env.evs.values()
        }

        # 3) Episode-level accumulators for reposition stats
        total_rep_reward: float = 0.0
        n_rep_moves: int = 0
        total_dispatched: int = 0

        # 4) Run ticks
        for t in range(self.ticks_per_ep):
            self._tick(t)
            tick_dispatches = getattr(self, "_last_dispatches", []) or []
            try:
                per_tick_dispatch_counts.append(len(tick_dispatches))
            except Exception:
                per_tick_dispatch_counts.append(0)

            if tick_dispatches:
                try:
                    all_dispatches.extend(tick_dispatches)
                    total_dispatched += len(tick_dispatches)
                except Exception:
                    pass
            tick_navs = getattr(self, "_last_nav_actions", []) or []
            if tick_navs:
                try:
                    all_nav_actions.extend(tick_navs)
                except Exception:
                    pass
            
            if self.pretty and tick_dispatches:
                n = len(tick_dispatches)
                sample = tick_dispatches[:3]
                #print(f"Tick {t:03d}: dispatches={n} sample={sample}")

            for ev in self.env.evs.values():
                r = ev.sarns.get("reward")
                if r not in (None, 0.0):
                    total_rep_reward += float(r)
                    n_rep_moves += 1

            try:
                n_servicing = sum(
                    1 for inc in self.env.incidents.values()
                    if inc.status == IncidentStatus.ASSIGNED
                )
                if n_servicing > max_concurrent_assigned:
                    max_concurrent_assigned = n_servicing
            except Exception:
                pass

        # 5) Average reposition reward
        avg_rep_reward = total_rep_reward / max(1, n_rep_moves)

        # Compact episode summary line
        total_dispatches = len(all_dispatches)
        try:
            unique_assigned_incidents = len(set(d[1] for d in all_dispatches))
        except Exception:
            unique_assigned_incidents = 0

        mean_util = 0.0
        if total_dispatches > 0:
            try:
                mean_util = sum(d[2] for d in all_dispatches) / total_dispatches
            except Exception:
                mean_util = 0.0

        total_nav = len(all_nav_actions)
        mean_nav_reward = 0.0
        mean_nav_eta = 0.0
        if total_nav > 0:
            try:
                mean_nav_reward = sum(x[2] for x in all_nav_actions) / total_nav
                mean_nav_eta = sum(x[3] for x in all_nav_actions) / total_nav
            except Exception:
                mean_nav_reward = 0.0
                mean_nav_eta = 0.0

        total_incidents_spawned = len(getattr(self, "_spawned_incidents", {}))
        avg_wait = 0.0
        max_wait = 0.0
        if total_incidents_spawned > 0:
            waits = [inc.get_wait_minutes() for inc in self._spawned_incidents.values()]
            avg_wait = sum(waits) / len(waits)
            max_wait = max(waits)

        busy_count = sum(1 for ev in self.env.evs.values() if ev.state == EvState.BUSY)
        idle_count = sum(1 for ev in self.env.evs.values() if ev.state == EvState.IDLE)
        
        # --- FIX: Calculate Average Loss ---
        avg_ep_loss = 0.0
        if len(self.ep_nav_losses) > 0:
            avg_ep_loss = sum(self.ep_nav_losses) / len(self.ep_nav_losses)

        avg_repo_loss = 0.0
        if len(self.ep_repo_losses) > 0:
            avg_repo_loss = sum(self.ep_repo_losses) / len(self.ep_repo_losses)

        #print("=" * 60)
        #print(f"EP {episode_idx:03d} Summary")
        #print("-" * 60)
        #print(f"Schedule: total={self.total_today} | spawned_success={self._spawn_success}")
        #print(f"Dispatch: total={total_dispatches} | unique={unique_assigned_incidents}")
        #print(f"Nav Loss: {avg_ep_loss:.4f}| Repo Loss: {avg_repo_loss:.4f}")
        #print("=" * 60)

        stats = {
            "episode": episode_idx,
            "avg_rep_reward": avg_rep_reward,
            "rep_moves": n_rep_moves,
            "max_servicing": max_concurrent_assigned,
            "dispatches": len(all_dispatches),
            "total_assignments": total_dispatches,
            "unique_assigned_incidents": unique_assigned_incidents,
            "dispatch_mean_util": mean_util,
            "nav_actions": total_nav,
            "nav_mean_reward": mean_nav_reward,
            "nav_mean_eta": mean_nav_eta,
            "incidents_spawned": total_incidents_spawned,
            "avg_patient_wait": avg_wait,
            "max_patient_wait": max_wait,
            "busy_evs": busy_count,
            "idle_evs": idle_count,
            "total_incidents": len(self.env.incidents),
            "average ep loss": avg_ep_loss,
            "average repo loss": avg_repo_loss,  # Added this key
        }
        return stats

            #"avg_idle_added": avg_idle_added,
            #"avg_energy_added": avg_energy_added,
        

        #return stats    
    import torch

    def _estimate_avg_max_q(self, which: str = "rep", sample_size: int = 256) -> float | None:
        """
        Estimate average max Q(s,·) over a random sample of states
        from the chosen replay buffer ('rep' or 'nav').
        """
        if which == "rep":
            buf = self.buffer_reposition
            net = self.dqn_reposition_main
        else:
            buf = self.buffer_navigation
            net = self.dqn_navigation_main

        if len(buf) == 0:
            return None

        # we just need states; your buffer stores (s, a, r, s2, d)
        n_samples = min(sample_size, len(buf))
        batch = buf.sample(n_samples, device = self.device)  # use your existing sample() that returns python objects

        states = batch[0]  # assuming sample returns (states, actions, rewards, next_states, dones)

        with torch.no_grad():
            s_t = torch.stack(
                [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in states]
            )
            if net is None:
                return None
            q_all = net(s_t)  # shape: (B, n_actions)
            if q_all.shape[1] == 0:
                return None
            q_max = q_all.max(dim=1).values  # (B,)
            return float(q_max.mean().item())


    '''def run_one_episode(self) -> None:
        print("[Controller] Resetting episode...")
        

        self._reset_episode()
        


        if self._current_day is not None:
            print(f"[Controller] Day selected: {self._current_day.date()}")
        else:
            print("[Controller] Warning: No day selected (dataset may be empty or invalid).")

        if self._schedule:
            total_incidents = sum(len(v) for v in self._schedule.values())
            print(f"[Controller] Total incidents today: {total_incidents}")
        else:
            print("[Controller] Warning: Schedule not built — no incidents will spawn.")

        # Only run 5 ticks for debugging
        for t in range(180):
            self._tick(t)

        print(f"[Controller] Episode debug run complete. Total incidents created: {len(self.env.incidents)}")
        '''
    '''
    # ---------- run one episode ----------
    def run_one_episode(self) -> None:
        print("[Controller] Resetting episode...")
        self._reset_episode()

        # Safe day print (avoid NoneType)
        if self._current_day is not None:
            print(f"[Controller] Day selected: {self._current_day.date()}")
        else:
            print("[Controller] Warning: No day selected (dataset may be empty or invalid).")

        # Safe schedule summary
        if self._schedule:
            total_incidents = sum(len(v) for v in self._schedule.values())
            print(f"[Controller] Total incidents today: {total_incidents}")
        else:
            print("[Controller] Warning: Schedule not built — no incidents will spawn.")


        # Run all ticks
        for t in range(self.ticks_per_ep):
            self._tick(t)
            if t % 30 == 0:
                print(f"Tick {t:03d}: incidents so far = {len(self.env.incidents)}")

        print(f"[Controller] Episode complete. Total incidents created: {len(self.env.incidents)}")
    '''
    def _build_offers_for_idle_evs(self) -> int:
        offers = 0
        for ev in self.env.evs.values():
            a_gi = ev.sarns["action"]
            if a_gi == ev.nextGrid and ev.status == "Repositioning" :
                offers += 1
        return offers
    def _tick_check(self, t: int) -> None:
            
            self.slot_idle_time = []
            self.slot_idle_energy = []
            self.list_metrics = {} #dict of evids and idle times
            

            # 1) spawn incidents for testing 
            self._spawn_incidents_for_tick(t)
           #self.env.tick_hospital_waits()
            
            for g in self.env.grids.values():
                g.imbalance = g.calculate_imbalance(self.env.evs, self.env.incidents)
            
            # 2) build states and actions for IDLE EVs
            for ev in self.env.evs.values():
                if ev.state == EvState.IDLE and ev.status == "Idle":

                    state_vec = self._build_state(ev)
                    ev.sarns["state"] = state_vec
                    a_gi = self._select_action(state_vec, ev.gridIndex)
                    ev.sarns["action"] = a_gi
                    idle_time = ev.aggIdleTime
                    #print("idle time collected", idle_time)
                    ev.metric.append(idle_time)
                    self.list_metrics[ev.id] = ev.metric
                    #print("in time slot metric appended", ev.id, ev.metric)
                    
                    

                    self.slot_idle_time.append(idle_time)
                    idle_energy = ev.aggIdleEnergy
                    self.slot_idle_energy.append(idle_energy)
                

            # 3) Accept offers
            self.env.accept_reposition_offers()
            
            # --- FIX: REMOVED DEBUG_DISPATCH ARGUMENT ---
            dispatches = self.env.dispatch_gridwise(beta=0.5)
            
            try:
                self._last_dispatches = dispatches
            except Exception:
                self._last_dispatches = []
            
            # collect per-tick navigation actions
            nav_actions: list = []
            for ev in self.env.evs.values():
                if ev.state == EvState.BUSY and ev.status == "Navigation":
                    state_vec = self.build_state_nav1(ev) 
                    ev.sarns["state"] = state_vec
                    a_gi = self._select_nav_action(state_vec)
                    ev.sarns["action"] = a_gi
                    ev.sarns["reward"] = 0.0
                    ev.navEtaMinutes = 0.0

                    h = self.env.hospitals.get(a_gi)
                    if h is not None:
                        eta = h.estimate_eta_minutes(ev.location[0], ev.location[1])
                        ev.nextGrid = self.env.next_grid_towards(ev.gridIndex, h.gridIndex)
                        ev.navdstGrid = h.gridIndex
                        ev.status = "Navigation"

                        if h.waitTime is not None:
                            w_busy = eta + h.waitTime
                            ev.navEtaMinutes = w_busy
                            reward = utility_navigation(w_busy)
                            ev.sarns["reward"] = reward
                        else:
                            ev.navEtaMinutes = eta
                            reward = utility_navigation(eta)
                            ev.sarns["reward"] = reward

                    try:
                        nav_actions.append((ev.id, a_gi, float(ev.sarns.get("reward", 0.0)), float(ev.navEtaMinutes)))
                    except Exception:
                        pass

            try:
                self._last_nav_actions = nav_actions
            except Exception:
                self._last_nav_actions = []
        
            self.env.update_after_tick(8)
            
        

            self.slot_idle_time_avg = sum(self.slot_idle_time)/len(self.slot_idle_time) if self.slot_idle_time else 0.0
            self.slot_idle_energy_avg = sum(self.slot_idle_energy)/len(self.slot_idle_energy) if self.slot_idle_energy else 0.0
            stats = {"slot idle time": self.slot_idle_time_avg, "slot idle energy": self.slot_idle_energy_avg, "list metrics": self.list_metrics}
         
                #print("in time slot metric added")
                #print("key vlaue pair in test",self.list_metrics.keys,self.list_metrics.values)
                #print("check", self.list_metrics[ev.id],ev.id)
            return self.list_metrics
    def run_test_episode(self, episode_idx: int) -> dict:
        self._reset_episode()

        total_rep_reward = 0.0
        n_rep_moves = 0
        total_dispatched = 0
        max_concurrent_assigned = 0
        all_dispatches = []
        all_nav_actions = []
        per_tick_dispatch_counts = []
        self.list_metrics = {} #evid : list of idle times or avg idle time
        for t in range(self.ticks_per_ep):
            metric_list = self._tick_check(t)
            #print("in test, the metrics observed are fetched")
            for evid in metric_list:
                #print("ev id ", evid," metric list", metric_list[evid])
                avg = sum(metric_list[evid])/len(metric_list[evid]) if metric_list[evid] else 0.0
                self.list_metrics[evid] = (avg)
                print("calculated avg idle time for ev", evid, "is", avg)
                         
           #dict ev.id: ev.idletime
            tick_dispatches = getattr(self, "_last_dispatches", []) or []
            try:
                per_tick_dispatch_counts.append(len(tick_dispatches))
            except Exception:
                per_tick_dispatch_counts.append(0)

            if tick_dispatches:
                try:
                    all_dispatches.extend(tick_dispatches)
                    total_dispatched += len(tick_dispatches)
                except Exception:
                    pass
            tick_navs = getattr(self, "_last_nav_actions", []) or []
            if tick_navs:
                try:
                    all_nav_actions.extend(tick_navs)
                except Exception:
                    pass
            
            if self.pretty and tick_dispatches:
                n = len(tick_dispatches)
                sample = tick_dispatches[:3]
                #print(f"Tick {t:03d}: dispatches={n} sample={sample}")

            for ev in self.env.evs.values():
                r = ev.sarns.get("reward")
                if r not in (None, 0.0):
                    total_rep_reward += float(r)
                    n_rep_moves += 1

            try:
                n_servicing = sum(
                    1 for inc in self.env.incidents.values()
                    if inc.status == IncidentStatus.ASSIGNED
                )
                if n_servicing > max_concurrent_assigned:
                    max_concurrent_assigned = n_servicing
            except Exception:
                pass
       
            
        avg_rep_reward = total_rep_reward / max(1, n_rep_moves)

        # Compact episode summary line
        total_dispatches = len(all_dispatches)
        try:
            unique_assigned_incidents = len(set(d[1] for d in all_dispatches))
        except Exception:
            unique_assigned_incidents = 0

        mean_util = 0.0
        if total_dispatches > 0:
            try:
                mean_util = sum(d[2] for d in all_dispatches) / total_dispatches
            except Exception:
                mean_util = 0.0

        total_nav = len(all_nav_actions)
        mean_nav_reward = 0.0
        mean_nav_eta = 0.0
        if total_nav > 0:
            try:
                mean_nav_reward = sum(x[2] for x in all_nav_actions) / total_nav
                mean_nav_eta = sum(x[3] for x in all_nav_actions) / total_nav
            except Exception:
                mean_nav_reward = 0.0
                mean_nav_eta = 0.0

        total_incidents_spawned = len(getattr(self, "_spawned_incidents", {}))
        avg_wait = 0.0
        max_wait = 0.0
        if total_incidents_spawned > 0:
            waits = [inc.get_wait_minutes() for inc in self._spawned_incidents.values()]
            avg_wait = sum(waits) / len(waits)
            max_wait = max(waits)

        busy_count = sum(1 for ev in self.env.evs.values() if ev.state == EvState.BUSY)
        idle_count = sum(1 for ev in self.env.evs.values() if ev.state == EvState.IDLE)
        
        # --- FIX: Calculate Average Loss ---
        avg_ep_loss = 0.0
        if len(self.ep_nav_losses) > 0:
            avg_ep_loss = sum(self.ep_nav_losses) / len(self.ep_nav_losses)

        avg_repo_loss = 0.0
        if len(self.ep_repo_losses) > 0:
            avg_repo_loss = sum(self.ep_repo_losses) / len(self.ep_repo_losses)

        #print("=" * 60)
        #print(f"EP {episode_idx:03d} Summary")
        #print("-" * 60)
        #print(f"Schedule: total={self.total_today} | spawned_success={self._spawn_success}")
        #print(f"Dispatch: total={total_dispatches} | unique={unique_assigned_incidents}")
        #print(f"Nav Loss: {avg_ep_loss:.4f}| Repo Loss: {avg_repo_loss:.4f}")
        #print("=" * 60)

        stats = {
            "episode": episode_idx,
            "avg_rep_reward": avg_rep_reward,
            "rep_moves": n_rep_moves,
            "max_servicing": max_concurrent_assigned,
            "dispatches": len(all_dispatches),
            "total_assignments": total_dispatches,
            "unique_assigned_incidents": unique_assigned_incidents,
            "dispatch_mean_util": mean_util,
            "nav_actions": total_nav,
            "nav_mean_reward": mean_nav_reward,
            "nav_mean_eta": mean_nav_eta,
            "incidents_spawned": total_incidents_spawned,
            "avg_patient_wait": avg_wait,
            "max_patient_wait": max_wait,
            "busy_evs": busy_count,
            "idle_evs": idle_count,
            "total_incidents": len(self.env.incidents),
            "average ep loss": avg_ep_loss,
            "average repo loss": avg_repo_loss,  # Added this key
            "average episodic idle times": self.list_metrics, #evid : avg idle time over episode
        }
        #print("episodic idle time",stats["average episodic idle times\n"])
        return stats
            
            