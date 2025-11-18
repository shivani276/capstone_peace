# Controller.py
import random
from typing import Optional, List

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from MAP_env import MAP
from Entities.ev import EvState
from utils.Epsilon import EpsilonScheduler, hard_update, soft_update
from utils.Helpers import (
    build_daily_incident_schedule,
    point_to_grid_index,
    W_MIN, W_MAX, E_MIN, E_MAX,H_MIN, H_MAX,
    utility_repositioning,
)

from DQN import DQNetwork, ReplayBuffer
from Entities.ev import EvState

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
        lat_col: Optional[str] = "Latitude",
        lng_col: Optional[str] = "Longitude",
        wkt_col: Optional[str] = None,
        
        
    ):
        self.env = env
        self.ticks_per_ep = ticks_per_ep
        self.rng = random.Random(seed)

        # agent params
        self.global_step = 0
        self.epsilon_scheduler = EpsilonScheduler(
        start=1.0,     # start high exploration
        end=0.1,       # end at 10 percent random
        decay_steps=5000
        )
        self.epsilon = 1.0 


        self.busy_fraction = 0.5

        # DQNs (state=19, action=9 [stay + 8 neighbours])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dim = 12
        action_dim = 9
        self.dqn_reposition_main = DQNetwork(state_dim, action_dim).to(self.device)
        self.dqn_reposition_target = DQNetwork(state_dim, action_dim).to(self.device)
        self.dqn_reposition_target.load_state_dict(self.dqn_reposition_main.state_dict())
        self.opt_reposition = torch.optim.Adam(self.dqn_reposition_main.parameters(), lr=1e-3)
        self.buffer_reposition = ReplayBuffer(100_000)

        state_dim_nav = 2 * NAV_K
        action_dim_nav = NAV_K
        self.nav_step = 0
        self.nav_target_update = 500  # soft update every N training steps
        self.nav_tau = 0.005          # Polyak factor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dqn_navigation_main = DQNetwork(state_dim_nav, action_dim_nav).to(self.device)
        self.dqn_navigation_target = DQNetwork(state_dim_nav, action_dim_nav).to(self.device)
        self.dqn_navigation_target.load_state_dict(self.dqn_navigation_main.state_dict())
        self.opt_navigation = torch.optim.Adam(self.dqn_navigation_main.parameters(), lr=1e-3)
        self.buffer_navigation = ReplayBuffer(100_000)

        hard_update(self.dqn_reposition_target, self.dqn_reposition_main)
        hard_update(self.dqn_navigation_target, self.dqn_navigation_main)


        print("[Controller] DQNs initialised:")
        print("  Reposition main / target:", sum(p.numel() for p in self.dqn_reposition_main.parameters()))
        print("  Navigation main / target:", sum(p.numel() for p in self.dqn_navigation_main.parameters()))
        print("  Device:", self.device)

        # dataset (for incident schedule per episode)
        self.df = pd.read_csv(csv_path)
        self.time_col = time_col
        self.lat_col = lat_col
        self.lng_col = lng_col
        self.wkt_col = wkt_col

        self._schedule = None
        self._current_day = None

        # EV randomisation bounds (already enforced in Helpers via constants)
        self.max_idle_minutes = W_MAX
        self.max_idle_energy = E_MAX
        self.max_wait_time_HC = H_MAX

    
    def _get_direction_neighbors_for_index(self, index: int) -> list[int]:

        n_rows = len(self.env.lat_edges) - 1
        n_cols = len(self.env.lng_edges) - 1

        cell_row = index // n_cols
        cell_col = index % n_cols

        # Offsets matching DIRECTION_ORDER
        offset_map = {
            "N":  (1, 0),
            "NE": (1, 1),
            "E":  (0, 1),
            "SE": (-1, 1),
            "S":  (-1, 0),
            "SW": (-1, -1),
            "W":  (0, -1),
            "NW": (1, -1),
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


    # ---------- state/action helpers ----------
    def _pad_neighbors(self, nbs: List[int]):
        N = 8
        n = (nbs[:N] if len(nbs) >= N else nbs + [-1] * (N - len(nbs)))
        #mask = [1 if x != -1 else 0 for x in n]
        #n_feat = [0 if x == -1 else x for x in n]
        return n #n_feat, mask


    #==================REPOSITION DQN STUFF=====================#

    #==============State=============================#
    '''
    def _build_state(self, ev) -> list[float]:
        gi = ev.gridIndex
        g = self.env.grids[gi]
        nbs = g.neighbours
        n8 = self._pad_neighbors(nbs)

        vec: list[float] = []

        # 1) Own grid index + own imbalance  (2 features)
        own_imb = float(g.calculate_imbalance(self.env.evs, self.env.incidents))
        vec.append(float(gi))
        vec.append(own_imb)

        # 2) For each of 8 neighbours: (neighbour_index, neighbour_imbalance) (16 features)
        for nb in n8:
            if nb == -1:
                vec.extend([0.0, 0.0])
            else:
                
                imb = float(self.env.grids[nb].calculate_imbalance(self.env.evs, self.env.incidents))
                vec.extend([float(nb), imb])

        # 3) EV idle time + idle energy (2 features)
        vec.append(float(ev.aggIdleTime))
        vec.append(float(ev.aggIdleEnergy))

        # Total length: 2 (own) + 8*2 (neighbours) + 2 (idle) = 20
        return vec
        '''
    
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

        # build state vector:
        # [gi, imb_self, imb_N, imb_NE, ..., imb_NW, aggIdleTime, aggIdleEnergy]
        vec: list[float] = []
        vec.append(float(gi))
        vec.append(imb_self)
        vec.extend(neigh_imbs)
        vec.append(float(ev.aggIdleTime))
        vec.append(float(ev.aggIdleEnergy))

        return vec

    #=========================ACTION=================================#
    '''
    def _select_action(self, state_vec: list[float], gi: int) -> int:
        # 9 slots: [stay] + 8 neighbours (padded)
        nbs = self.env.grids[gi].neighbours
        n8= self._pad_neighbors(nbs)
        actions = [gi] + n8
        #mask = [1] + mask8

        if self.rng.random() < self.epsilon:
            valid = [i for i, a in enumerate(actions) if a != -1]
            slot = self.rng.choice(valid) if valid else 0
            return actions[slot]

        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.dqn_reposition_main(s).detach().cpu().numpy().ravel()
        for i, a in enumerate(actions):
            if a == -1:
                q[i] = -1e9
        slot = int(np.argmax(q))
        return actions[slot]
        '''
    def _select_action(self, state_vec: list[float], gi: int) -> int:

        neighbours = self._get_direction_neighbors_for_index(gi)  # len 8

        # validity mask: stay is always valid; moves valid only if neighbour exists
        valid_mask = [1]  # slot 0 (stay)
        for nb_idx in neighbours:
            valid_mask.append(1 if nb_idx != -1 else 0)

        # epsilon-greedy over slots 0..8
        if self.rng.random() < self.epsilon:
            valid_slots = [i for i, m in enumerate(valid_mask) if m == 1]
            slot = self.rng.choice(valid_slots) if valid_slots else 0
        else:
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.dqn_reposition_main(s).detach().cpu().numpy().ravel()  # shape (9,)
            # mask invalid actions
            for i, m in enumerate(valid_mask):
                if m == 0:
                    q[i] = -1e9
            slot = int(np.argmax(q))

        # map slot back to actual grid index
        if slot == 0:
            return gi  # stay
        else:
            dir_index = slot - 1          # 0..7
            nb_idx = neighbours[dir_index]
            return nb_idx if nb_idx != -1 else gi

    
    #==================Push Reposition Experiences===============#
    def _push_reposition_transition(self, ev) -> None:
        """
        Take what we stored in ev.sarns, build s', and push (s,a,r,s').
        """
        s  = ev.sarns.get("state")
        a  = ev.sarns.get("action")
        r  = ev.sarns.get("reward")
        if s is None or a is None:
            return
        # next-state is built wrt the EV's chosen nextGrid if accepted,
        # otherwise its current grid (stay)
        if ev.state == EvState.IDLE and ev.status == "Repositioning":
            next_g = ev.gridIndex

        s2 = self._build_state(ev)
        done = 0.0  # not terminal at this stage

        # push to replay (tensorise once; buffer will normalise if needed)
        import torch
        s_t  = torch.tensor(s,  dtype=torch.float32)
        a_t  = torch.tensor(a,  dtype=torch.int64)
        r_t  = torch.tensor(r,  dtype=torch.float32)
        s2_t = torch.tensor(s2, dtype=torch.float32)
        d_t  = torch.tensor(done, dtype=torch.float32)
        print(f"[RepositionReplay] push EV={ev.id} r={float(r):.3f} a={a}")
        self.buffer_reposition.push(s_t, a_t, r_t, s2_t, d_t)

    #==================Train Reposition==========================#
    import torch.nn.functional as F

    def _train_reposition(self, batch_size: int = 64, gamma: float = 0.99) -> None:
        if len(self.buffer_reposition) < batch_size:
            return

        # sample from replay buffer
        states, actions, rewards, next_states, dones = self.buffer_reposition.sample(
            batch_size,
            device=self.device
        )

        # Q(s,a) from main net
        q_values = self.dqn_reposition_main(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target: r + gamma * max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.dqn_reposition_target(next_states).max(1)[0]
            target = rewards + gamma * (1.0 - dones) * q_next

        loss = F.smooth_l1_loss(q_sa, target)

        self.opt_reposition.zero_grad()
        loss.backward()
        self.opt_reposition.step()

        # soft-update target net
        tau = 0.005
        for t_param, o_param in zip(self.dqn_reposition_target.parameters(),
                                    self.dqn_reposition_main.parameters()):
            t_param.data.mul_(1.0 - tau).add_(tau * o_param.data)


    # ---------- episode reset ----------
    def _reset_episode(self) -> None:
        import pandas as pd

        # 1) clear incidents from env + grids
        self.env.incidents.clear()
        for g in self.env.grids.values():
            g.incidents.clear()

        # 2) randomise EV placement and base fields (no offers yet)
        all_idx = list(self.env.grids.keys())
        for ev in self.env.evs.values():
            gi = self.rng.choice(all_idx)
            self.env.move_ev_to_grid(ev.id, gi)
            # 3) busy/idle split only (do NOT build offers here)
            if self.rng.random() < self.busy_fraction:
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
            # clear SARNS
            ev.sarns.clear()
            ev.sarns["state"] = None
            ev.sarns["action"] = None
            ev.sarns["utility"] = None
            ev.sarns["reward"] = None
            ev.sarns["next_state"] = None

        # 4) pick a random day and build schedule
        series = pd.to_datetime(self.df[self.time_col], errors="coerce").dt.normalize().dropna()
        days = series.unique()
        if len(days) == 0:
            raise RuntimeError(f"No valid dates in dataset for {self.time_col}")
        self._current_day = pd.Timestamp(self.rng.choice(list(days)))
        self._schedule = build_daily_incident_schedule(
            self.df,
            self._current_day,
            time_col=self.time_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            wkt_col=self.wkt_col,
        )

        # 5) initialise hospital waits for this episode (random range)
        #    signature: reset_hospital_waits(low_min, high_min, seed)
        self.env.reset_hospital_waits(low_min=H_MIN, high_min=H_MAX, seed=self.rng.randint(1, 10_000))

        # 6) log
        total_today = 0 if not self._schedule else sum(len(v) for v in self._schedule.values())
        print(f"[Controller] _reset_episode ready: day={self._current_day.date()} incidents_today={total_today}")

    # ---------- per-tick ----------
    def _spawn_incidents_for_tick(self, t: int) -> None:
        todays_at_tick = self._schedule.get(t, []) if self._schedule else []
        for (lat, lng) in todays_at_tick:
            gi = point_to_grid_index(lat, lng, self.env.lat_edges, self.env.lng_edges)
            self.env.create_incident(grid_index=gi, location=(lat, lng), priority="MED")

    def _tick(self, t: int) -> None:

        #Hard_update
        hard_update(self.dqn_reposition_target, self.dqn_reposition_main)

        # 1) spawn incidents for this tick
        self._spawn_incidents_for_tick(t)
        
        for g in self.env.grids.values():
            g.imbalance = g.calculate_imbalance(self.env.evs, self.env.incidents)

        # 2) build states and actions for IDLE EVs only
        for ev in self.env.evs.values():
            if ev.state == EvState.IDLE and ev.status == "Idle":
                state_vec = self._build_state(ev)
                ev.sarns["state"] = state_vec
                a_gi = self._select_action(state_vec, ev.gridIndex)
                ev.sarns["action"] = a_gi

        # 3) Algorithm 1: accept offers (sets nextGrid and reward; no movement yet)
        self.env.accept_reposition_offers()
        
        n_offers = self._build_offers_for_idle_evs()

        #update state, action, reward and next_state in replay buffer
        for ev in self.env.evs.values():
            if ev.state == EvState.IDLE and ev.status == "Repositioning":
                self._push_reposition_transition(ev)
        
        # after the loop that calls _push_reposition_transition(ev)
        self._train_reposition(batch_size=64, gamma=0.99)

                
        # 4) Gridwise dispatch (Algorithm 2) using EVs that stayed/rejected
        dispatches = self.env.dispatch_gridwise(beta=0.5)

        # 5) build states and actions for IDLE EVs only
        for ev in self.env.evs.values():
            if ev.state == EvState.BUSY and ev.status == "Navigation":
                state_vec = self._build_state(ev)
                ev.sarns["state"] = state_vec
                a_gi = self._select_action(state_vec, ev.gridIndex)
                ev.sarns["action"] = a_gi

        '''
        # 5) Debug snapshot so you can see it running
        todays = self._schedule.get(t, []) if self._schedule else []
        accepted = sum(
            1
            for ev in self.env.evs.values()
            if ev.state == EvState.IDLE and ev.sarns.get("reward") not in (None, 0.0)
        )
        print(
            f"Tick {t:03d} | incidents+{len(todays):2d} | offers={n_offers:2d} | "
            f"accepted={accepted:2d} | dispatched={len(dispatches):2d}"
        )'''

        self.env.update_after_timeslot(8)
        
        '''for g in self.env.grids.values():
            vehicle_id = []
            vehicle_id = g.evs
            print("Before", vehicle_id)
                  
        

        for g in self.env.grids.values():
            vehicle_id = []
            vehicle_id = g.evs
            print("After", vehicle_id)
        '''
        '''
        # 6) NAV per tick (only for dispatched EVs)
        #    Update hospital waits, then choose hospital via DQN (action), reward = U^N, store transition.
        self.env.tick_hospital_waits(lam=0.04, wmin=5.0, wmax=90.0)

        for (eid, inc_id, _Ud) in dispatches:
            ev = self.env.evs[eid]

            # If EV already at its hospital's grid (when you implement movement), skip NAV
            hid_prev = getattr(ev, "navTargetHospitalId", None)
            if hid_prev is not None:
                hc_prev = self.env.hospitals.get(hid_prev)
                if hc_prev is not None and ev.gridIndex == hc_prev.gridIndex:
                    continue

            # Build state over candidates (patient -> hospital ETAs, hospital waits)
            s_vec, cand_hids, mask = self._build_nav_state(inc_id)
            if sum(mask) == 0:
                continue  # no valid hospitals

            # Epsilon-greedy action over candidates
            slot = self._select_nav_action(s_vec, mask)
            hid = cand_hids[slot]

            # Reward = U^N for this choice
            hc = self.env.hospitals[hid]
            # recompute current eta & wait for the chosen one
            hids, etas, waits = self.env.get_nav_candidates(inc_id, max_k=NAV_K)
            j = hids.index(hid)
            eta_ph, wait_h = etas[j], waits[j]
            r_nav = self._compute_un(eta_ph, wait_h)

            # Record on EV (not movement yet)
            ev.navTargetHospitalId = hid
            ev.navEtaMinutes = eta_ph
            ev.navUtility = r_nav

            # Build next-state immediately (you can also defer to next tick)
            s2_vec, _, _ = self._build_nav_state(inc_id)
            done = 0.0  # no terminal yet (arrival handled when you add movement)

            # Push to replay
            s_t = torch.tensor(s_vec, dtype=torch.float32)
            a_t = torch.tensor(slot, dtype=torch.int64)
            r_t = torch.tensor(r_nav, dtype=torch.float32)
            s2_t = torch.tensor(s2_vec, dtype=torch.float32)
            d_t = torch.tensor(done, dtype=torch.float32)
            self.buffer_navigation.push(s_t, a_t, r_t, s2_t, d_t)

        # Single nav training step per tick
        self._train_navigation(batch_size=64, gamma=0.99)'''


    def run_training_episode(self, episode_idx: int) -> dict:
        self._reset_episode()

        total_rep_reward = 0.0
        n_rep_moves = 0
        total_dispatched = 0

        for t in range(self.ticks_per_ep):
            self._tick(t)

            # collect simple stats
            for ev in self.env.evs.values():
                r = ev.sarns.get("reward")
                if r not in (None, 0.0):
                    total_rep_reward += float(r)
                    n_rep_moves += 1

            # you already get dispatches count from dispatch_gridwise,
            # but easiest is: count SERVICING incidents
            n_servicing = sum(
                1 for inc in self.env.incidents.values()
                if inc.status.name == "SERVICING"
            )
            total_dispatched = max(total_dispatched, n_servicing)

        avg_rep_reward = total_rep_reward / max(1, n_rep_moves)

        stats = {
            "episode": episode_idx,
            "avg_rep_reward": avg_rep_reward,
            "rep_moves": n_rep_moves,
            "max_servicing": total_dispatched,
            "total_incidents": len(self.env.incidents),
        }

        print(
            f"[EP {episode_idx:03d}] avg_rep_reward={avg_rep_reward:.3f} "
            f"moves={n_rep_moves:3d} servicing_max={total_dispatched:3d} "
            f"incidents={len(self.env.incidents):3d}"
        )

        return stats

    

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
    

    #======================NAVIGATION============================#

    def _build_nav_state(self, inc_id: int) -> tuple[list[float], list[int], list[int]]:
        
        hids, etas, waits = self.env.get_nav_candidates(inc_id, max_k=NAV_K)

        # normalise by H_MAX so values are ~0..1
        feats: list[float] = []
        for i in range(NAV_K):
            if i < len(hids):
                feats.append(etas[i] / max(H_MAX, 1e-6))
                feats.append(waits[i] / max(H_MAX, 1e-6))
            else:
                feats.extend([0.0, 0.0])

        mask = [1]*len(hids) + [0]*(NAV_K - len(hids))
        return feats, hids, mask

    def _select_nav_action(self, s_vec: list[float], mask: list[int]) -> int:
        """
        Epsilon-greedy over NAV_K slots. Returns slot index 0..NAV_K-1.
        Masked slots are invalid.
        """
        import numpy as np
        if self.rng.random() < self.epsilon:
            valid = [i for i, m in enumerate(mask) if m == 1]
            return self.rng.choice(valid) if valid else 0

        s = torch.tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.dqn_navigation_main(s).detach().cpu().numpy().ravel()
        for i, m in enumerate(mask):
            if m == 0:
                q[i] = -1e9
        return int(np.argmax(q))
    
    #========================NAV-STATE-ACTION=======================#

    '''def _compute_un(self, eta_ph: float, wait_h: float) -> float:
        # U^N per Eq. (14), using Helpers util you already added
        from utils.Helpers import utility_navigation_un
        # remaining slack style: larger slack ⇒ higher utility
        W_busy = max(0.0, H_MAX - (eta_ph + wait_h))
        return utility_navigation_un(W_busy, H_MIN, H_MAX)'''
    

    def _train_navigation(self, batch_size: int = 64, gamma: float = 0.99):
        # need enough samples
        if len(self.buffer_navigation) < batch_size:
            return

        # sample a batch (ReplayBuffer.sample should accept device=... and return tensors)
        try:
            s, a, r, s2, done = self.buffer_navigation.sample(batch_size, device=self.device)
        except TypeError:
            # fallback if your sample() returns python lists/np arrays
            batch = self.buffer_navigation.sample(batch_size)
            s   = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[0]])
            a   = torch.as_tensor(batch[1], dtype=torch.long,   device=self.device)
            r   = torch.as_tensor(batch[2], dtype=torch.float32, device=self.device)
            s2  = torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch[3]])
            done= torch.as_tensor(batch[4], dtype=torch.float32, device=self.device)

        # target: r + γ (1-done) max_a' Q_target(s')
        with torch.no_grad():
            q2 = self.dqn_navigation_target(s2).max(dim=1).values
            y  = r + gamma * (1.0 - done) * q2

        # current Q(s,a)
        q = self.dqn_navigation_main(s).gather(1, a.view(-1, 1)).squeeze(1)

        loss = torch.nn.functional.smooth_l1_loss(q, y)
        self.opt_navigation.zero_grad()
        loss.backward()
        self.opt_navigation.step()

        # soft-update target
        self.nav_step += 1
        if self.nav_step % self.nav_target_update == 0:
            with torch.no_grad():
                for p_t, p in zip(self.dqn_navigation_target.parameters(),
                                self.dqn_navigation_main.parameters()):
                    p_t.data.mul_(1.0 - self.nav_tau).add_(self.nav_tau * p.data)

        if self.nav_step % 500 == 0:
            print(f"[Controller] NAV train step={self.nav_step} loss={loss.item():.4f}")
    
