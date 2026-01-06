# simple_timeslot_runner.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from MAP_env import MAP


from Entities.ev import EvState
from Entities.Incident import IncidentStatus

from utils.Helpers import build_daily_incident_schedule, point_to_grid_index



LatLng = Tuple[float, float]


@dataclass
class GridTickMetrics:
    tick: int
    gridIndex: int
    imbalance: int

    spawnedInGrid: int
    spawnedSystem: int

    assignedInGrid: int

    unassignedFromSpawnedInGrid: int
    backlogUnassignedInGrid: int
    totalUnassignedInGrid: int

    idleEvs: int
    busyEvs: int


class SimpleTimeslotRunner:
    def __init__(
        self,
        env: MAP,
        *,
        seed: int = 123,
        ticks: int = 30,
        slot_minutes: int = 8,
        busy_min_slots: int = 4,
        busy_max_slots: int = 5,
    ):
        self.env = env
        self.rng = random.Random(seed)
        self.ticks = int(ticks)
        self.slotMinutes = int(slot_minutes)
        self.busyMin = int(busy_min_slots)
        self.busyMax = int(busy_max_slots)

        self.evBusyLeft: Dict[int, int] = {}
        self.evToIncident: Dict[int, int] = {}

        self.currentDay: Optional[pd.Timestamp] = None
        self.schedule: Optional[
            Dict[int, List[Tuple[int, pd.Timestamp, float, float, int, Any, Any]]]
        ] = None

        self._synth_id: int = 1_000_000

    def reset_world(self) -> None:
        self.env.incidents.clear()
        for g in self.env.grids.values():
            g.incidents.clear()

        for ev in self.env.evs.values():
            ev.release_incident()
            ev.state = EvState.IDLE
            ev.status = "Idle"

        self.evBusyLeft.clear()
        self.evToIncident.clear()

    def init_evs_if_needed(self, seed: int = 42) -> None:
        if not getattr(self.env, "evs", None) or len(self.env.evs) == 0:
            self.env.init_evs(seed=seed)

    def build_schedule_from_csv(
        self,
        csv_path: str,
        *,
        time_col: str = "Received DtTm",
        lat_col: Optional[str] = None,
        lng_col: Optional[str] = None,
        wkt_col: Optional[str] = "case_location",
    ) -> None:
        df = pd.read_csv(csv_path)

        series = pd.to_datetime(
            df[time_col],
            format="%Y %b %d %I:%M:%S %p",
            errors="coerce",
        ).dropna()

        days = series.dt.normalize().unique()
        if len(days) == 0:
            raise RuntimeError("No valid days found in dataset for the given time column")

        self.currentDay = pd.Timestamp(self.rng.choice(list(days)))

        self.schedule = build_daily_incident_schedule(
            df,
            day=self.currentDay,
            time_col=time_col,
            lat_col=lat_col,
            lng_col=lng_col,
            wkt_col=wkt_col,
        )

    def spawn_incidents_for_tick(self, t: int) -> List[int]:
        spawned: List[int] = []
        if self.schedule is None:
            return spawned

        rows = self.schedule.get(t, [])
        for (inc_id, ts, lat, lng, pri, _rsp, _hosp) in rows:
            gi = point_to_grid_index(lat, lng, self.env.lat_edges, self.env.lng_edges)
            if gi is None or gi < 0:
                continue

            ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            inc = self.env.create_incident(
                incident_id=int(inc_id),
                grid_index=int(gi),
                location=(float(lat), float(lng)),
                timestamp=ts_py,
                priority=int(pri) if pri is not None else 1,
            )
            spawned.append(int(inc.id))

        return spawned

    def spawn_synthetic_for_tick(self, t: int, *, mean_per_grid: float = 1.0) -> List[int]:
        spawned: List[int] = []
        base_ts = datetime.now().replace(second=0, microsecond=0) + timedelta(
            minutes=t * self.slotMinutes
        )

        for gi in self.env.grids.keys():
            n = self._poisson(mean_per_grid)
            for _ in range(n):
                lat, lng = self.env.grid_center(gi)

                jitter_lat = (self.rng.random() - 0.5) * 0.0005
                jitter_lng = (self.rng.random() - 0.5) * 0.0005

                pri = self.rng.choice([1, 2, 3])
                inc_id = self._next_synth_inc_id()

                inc = self.env.create_incident(
                    incident_id=inc_id,
                    grid_index=int(gi),
                    location=(float(lat + jitter_lat), float(lng + jitter_lng)),
                    timestamp=base_ts,
                    priority=int(pri),
                )
                spawned.append(int(inc.id))

        return spawned

    def dispatch_random_gridwise(self) -> Dict[int, int]:
        assignments: Dict[int, int] = {}

        for gi, g in self.env.grids.items():
            idle_evs: List[int] = []
            for ev_id in g.evs:
                ev = self.env.evs[ev_id]
                if ev.state == EvState.IDLE and ev.status == "Idle" and ev.assignedPatientId is None:
                    idle_evs.append(int(ev_id))

            pending: List[int] = []
            for inc_id in g.incidents:
                inc = self.env.incidents.get(inc_id)
                if inc is None:
                    continue
                if inc.status == IncidentStatus.UNASSIGNED and inc.assignedEvId is None:
                    pending.append(int(inc_id))

            self.rng.shuffle(idle_evs)
            self.rng.shuffle(pending)

            k = min(len(idle_evs), len(pending))
            for i in range(k):
                ev_id = idle_evs[i]
                inc_id = pending[i]

                ev = self.env.evs[ev_id]
                inc = self.env.incidents[inc_id]

                ev.assign_incident(inc_id)
                ev.state = EvState.BUSY
                ev.status = "Busy"

                inc.assign_ev(ev_id)

                busy_slots = self.rng.randint(self.busyMin, self.busyMax)
                self.evBusyLeft[ev_id] = int(busy_slots)
                self.evToIncident[ev_id] = int(inc_id)
                assignments[ev_id] = int(inc_id)

        return assignments

    def release_finished(self) -> List[Tuple[int, int]]:
        finished: List[Tuple[int, int]] = []

        for ev_id in list(self.evBusyLeft.keys()):
            left = int(self.evBusyLeft.get(ev_id, 0)) - 1
            self.evBusyLeft[ev_id] = int(left)

            if left > 0:
                continue

            inc_id = self.evToIncident.get(ev_id)
            ev = self.env.evs.get(ev_id)
            inc = self.env.incidents.get(inc_id) if inc_id is not None else None

            if inc is not None:
                inc.mark_resolved()

                g = self.env.grids.get(inc.gridIndex)
                if g is not None:
                    g.remove_incident(inc.id)

                if inc.id in self.env.incidents:
                    del self.env.incidents[inc.id]

            if ev is not None:
                ev.release_incident()
                ev.state = EvState.IDLE
                ev.status = "Idle"

            finished.append((int(ev_id), int(inc_id) if inc_id is not None else -1))

            del self.evBusyLeft[ev_id]
            if ev_id in self.evToIncident:
                del self.evToIncident[ev_id]

        return finished

    def _tick_spawn(
        self,
        t: int,
        *,
        use_csv_schedule: bool,
        synthetic_mean_per_grid: float,
    ) -> Tuple[List[int], Dict[int, List[int]], Dict[int, int]]:
        if use_csv_schedule:
            spawnedIds = self.spawn_incidents_for_tick(t)
        else:
            spawnedIds = self.spawn_synthetic_for_tick(t, mean_per_grid=synthetic_mean_per_grid)

        spawnedIdsByGrid: Dict[int, List[int]] = {}
        spawnedCountByGrid: Dict[int, int] = {}

        for inc_id in spawnedIds:
            inc = self.env.incidents.get(inc_id)
            if inc is None:
                continue
            gi = int(inc.gridIndex)
            if gi not in spawnedIdsByGrid:
                spawnedIdsByGrid[gi] = []
            spawnedIdsByGrid[gi].append(int(inc_id))

        for gi, ids in spawnedIdsByGrid.items():
            spawnedCountByGrid[gi] = int(len(ids))

        return spawnedIds, spawnedIdsByGrid, spawnedCountByGrid

    def compute_metrics(
        self,
        tick: int,
        assignments: Dict[int, int],
        spawnedSystem: int,
        spawnedIdsByGrid: Dict[int, List[int]],
        spawnedCountByGrid: Dict[int, int],
    ) -> List[GridTickMetrics]:
        assignedCountByGrid: Dict[int, int] = {}
        for _ev_id, inc_id in assignments.items():
            inc = self.env.incidents.get(inc_id)
            if inc is None:
                continue
            gi = int(inc.gridIndex)
            assignedCountByGrid[gi] = assignedCountByGrid.get(gi, 0) + 1

        out: List[GridTickMetrics] = []

        for gi, g in self.env.grids.items():
            gi2 = int(gi)

            spawnedHere = int(spawnedCountByGrid.get(gi2, 0))
            spawnedSetHere = set(spawnedIdsByGrid.get(gi2, []))

            unassignedFromSpawned = 0
            for inc_id in spawnedSetHere:
                inc = self.env.incidents.get(inc_id)
                if inc is None:
                    continue
                if inc.status == IncidentStatus.UNASSIGNED and inc.assignedEvId is None:
                    unassignedFromSpawned += 1

            backlogUnassigned = 0
            for inc_id in g.incidents:
                if int(inc_id) in spawnedSetHere:
                    continue
                inc = self.env.incidents.get(inc_id)
                if inc is None:
                    continue
                if inc.status == IncidentStatus.UNASSIGNED and inc.assignedEvId is None:
                    backlogUnassigned += 1

            totalUnassigned = int(unassignedFromSpawned + backlogUnassigned)

            idle_here = g.count_idle_available_evs(self.env.evs)

            busy_here = 0
            for ev_id in g.evs:
                ev = self.env.evs[ev_id]
                if ev.state == EvState.BUSY:
                    busy_here += 1

            imb = g.calculate_imbalance(self.env.evs, self.env.incidents)
            assignedHere = int(assignedCountByGrid.get(gi2, 0))

            out.append(
                GridTickMetrics(
                    tick=int(tick),
                    gridIndex=int(gi2),
                    imbalance=int(imb),
                    spawnedInGrid=int(spawnedHere),
                    spawnedSystem=int(spawnedSystem),
                    assignedInGrid=int(assignedHere),
                    unassignedFromSpawnedInGrid=int(unassignedFromSpawned),
                    backlogUnassignedInGrid=int(backlogUnassigned),
                    totalUnassignedInGrid=int(totalUnassigned),
                    idleEvs=int(idle_here),
                    busyEvs=int(busy_here),
                )
            )

        return out

    def run(
        self,
        *,
        use_csv_schedule: bool = True,
        csv_path: Optional[str] = None,
        time_col: str = "Received DtTm",
        lat_col: Optional[str] = None,
        lng_col: Optional[str] = None,
        wkt_col: Optional[str] = "case_location",
        synthetic_mean_per_grid: float = 1.0,
        print_each_tick: bool = True,
        tail_ticks: int = 8,
    ) -> List[GridTickMetrics]:
        self.init_evs_if_needed()
        self.reset_world()

        if use_csv_schedule:
            if not csv_path:
                raise RuntimeError("use_csv_schedule is True but csv_path is None")
            self.build_schedule_from_csv(
                csv_path,
                time_col=time_col,
                lat_col=lat_col,
                lng_col=lng_col,
                wkt_col=wkt_col,
            )

        logs: List[GridTickMetrics] = []

        for t in range(self.ticks):
            spawnedIds, spawnedIdsByGrid, spawnedCountByGrid = self._tick_spawn(
                t,
                use_csv_schedule=use_csv_schedule,
                synthetic_mean_per_grid=synthetic_mean_per_grid,
            )
            spawnedSystem = int(len(spawnedIds))

            assignments = self.dispatch_random_gridwise()
            self.release_finished()

            metrics = self.compute_metrics(
                t,
                assignments,
                spawnedSystem,
                spawnedIdsByGrid,
                spawnedCountByGrid,
            )
            logs.extend(metrics)

            if print_each_tick:
                imbByGrid: Dict[int, int] = {}
                spawnedByGrid: Dict[int, int] = {}
                for m in metrics:
                    imbByGrid[int(m.gridIndex)] = int(m.imbalance)
                    spawnedByGrid[int(m.gridIndex)] = int(m.spawnedInGrid)

                totalAssigned = sum(m.assignedInGrid for m in metrics)
                totalUnassigned = sum(m.totalUnassignedInGrid for m in metrics)
                totalIdle = sum(m.idleEvs for m in metrics)
                totalBusy = sum(m.busyEvs for m in metrics)

                print(
                    f"tick={t:03d} spawnedSystem={spawnedSystem} assignedSystem={totalAssigned} "
                    f"unassignedSystem={totalUnassigned} idleSystem={totalIdle} busySystem={totalBusy}"
                )
                print("spawnedByGrid:", spawnedByGrid)
                print("imbalances:", imbByGrid)

        df = pd.DataFrame([m.__dict__ for m in logs])

        if len(df) > 0:
            maxTick = int(df["tick"].max())
            startTick = maxTick - (int(tail_ticks) - 1)
            tail = df[df["tick"] >= startTick].copy()
            tail = tail.sort_values(["tick", "gridIndex"])

            print("\nLast ticks (key columns):")
            cols = [
                "tick",
                "gridIndex",
                "imbalance",
                "spawnedInGrid",
                "assignedInGrid",
                "unassignedFromSpawnedInGrid",
                "backlogUnassignedInGrid",
                "totalUnassignedInGrid",
                "idleEvs",
                "busyEvs",
            ]
            print(tail[cols].to_string(index=False))

        return logs

    def _poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        L = pow(2.718281828459045, -lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1

    def _next_synth_inc_id(self) -> int:
        self._synth_id += 1
        return int(self._synth_id)


def main() -> None:
    grid_config_path = "Data/grid_config_2d.json"

    env = MAP(grid_config_path)
    env.init_evs(seed=42)

    runner = SimpleTimeslotRunner(
        env,
        seed=123,
        ticks=40,
        slot_minutes=8,
        busy_min_slots=4,
        busy_max_slots=5,
    )

    logs = runner.run(
        use_csv_schedule=False,
        synthetic_mean_per_grid=0.4,
        print_each_tick=True,
        tail_ticks=8,
    )

    df = pd.DataFrame([m.__dict__ for m in logs])
    out_path = "timeslot_metrics.csv"
    df.to_csv(out_path, index=False)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
