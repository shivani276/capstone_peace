# services/repositioning.py
#
from typing import Dict, List
from Entities.ev import EV, EvState
from Entities.GRID import Grid
from Entities.Incident import Incident
from utils.Helpers import utility_repositioning
class RepositioningService:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def accept_reposition_offers(self, evs, grids, incidents):
        for g_idx, g in grids.items():
            offers = []
            for nb in g.neighbours:
                if nb in grids:
                    for ev_id in grids[nb].evs:
                        v = evs[ev_id]
                        if v.state == EvState.IDLE:
                            u = utility_repositioning(v.aggIdleTime, v.aggIdleEnergy)
                            if u is None:
                                continue
                            offers.append((u, v))

            offers.sort(key=lambda x: x[0], reverse=True)

            cap = max(0, g.imbalance)
            for u, v in offers[:cap]:
                #print(f"[Redeployment] ACCEPT EV {v.id} → Grid {g_idx} | u={u:.3f}")
                v.status     = "Repositioning"
                v.nextGrid   = g_idx
                v.sarns["reward"] = u
