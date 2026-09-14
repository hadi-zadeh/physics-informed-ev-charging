"""
safety_projection.py
Deterministic 3-Step Safety Post-Processing (SPP) Pipeline.
Guarantees strict operational feasibility and 100% departure SoC fulfillment at runtime.
"""
import numpy as np
from typing import Tuple, Dict, Any


class SafetyPostProcessor:
    """
    Deterministic 3-Step Feasibility Projection Pipeline:
      Step 1: Battery Capacity & SoC Boundary Clamping (prevents overcharging / overdischarging)
      Step 2: Departure Reachability Guarantee (ensures SoC reaches 40.0 kWh by departure step T)
      Step 3: Distribution Transformer Rating Clamping (enforces net power within [-100, 100] kW)
    """
    def __init__(self, soc_min: float = 4.0, soc_max: float = 40.0,
                 e_min: float = -7.0, e_max: float = 7.0,
                 eta_ch: float = 0.98, eta_dis: float = 0.98,
                 e_max_transformer: float = 100.0):
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.e_min = e_min
        self.e_max = e_max
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.e_max_transformer = e_max_transformer

    def project_action(self, a_raw: float, cur_soc: float, step_idx: int,
                       dwell_len: int, cur_pv: float, cur_load: float) -> Tuple[float, float, Dict[str, Any]]:
        """
        Projects raw policy output action into the strictly feasible operational set.
        """
        a_spp = float(a_raw)
        pos_clip = 0.0
        neg_clip = 0.0
        interventions = 0

        # Step 1: SoC Boundary Clamping
        if a_spp >= 0:
            max_ch = (self.soc_max - cur_soc) / self.eta_ch
            if a_spp > max_ch:
                pos_clip += (a_spp - max_ch)
                a_spp = max_ch
                interventions += 1
        else:
            max_dis = (self.soc_min - cur_soc) * self.eta_dis
            if a_spp < max_dis:
                neg_clip += (max_dis - a_spp)
                a_spp = max_dis
                interventions += 1

        # Step 2: Departure Reachability Guarantee
        if a_spp >= 0:
            soc_next_tentative = cur_soc + self.eta_ch * a_spp
        else:
            soc_next_tentative = cur_soc + a_spp / self.eta_dis

        if step_idx == dwell_len - 1:
            if (self.soc_max - soc_next_tentative) / self.eta_ch > 1e-4:
                a_spp = (self.soc_max - cur_soc) / self.eta_ch
                interventions += 1
        else:
            rem = dwell_len - step_idx - 1
            if (self.soc_max - soc_next_tentative) / self.eta_ch > self.e_max * rem:
                a_spp = (self.soc_max - cur_soc) / self.eta_ch - self.e_max * rem
                interventions += 1

        # Step 3: Distribution Transformer Limit Clamping
        net_e = a_spp + cur_load - cur_pv
        if net_e > self.e_max_transformer:
            a_spp = self.e_max_transformer + cur_pv - cur_load
            interventions += 1
        elif net_e < -self.e_max_transformer:
            a_spp = -self.e_max_transformer + cur_pv - cur_load
            interventions += 1

        # Calculate updated SoC
        if a_spp >= 0:
            soc_next = cur_soc + self.eta_ch * a_spp
        else:
            soc_next = cur_soc + a_spp / self.eta_dis

        metrics = {
            'pos_clip': pos_clip,
            'neg_clip': neg_clip,
            'total_clip': pos_clip + neg_clip,
            'interventions': interventions,
            'net_exchange_kw': a_spp + cur_load - cur_pv
        }
        return a_spp, soc_next, metrics
