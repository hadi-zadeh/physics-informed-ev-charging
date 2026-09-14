"""
data_pipeline.py
Data normalization, state tensor assembly, and scenario loading pipeline.
"""
import numpy as np
import torch
from typing import Tuple, Dict, Any


class DataPipeline:
    """
    Applies strict physical normalization bounds across all state variables:
      - Price: [0.008, 0.080] $/kWh
      - PV Generation: [0.0, 10.0] kW
      - Residential Baseload: [0.0, 5.5] kW
      - Battery SoC: [4.0, 40.0] kWh
      - Time of Day: 24.0 hours scaling
    """
    def __init__(self, price_min: float = 0.008, price_max: float = 0.080,
                 pv_min: float = 0.0, pv_max: float = 10.0,
                 load_min: float = 0.0, load_max: float = 5.5,
                 soc_min: float = 4.0, soc_max: float = 40.0,
                 time_scaling: float = 24.0):
        self.price_min = price_min
        self.price_max = price_max
        self.pv_min = pv_min
        self.pv_max = pv_max
        self.load_min = load_min
        self.load_max = load_max
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.time_scaling = time_scaling

    def normalize_state(self, raw_states: np.ndarray, cur_soc: float,
                        device: torch.device) -> torch.Tensor:
        pn = (torch.tensor(raw_states[:24], device=device, dtype=torch.float) - self.price_min) / (self.price_max - self.price_min)
        pvn = (torch.tensor(raw_states[24:48], device=device, dtype=torch.float) - self.pv_min) / (self.pv_max - self.pv_min)
        ln = (torch.tensor(raw_states[48:72], device=device, dtype=torch.float) - self.load_min) / (self.load_max - self.load_min)
        tn = torch.tensor([raw_states[72]], device=device, dtype=torch.float) / self.time_scaling
        sn = (torch.tensor([cur_soc], device=device, dtype=torch.float) - self.soc_min) / (self.soc_max - self.soc_min)
        return torch.cat([pn.unsqueeze(0), pvn.unsqueeze(0), ln.unsqueeze(0), tn.unsqueeze(0), sn.unsqueeze(0)], dim=1)
