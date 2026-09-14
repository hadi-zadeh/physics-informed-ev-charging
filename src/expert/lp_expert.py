"""
lp_expert.py
Clairvoyant Linear Programming (LP) Expert Solver.
Solves the global optimal residential EV charging schedule over the dwell session [t_arr, t_dep]
using the COIN-OR Clp simplex solver via Python-MIP 1.17.6.

Mathematical Guarantee:
  Under charging and discharging efficiencies eta_c, eta_d < 1 and positive buying tariff,
  Theorem 1 guarantees continuous LP relaxation exactness (no simultaneous charging and discharging).
"""
import numpy as np
from mip import Model, xsum, minimize, CONTINUOUS, CBC
from typing import Dict, Any, Tuple


class ClairvoyantLPExpert:
    def __init__(self, soc_min: float = 4.0, soc_max: float = 40.0,
                 e_min: float = -7.0, e_max: float = 7.0,
                 eta_ch: float = 0.98, eta_dis: float = 0.98,
                 eta_s: float = 0.60, e_max_transformer: float = 100.0,
                 tolerance: float = 1e-7, seed: int = 0):
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.e_min = e_min
        self.e_max = e_max
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.eta_s = eta_s
        self.e_max_transformer = e_max_transformer
        self.tolerance = tolerance
        self.seed = seed

    def solve(self, arrival_time: int, departure_time: int, arrival_soc: float,
              prices: np.ndarray, pv: np.ndarray, load: np.ndarray) -> Dict[str, Any]:
        """
        Solves the continuous linear program over the active dwell horizon.
        Uses Python-MIP's CBC interface with continuous variables (Clp simplex engine).
        """
        slot = departure_time - arrival_time + 24 if departure_time < arrival_time else departure_time - arrival_time
        ha_all = 72
        P_b = prices[arrival_time + ha_all: arrival_time + ha_all + slot]
        P_s = self.eta_s * P_b
        pv_slice = pv[arrival_time + ha_all: arrival_time + ha_all + slot]
        load_slice = load[arrival_time + ha_all: arrival_time + ha_all + slot]

        m = Model(solver_name=CBC)
        m.seed = self.seed
        m.opt_tol = self.tolerance
        m.pump_l = 0

        # All decision variables are CONTINUOUS; continuous LP relaxation exactness is guaranteed by Theorem 1.
        a_ch = [m.add_var(lb=0.0, ub=self.e_max, var_type=CONTINUOUS) for _ in range(slot)]
        a_dis = [m.add_var(lb=self.e_min, ub=0.0, var_type=CONTINUOUS) for _ in range(slot)]
        E_b = [m.add_var(lb=0.0, var_type=CONTINUOUS) for _ in range(slot)]
        E_s = [m.add_var(lb=0.0, var_type=CONTINUOUS) for _ in range(slot)]
        soc = [m.add_var(lb=self.soc_min, ub=self.soc_max, var_type=CONTINUOUS) for _ in range(slot)]

        m.objective = minimize(xsum(P_b[i] * E_b[i] - P_s[i] * E_s[i] for i in range(slot)))

        for i in range(slot):
            m += E_b[i] + pv_slice[i] == E_s[i] + a_ch[i] + a_dis[i] + load_slice[i]
            if i == 0:
                m += soc[i] == arrival_soc
            else:
                m += soc[i] == soc[i - 1] + self.eta_ch * a_ch[i - 1] + a_dis[i - 1] / self.eta_dis
            m += soc[i] <= self.soc_max
            m += soc[i] >= self.soc_min
            m += E_b[i] + E_s[i] <= self.e_max_transformer

        # Departure fulfillment constraint
        m += soc[-1] + self.eta_ch * a_ch[-1] + a_dis[-1] / self.eta_dis == self.soc_max

        status = m.optimize()
        actions = np.array([a_ch[i].x if a_ch[i].x != 0 else a_dis[i].x for i in range(slot)])
        soc_traj = np.array([soc[i].x for i in range(slot)])
        cost_opt = float(m.objective_value)

        return {
            'status': status,
            'cost': cost_opt,
            'actions': actions,
            'soc': soc_traj,
            'slot': slot
        }
