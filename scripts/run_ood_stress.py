"""
ood_stress_testing.py
Out-of-Distribution (OOD) Robustness and Stress Testing Suite.
Evaluates B0 (Control), B0S, E1, and E2 (Proposed) under 3 challenging OOD stress regimes:
  1. Extreme Electricity Price Shocks (2x, 3x, 5x price surges)
  2. Severe Dwell Time Compression & Erratic Departures (-25%, -50% dwell, deep arrival depletion)
  3. Compound Stress (Extreme Price + Solar Curtailment + Deep Depletion)
"""
import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt

# Ensure clean UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

SANDBOX_ROOT = Path(r"C:\Users\Hadi\Desktop\MyResearch\Top1Percent_Sandbox")
sys.path.insert(0, str(SANDBOX_ROOT / "Code" / "OriginalIL_EVCS"))
from agent import Agent as BaselineAgent

model_dir = SANDBOX_ROOT / "Models" / "AttentionIL"
sys.path.insert(0, str(model_dir))
import importlib.util

spec_model = importlib.util.spec_from_file_location("attention_models", str(model_dir / "model.py"))
attention_models = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(attention_models)

AttentionPolicy = attention_models.AttentionPolicy
LSTMPolicy = attention_models.LSTMPolicy

# Physical Constants
soc_min, soc_max = 4.0, 40.0
e_min, e_max = -7.0, 7.0
charge_efficiency, discharge_efficiency = 0.98, 0.98
eta_s = 0.60
E_max = 100.0
price_min, price_max = 0.008, 0.08
pv_min, pv_max = 0.0, 10.0
load_min, load_max = 0.0, 5.5

class SafetyPostProcessor:
    def __init__(self):
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.e_min = e_min
        self.e_max = e_max
        self.eta_ch = charge_efficiency
        self.eta_dis = discharge_efficiency
        self.E_max = E_max

    def project(self, a_raw: float, cur_soc: float, step_idx: int, dwell_len: int, cur_pv: float, cur_load: float):
        a_spp = float(a_raw)
        pos_clip, neg_clip = 0.0, 0.0
        interventions = 0

        # Step 1: SoC boundary clamping
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

        # Step 2: Departure reachability
        soc_next_tentative = cur_soc + self.eta_ch * a_spp if a_spp >= 0 else cur_soc + a_spp / self.eta_dis
        if step_idx == dwell_len - 1:
            if (self.soc_max - soc_next_tentative) / self.eta_ch > 1e-4:
                a_spp = (self.soc_max - cur_soc) / self.eta_ch
                interventions += 1
        else:
            rem = dwell_len - step_idx - 1
            if (self.soc_max - soc_next_tentative) / self.eta_ch > self.e_max * rem:
                a_spp = (self.soc_max - cur_soc) / self.eta_ch - self.e_max * rem
                interventions += 1

        # Step 3: Transformer limit
        net_e = a_spp + cur_load - cur_pv
        if net_e > self.E_max:
            a_spp = self.E_max + cur_pv - cur_load
            interventions += 1
        elif net_e < -self.E_max:
            a_spp = -self.E_max + cur_pv - cur_load
            interventions += 1

        if a_spp >= 0:
            soc_next = cur_soc + self.eta_ch * a_spp
        else:
            soc_next = cur_soc + a_spp / self.eta_dis

        return a_spp, soc_next, {'pos_clip': pos_clip, 'neg_clip': neg_clip, 'clip_total': pos_clip + neg_clip, 'interventions': interventions}


def load_test_scenarios(n_scenarios=180):
    scen_dir = SANDBOX_ROOT / "Data"
    data_test = np.load(scen_dir / "pjm" / "2021.npy")
    price_test = np.load(scen_dir / "california iso" / "2020.8.npy")[:, 181:]
    load_test = np.load(scen_dir / "uk power network" / "load.npy")[:, 547:]
    pv_test = data_test[:, 1, 28:212]

    lp_agent = BaselineAgent('cpu', {'save_name': 'tmp', 'hidden1': 512, 'hidden2': 256, 'hidden3': 128, 'train': False, 'lr': 0.01, 'weight_decay': 1e-4, 'adam': True, 'momentum': 0.9})

    scenarios = []
    for n in range(min(n_scenarios, price_test.shape[1] - 4)):
        traj, tlen = [], []
        p = price_test[:, n:n+5].flatten('F')
        pv = pv_test[:, n:n+5].flatten('F')
        l = load_test[:, n:n+5].flatten('F')
        traj, tlen = lp_agent.optimum_generator(traj, tlen, p, pv, l)
        scenarios.append({'traj': traj, 'tlen': tlen, 'p': p, 'pv': pv, 'l': l})
    return scenarios


def evaluate_policy_on_scenarios(policy, scenarios, device='cpu', regime="nominal", price_mult=1.0, dwell_ratio=1.0, pv_mult=1.0, dep_soc_floor=None):
    policy.eval()
    spp = SafetyPostProcessor()
    records = []

    with torch.no_grad():
        for s_idx, scen in enumerate(scenarios):
            traj = scen['traj']
            m_len = int(len(traj) * dwell_ratio)
            if m_len < 2:
                m_len = 2

            gt_states = np.array([t[0] for t in traj[:m_len]])

            prices = gt_states[:, 0] * price_mult
            pvs = gt_states[:, 24] * pv_mult
            loads = gt_states[:, 48]

            cur_soc = float(gt_states[0, -1])
            if dep_soc_floor is not None:
                cur_soc = min(cur_soc, dep_soc_floor)

            spp_actions = np.zeros(m_len)
            raw_actions = np.zeros(m_len)
            total_clip = 0.0
            total_interv = 0

            for i in range(m_len):
                # Normalized inputs
                p_hist = torch.tensor((gt_states[i, :24] * price_mult - price_min) / (price_max - price_min), dtype=torch.float).view(1, 24)
                pv_hist = torch.tensor((gt_states[i, 24:48] * pv_mult - pv_min) / (pv_max - pv_min), dtype=torch.float).view(1, 24)
                l_hist = torch.tensor((gt_states[i, 48:72] - load_min) / (load_max - load_min), dtype=torch.float).view(1, 24)
                tn = torch.tensor(gt_states[i, 72:73] / 24.0, dtype=torch.float).view(1, 1)
                sn = torch.tensor([(cur_soc - soc_min) / (soc_max - soc_min)], dtype=torch.float).view(1, 1)

                nin = torch.cat([p_hist, pv_hist, l_hist, tn, sn], dim=1).to(device)
                a_raw = policy(nin).item()
                raw_actions[i] = a_raw

                a_spp, cur_soc, met = spp.project(a_raw, cur_soc, i, m_len, pvs[i], loads[i])
                spp_actions[i] = a_spp
                total_clip += met['clip_total']
                total_interv += met['interventions']

            net_spp = spp_actions + loads - pvs
            cost_spp = np.sum(np.where(net_spp >= 0, prices * net_spp, eta_s * prices * net_spp))

            records.append({
                'scenario_id': s_idx,
                'regime': regime,
                'cost_spp': cost_spp,
                'spp_clipping_kwh': total_clip,
                'spp_interventions': total_interv,
                'departure_soc': cur_soc
            })

    return pd.DataFrame(records)


def run_ood_suite():
    print("=" * 70)
    print("  OUT-OF-DISTRIBUTION (OOD) ROBUSTNESS & STRESS SUITE")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios = load_test_scenarios(180)
    print(f"Loaded {len(scenarios)} base scenarios.")

    args_lstm = {'state_dim': 74, 'action_dim': 1, 'e_min': e_min, 'e_max': e_max, 'hidden1': 512, 'hidden2': 256, 'hidden3': 128}

    models_to_test = {
        'B0': SANDBOX_ROOT / "Checkpoints" / "Val_B0" / "seed_1" / "best.pt",
        'E2': SANDBOX_ROOT / "Checkpoints" / "Val_E2" / "seed_1" / "best.pt"
    }

    policies = {}
    for name, ckpt in models_to_test.items():
        pol = LSTMPolicy(args_lstm).to(device)
        pol.load_state_dict(torch.load(ckpt, map_location=device))
        policies[name] = pol
        print(f"Loaded model: {name} from {ckpt}")

    # Define OOD Stress Regimes
    regimes = [
        {'name': 'Nominal (Baseline Test)', 'price_mult': 1.0, 'dwell_ratio': 1.0, 'pv_mult': 1.0, 'dep_soc_floor': None},
        {'name': 'Price Surge 2x', 'price_mult': 2.0, 'dwell_ratio': 1.0, 'pv_mult': 1.0, 'dep_soc_floor': None},
        {'name': 'Price Shock 3x', 'price_mult': 3.0, 'dwell_ratio': 1.0, 'pv_mult': 1.0, 'dep_soc_floor': None},
        {'name': 'Price Catastrophe 5x', 'price_mult': 5.0, 'dwell_ratio': 1.0, 'pv_mult': 1.0, 'dep_soc_floor': None},
        {'name': 'Dwell Compression -25%', 'price_mult': 1.0, 'dwell_ratio': 0.75, 'pv_mult': 1.0, 'dep_soc_floor': None},
        {'name': 'Dwell Compression -50%', 'price_mult': 1.0, 'dwell_ratio': 0.50, 'pv_mult': 1.0, 'dep_soc_floor': None},
        {'name': 'Deep Depletion (10% SoC)', 'price_mult': 1.0, 'dwell_ratio': 1.0, 'pv_mult': 1.0, 'dep_soc_floor': 4.0},
        {'name': 'Compound Stress (3x Price + Low PV + -25% Dwell)', 'price_mult': 3.0, 'dwell_ratio': 0.75, 'pv_mult': 0.50, 'dep_soc_floor': 4.0},
    ]

    all_results = []

    for r in regimes:
        r_name = r['name']
        print(f"\nEvaluating Regime: {r_name} ...")
        df_b0 = evaluate_policy_on_scenarios(policies['B0'], scenarios, device=device, regime=r_name,
                                             price_mult=r['price_mult'], dwell_ratio=r['dwell_ratio'],
                                             pv_mult=r['pv_mult'], dep_soc_floor=r['dep_soc_floor'])
        df_e2 = evaluate_policy_on_scenarios(policies['E2'], scenarios, device=device, regime=r_name,
                                             price_mult=r['price_mult'], dwell_ratio=r['dwell_ratio'],
                                             pv_mult=r['pv_mult'], dep_soc_floor=r['dep_soc_floor'])

        b0_cost_mean = df_b0['cost_spp'].mean()
        e2_cost_mean = df_e2['cost_spp'].mean()
        cost_saving_pct = (b0_cost_mean - e2_cost_mean) / b0_cost_mean * 100.0

        b0_clip_mean = df_b0['spp_clipping_kwh'].mean()
        e2_clip_mean = df_e2['spp_clipping_kwh'].mean()
        clip_red_pct = (b0_clip_mean - e2_clip_mean) / b0_clip_mean * 100.0

        w_p = stats.wilcoxon(df_e2['cost_spp'], df_b0['cost_spp']).pvalue

        print(f"  B0 Cost: ${b0_cost_mean:.2f} | E2 Cost: ${e2_cost_mean:.2f} (E2 Saving: {cost_saving_pct:+.2f}%, p={w_p:.4e})")
        print(f"  B0 Clipping: {b0_clip_mean:.2f} kWh | E2 Clipping: {e2_clip_mean:.2f} kWh (Boundary Reduction: {clip_red_pct:.2f}%)")

        all_results.append({
            'Regime': r_name,
            'B0_Cost_USD': b0_cost_mean,
            'E2_Cost_USD': e2_cost_mean,
            'Cost_Savings_Pct': cost_saving_pct,
            'Cost_P_Value': w_p,
            'B0_Clipping_kWh': b0_clip_mean,
            'E2_Clipping_kWh': e2_clip_mean,
            'Clipping_Reduction_Pct': clip_red_pct,
            'B0_Interventions': df_b0['spp_interventions'].mean(),
            'E2_Interventions': df_e2['spp_interventions'].mean(),
        })

    summary_df = pd.DataFrame(all_results)
    out_csv = SANDBOX_ROOT / "Results" / "ood_stress_testing_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved OOD Summary table to: {out_csv}")

    # Generate Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar plot of clipping across regimes
    labels = [r['name'].replace(' (Baseline Test)', '') for r in regimes]
    x = np.arange(len(labels))
    width = 0.35

    axes[0].bar(x - width/2, summary_df['B0_Clipping_kWh'], width, label='B0 Baseline (Control)', color='#b0bec5', edgecolor='black', hatch='//')
    axes[0].bar(x + width/2, summary_df['E2_Clipping_kWh'], width, label='E2 Proposed (Physics-Informed)', color='#1e88e5', edgecolor='black')
    axes[0].set_ylabel('Mean Boundary SPP Clipping Energy (kWh)', fontsize=11, fontweight='bold')
    axes[0].set_title('Constraint Violation & Safety Intervention Under OOD Stress', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    axes[0].legend(frameon=True)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Cost comparison across regimes
    axes[1].bar(x - width/2, summary_df['B0_Cost_USD'], width, label='B0 Baseline (Control)', color='#ffab91', edgecolor='black', hatch='\\\\')
    axes[1].bar(x + width/2, summary_df['E2_Cost_USD'], width, label='E2 Proposed (Physics-Informed)', color='#2e7d32', edgecolor='black')
    axes[1].set_ylabel('Mean Total Operational Electricity Cost ($USD)', fontsize=11, fontweight='bold')
    axes[1].set_title('Operational Cost Resilience Under Stress Regimes', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    axes[1].legend(frameon=True)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = SANDBOX_ROOT / "Figures" / "figure_ood_stress_resilience.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to: {fig_path}")

if __name__ == "__main__":
    run_ood_suite()
