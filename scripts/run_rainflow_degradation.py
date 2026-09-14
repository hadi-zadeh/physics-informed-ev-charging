"""
battery_degradation_analysis.py
Non-Linear Li-ion Battery Degradation & Health Preservation Modeling Suite.
Implements:
  1. Rainflow Cycle-Counting Algorithm for EV battery SoC trajectories.
  2. Semi-Empirical Depth-of-Discharge (DoD) & Average SoC Stress Model (Wang/Han formulation).
  3. Monetary Battery Capacity Fade & Cycle Life Extension Quantification.
  4. Total Cost of Ownership: Electricity Procurement Cost + Battery Degradation Cost.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import matplotlib.pyplot as plt
from pathlib import Path

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

LSTMPolicy = attention_models.LSTMPolicy

# Physical & Battery Parameters
soc_min, soc_max = 4.0, 40.0
e_min, e_max = -7.0, 7.0
charge_efficiency, discharge_efficiency = 0.98, 0.98
eta_s = 0.60
E_max = 100.0
price_min, price_max = 0.008, 0.08
pv_min, pv_max = 0.0, 10.0
load_min, load_max = 0.0, 5.5

BATTERY_CAPACITY_KWH = 40.0
BATTERY_CAPEX_USD = 5200.0  # ~$130/kWh * 40 kWh
EOL_CAPACITY_FADE = 0.20    # 20% capacity loss defines EOL (80% SOH)

# Semi-empirical degradation constants (NMC chemistry)
# Wang et al., Journal of Power Sources
DOD_ALPHA = 1.95             # Non-linear DoD power factor
SOC_BETA = 0.82              # High SoC holding stress
SEI_COEFF = 1.35e-4          # Cycle fade scaling per equivalent cycle

def rainflow_counting(soc_series: np.ndarray):
    """
    Standard rainflow cycle counting algorithm for battery SoC profiles.
    Returns list of (cycle_range_kwh, mean_soc_kwh, count) tuples.
    """
    # Find turning points (local extrema)
    diffs = np.diff(soc_series)
    turns = [soc_series[0]]
    for i in range(1, len(diffs)):
        if (diffs[i-1] > 0 and diffs[i] < 0) or (diffs[i-1] < 0 and diffs[i] > 0):
            turns.append(soc_series[i])
    turns.append(soc_series[-1])
    
    if len(turns) < 3:
        rng = abs(soc_series[-1] - soc_series[0])
        avg = (soc_series[-1] + soc_series[0]) / 2.0
        return [(rng, avg, 0.5)]

    stack = []
    cycles = []
    
    for pt in turns:
        stack.append(pt)
        while len(stack) >= 3:
            s0, s1, s2 = stack[-3], stack[-2], stack[-1]
            dy1 = abs(s1 - s0)
            dy2 = abs(s2 - s1)
            if dy2 >= dy1:
                # dy1 is a closed cycle
                avg_val = (s0 + s1) / 2.0
                cycles.append((dy1, avg_val, 1.0))
                stack.pop(-2)
                stack.pop(-2)
            else:
                break
                
    # Residual half cycles
    for i in range(len(stack) - 1):
        dy = abs(stack[i+1] - stack[i])
        avg = (stack[i+1] + stack[i]) / 2.0
        cycles.append((dy, avg, 0.5))
        
    return cycles


def compute_cycle_degradation(cycles):
    """
    Computes capacity fade and monetary degradation cost from rainflow cycles.
    """
    total_loss_pct = 0.0
    half_cycles = 0.0
    full_cycles = 0.0
    deep_cycles = 0.0  # DoD > 60%
    
    for dy_kwh, avg_kwh, count in cycles:
        dod = min(1.0, dy_kwh / BATTERY_CAPACITY_KWH)
        avg_soc_norm = min(1.0, max(0.0, avg_kwh / BATTERY_CAPACITY_KWH))
        
        # Wang semi-empirical stress model
        stress_dod = (dod ** DOD_ALPHA)
        stress_soc = np.exp(SOC_BETA * (avg_soc_norm - 0.5))
        loss = SEI_COEFF * stress_dod * stress_soc * count
        
        total_loss_pct += loss
        if count >= 1.0:
            full_cycles += count
        else:
            half_cycles += count
            
        if dod >= 0.60:
            deep_cycles += count
            
    # Monetary cost of capacity fade
    # Cost = Pack_Cost * (Fade / EOL_Fade)
    cost_deg_usd = BATTERY_CAPEX_USD * (total_loss_pct / EOL_CAPACITY_FADE)
    
    return {
        'capacity_fade_pct': total_loss_pct * 100.0,
        'degradation_cost_usd': cost_deg_usd,
        'full_cycles': full_cycles,
        'half_cycles': half_cycles,
        'deep_cycles': deep_cycles
    }


def run_battery_degradation_evaluation():
    print("=" * 70)
    print("  EXPLICIT NON-LINEAR BATTERY DEGRADATION MODELING SUITE")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load scenarios
    from ood_stress_testing import load_test_scenarios, SafetyPostProcessor
    scenarios = load_test_scenarios(180)
    print(f"Loaded {len(scenarios)} frozen test scenarios.")
    
    args_lstm = {'state_dim': 74, 'action_dim': 1, 'e_min': e_min, 'e_max': e_max, 'hidden1': 512, 'hidden2': 256, 'hidden3': 128}
    
    models = {
        'B0': SANDBOX_ROOT / "Checkpoints" / "Val_B0" / "seed_1" / "best.pt",
        'E2': SANDBOX_ROOT / "Checkpoints" / "Val_E2" / "seed_1" / "best.pt"
    }
    
    policies = {}
    for name, pth in models.items():
        pol = LSTMPolicy(args_lstm).to(device)
        pol.load_state_dict(torch.load(pth, map_location=device))
        policies[name] = pol

    spp = SafetyPostProcessor()
    
    results = {'B0': [], 'E2': [], 'UC': []}
    
    for s_idx, scen in enumerate(scenarios):
        traj = scen['traj']
        m_len = len(traj)
        gt_states = np.array([t[0] for t in traj])
        prices = gt_states[:, 0]
        pvs = gt_states[:, 24]
        loads = gt_states[:, 48]
        
        # 1. Evaluate B0 and E2
        for pol_name, policy in policies.items():
            cur_soc = float(gt_states[0, -1])
            soc_trajectory = [cur_soc]
            actions = np.zeros(m_len)
            
            with torch.no_grad():
                for i in range(m_len):
                    p_hist = torch.tensor((gt_states[i, :24] - price_min) / (price_max - price_min), dtype=torch.float).view(1, 24)
                    pv_hist = torch.tensor((gt_states[i, 24:48] - pv_min) / (pv_max - pv_min), dtype=torch.float).view(1, 24)
                    l_hist = torch.tensor((gt_states[i, 48:72] - load_min) / (load_max - load_min), dtype=torch.float).view(1, 24)
                    tn = torch.tensor(gt_states[i, 72:73] / 24.0, dtype=torch.float).view(1, 1)
                    sn = torch.tensor([(cur_soc - soc_min) / (soc_max - soc_min)], dtype=torch.float).view(1, 1)
                    
                    nin = torch.cat([p_hist, pv_hist, l_hist, tn, sn], dim=1).to(device)
                    a_raw = policy(nin).item()
                    a_spp, cur_soc, met = spp.project(a_raw, cur_soc, i, m_len, pvs[i], loads[i])
                    actions[i] = a_spp
                    soc_trajectory.append(cur_soc)
                    
            net_e = actions + loads - pvs
            energy_cost = np.sum(np.where(net_e >= 0, prices * net_e, eta_s * prices * net_e))
            
            # Rainflow degradation
            cycles = rainflow_counting(np.array(soc_trajectory))
            deg_met = compute_cycle_degradation(cycles)
            
            results[pol_name].append({
                'scenario_id': s_idx,
                'energy_cost_usd': energy_cost,
                'capacity_fade_pct': deg_met['capacity_fade_pct'],
                'deg_cost_usd': deg_met['degradation_cost_usd'],
                'total_cost_usd': energy_cost + deg_met['degradation_cost_usd'],
                'deep_cycles': deg_met['deep_cycles'],
                'full_cycles': deg_met['full_cycles']
            })
            
        # 2. Evaluate UC (Uncontrolled Baseline)
        soc_uc = float(gt_states[0, -1])
        soc_uc_traj = [soc_uc]
        actions_uc = np.zeros(m_len)
        for i in range(m_len):
            a_uc = min(e_max, (soc_max - soc_uc) / charge_efficiency)
            soc_uc += a_uc * charge_efficiency
            soc_uc_traj.append(soc_uc)
            actions_uc[i] = a_uc
        net_uc = actions_uc + loads - pvs
        cost_uc = np.sum(np.where(net_uc >= 0, prices * net_uc, eta_s * prices * net_uc))
        cycles_uc = rainflow_counting(np.array(soc_uc_traj))
        deg_uc = compute_cycle_degradation(cycles_uc)
        results['UC'].append({
            'scenario_id': s_idx,
            'energy_cost_usd': cost_uc,
            'capacity_fade_pct': deg_uc['capacity_fade_pct'],
            'deg_cost_usd': deg_uc['degradation_cost_usd'],
            'total_cost_usd': cost_uc + deg_uc['degradation_cost_usd'],
            'deep_cycles': deg_uc['deep_cycles'],
            'full_cycles': deg_uc['full_cycles']
        })

    # Summaries
    df_b0 = pd.DataFrame(results['B0'])
    df_e2 = pd.DataFrame(results['E2'])
    df_uc = pd.DataFrame(results['UC'])
    
    summary_data = [
        {
            'Policy': 'Uncontrolled Charging (UC)',
            'Mean Energy Cost ($)': df_uc['energy_cost_usd'].mean(),
            'Mean Deg Cost ($)': df_uc['deg_cost_usd'].mean(),
            'Mean Total Cost ($)': df_uc['total_cost_usd'].mean(),
            'Cumulative Fade (%)': df_uc['capacity_fade_pct'].sum(),
            'Mean Deep Cycles': df_uc['deep_cycles'].mean(),
            'Estimated Battery Life (Years)': 10.0 * (EOL_CAPACITY_FADE * 100.0 / (df_uc['capacity_fade_pct'].sum() * (365.0 / 180.0)))
        },
        {
            'Policy': 'Baseline B0 (Uniform BC)',
            'Mean Energy Cost ($)': df_b0['energy_cost_usd'].mean(),
            'Mean Deg Cost ($)': df_b0['deg_cost_usd'].mean(),
            'Mean Total Cost ($)': df_b0['total_cost_usd'].mean(),
            'Cumulative Fade (%)': df_b0['capacity_fade_pct'].sum(),
            'Mean Deep Cycles': df_b0['deep_cycles'].mean(),
            'Estimated Battery Life (Years)': 10.0 * (EOL_CAPACITY_FADE * 100.0 / (df_b0['capacity_fade_pct'].sum() * (365.0 / 180.0)))
        },
        {
            'Policy': 'Proposed E2 (Physics-Informed)',
            'Mean Energy Cost ($)': df_e2['energy_cost_usd'].mean(),
            'Mean Deg Cost ($)': df_e2['deg_cost_usd'].mean(),
            'Mean Total Cost ($)': df_e2['total_cost_usd'].mean(),
            'Cumulative Fade (%)': df_e2['capacity_fade_pct'].sum(),
            'Mean Deep Cycles': df_e2['deep_cycles'].mean(),
            'Estimated Battery Life (Years)': 10.0 * (EOL_CAPACITY_FADE * 100.0 / (df_e2['capacity_fade_pct'].sum() * (365.0 / 180.0)))
        }
    ]
    
    summary_df = pd.DataFrame(summary_data)
    out_csv = SANDBOX_ROOT / "Results" / "battery_degradation_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 70)
    print("BATTERY HEALTH & DEGRADATION EVALUATION RESULTS:")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    
    # Statistical tests
    p_deg = stats.wilcoxon(df_e2['deg_cost_usd'], df_b0['deg_cost_usd']).pvalue
    p_tot = stats.wilcoxon(df_e2['total_cost_usd'], df_b0['total_cost_usd']).pvalue
    print(f"\nWilcoxon Deg Cost P-Value (E2 vs B0): {p_deg:.4e}")
    print(f"Wilcoxon Total Combined Cost P-Value (E2 vs B0): {p_tot:.4e}")
    
    # Plotting Pareto / Cost Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Stacked Bar Chart: Energy Cost + Degradation Cost
    policies_labels = ['UC (Uncontrolled)', 'B0 (Baseline BC)', 'E2 (Physics-Informed)']
    energy_costs = summary_df['Mean Energy Cost ($)'].values
    deg_costs = summary_df['Mean Deg Cost ($)'].values
    
    x = np.arange(len(policies_labels))
    width = 0.45
    
    axes[0].bar(x, energy_costs, width, label='Direct Electricity Cost ($)', color='#1976d2', edgecolor='black')
    axes[0].bar(x, deg_costs, width, bottom=energy_costs, label='Battery Degradation Cost ($)', color='#f57c00', edgecolor='black')
    axes[0].set_ylabel('Mean Cost Per Scenario ($USD)', fontsize=11, fontweight='bold')
    axes[0].set_title('Total Cost of Ownership (TCO) Decomposition', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policies_labels, fontsize=10)
    axes[0].legend(frameon=True)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # 2. Cumulative Capacity Fade & Estimated Lifespan
    axes[1].bar(x, summary_df['Cumulative Fade (%)'], width, color=['#e53935', '#fb8c00', '#43a047'], edgecolor='black')
    axes[1].set_ylabel('Cumulative Battery Capacity Loss (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Cumulative Battery Aging Over 180 Evaluation Scenarios', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policies_labels, fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate lifespan on bars
    for i, v in enumerate(summary_df['Cumulative Fade (%)']):
        years = summary_df['Estimated Battery Life (Years)'].iloc[i]
        axes[1].text(i, v * 0.5, f"Est. Life:\n{years:.1f} Yrs", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    fig_path = SANDBOX_ROOT / "Figures" / "figure_battery_degradation_tco.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved publication figure to: {fig_path}")

if __name__ == '__main__':
    run_battery_degradation_evaluation()
