# Benchmark Evaluation & Figures (`evaluation/`)

This directory contains the certified evaluation outputs, ablation summaries, statistical hypothesis test results, and publication figures:

## Subdirectories
- **`main_results/`**:
  - `final_test_summary_table.csv`: Complete benchmark comparison across B0, B0S, E1, E2, E3, and LP over the 180 frozen test scenarios.
- **`ablations/`**:
  - `PHASE_2_ISOLATED_LOSS_ABLATION_RESULTS.json`: Quantitative outcomes for isolated loss ablations Setting F (Boundary only) and Setting G (Tariff only).
- **`statistical_tests/`**:
  - `b0_vs_e2_paired_test_stats.json`: Paired Wilcoxon signed-rank tests, Cohen's $d_z$ effect sizes, 10,000-sample bootstrap confidence intervals, and scenario win rates.
- **`figures/`**:
  - `figure1_cost_parity_b0_vs_e2.png`: Cost parity scatter plot and cumulative distribution.
  - `figure2_optimality_gap_distributions.png`: Relative optimality gap distributions vs clairvoyant LP.
  - `figure3_spp_clipping_distributions.png`: Action Boundary Correction Energy reduction distribution.
  - `figure4_scenario_improvement_waterfall.png`: Waterfall improvement ranking across 180 scenarios.
  - `figure5_val_vs_test_generalization.png`: Validation vs. test generalization fidelity.
  - `figure6_seed_robustness_boxplot.png`: Seed robustness boxplot across Seeds 1 to 5.
  - `figure7_cost_vs_spp_tradeoff_test.png`: Cost vs. feasibility safety trade-off curve.
