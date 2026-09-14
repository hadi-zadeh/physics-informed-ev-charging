# One-Command Reproduction Guide (`reproduction/`)

This directory provides a unified entry point for rapid verification of the paper's scientific results and hypothesis tests.

## Fast Forensic Verification Mode (`--mode verify`)
Instantly verifies that the repository's models, evaluation tables, statistical tests, parameter counts, and data splits match the certified values in the IEEE TSG manuscript:

```bash
python reproduction/run_reproduction.py --mode verify
```

### Verified Benchmark Invariants:
- **E2 Cost**: $\$199.00 \pm 2.29$ (LP Reference: $\$199.50$, $-0.25\% \pm 1.15\%$)
- **Boundary Energy Reduction**: $-33.81\%$ ($1139.67 \pm 659.43$ kWh $ightarrow 754.33 \pm 221.41$ kWh)
- **Hypothesis Testing**:
  - Feasibility Wilcoxon $p = 1.0365 	imes 10^{-19}$ (Cohen's $d_z = -0.745$, 85.00% win rate)
  - Cost Wilcoxon $p = 2.5313 	imes 10^{-8}$ (Cohen's $d_z = -0.366$, 71.67% win rate)
- **Isolated Ablations**: Setting F ($217.83$ kWh, $\$96.97$) and Setting G ($189.84$ kWh, $\$97.09$)
- **Architecture**: Exactly 216,321 parameters for Proposed E2 (`LSTMPolicy`)

## Full Test Scenario Evaluation Mode (`--mode full`)
Executes policy evaluation across all 180 frozen test scenarios:

```bash
python reproduction/run_reproduction.py --mode full
```
