# Evaluation & Benchmark Scripts (`scripts/`)

This directory contains standalone execution scripts for evaluating policies and running ablation experiments:

## Scripts
1. **`run_phase4_frozen_test.py`**:
   - Executes policy evaluation across all 180 frozen out-of-sample test scenarios (2,532 dwell hours).
   - Applies the deterministic 3-step Safety Post-Processing (SPP) at each operational step.
   - Measures online decision latency, clipping energy, and economic costs.
   ```bash
   python scripts/run_phase4_frozen_test.py
   ```

2. **`run_phase3_6_isolated_ablations.py`**:
   - Analyzes the isolated loss component ablations: Setting F ($\lambda_{\mathrm{tariff}}=0, \lambda_{\mathrm{boundary}}=0.05$) and Setting G ($\lambda_{\mathrm{tariff}}=0.10, \lambda_{\mathrm{boundary}}=0$).
   - Evaluated on the 87 validation scenarios across seeds [1, 2, 3].
   ```bash
   python scripts/run_phase3_6_isolated_ablations.py
   ```
