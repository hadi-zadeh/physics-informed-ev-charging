# Step-by-Step Reproducibility Manual

This manual provides instructions to verify all numerical results, hypothesis tests, and figures reported in the IEEE TSG manuscript.

## 1. Computational Environment
- **Operating System**: Windows 11 / Linux (Ubuntu 22.04 LTS tested)
- **Processor**: Intel Core i7-12700H CPU @ 2.30 GHz or equivalent (14 cores, 20 threads)
- **Memory**: 16 GB DDR4 RAM
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)
- **Python Version**: 3.10.x
- **Key Dependencies**:
  - `torch==2.0.1`
  - `mip==1.17.6` (COIN-OR CBC / Clp simplex solver)
  - `scipy==1.11.2`
  - `pandas==2.0.3`

## 2. Quick Forensic Verification (< 5 seconds)
Run the automated verification suite:

```bash
python reproduction/run_reproduction.py --mode verify
```

Expected output:
```text
==============================================================================
  IEEE TRANSACTIONS ON SMART GRID — REPRODUCIBILITY VERIFICATION SUITE
  Paper: Physics-Informed Cost-and-Boundary-Aware Deep Imitation Learning
==============================================================================

[1/5] Verifying Neural Architecture Parameter Counts...
  * Proposed E2 (LSTM Policy):      216,321 params (Expected: 216,321) -> PASS
  * Ablation E1 (Attention Policy): 990,917 params (Expected: 990,917) -> PASS

[2/5] Verifying Frozen Test Evaluation Results (180 Scenarios, 5 Seeds)...
  * Proposed E2 Cost:              199.00 ± 2.29 (Expected: 199.00 ± 2.29) -> PASS
  * Baseline B0 Cost:              201.08 ± 2.32 (Expected: 201.08 ± 2.32) -> PASS
  * Proposed E2 SPP Clipping:      754.33 ± 221.41 (Expected: 754.33 ± 221.41) -> PASS
  * Baseline B0 SPP Clipping:      1139.67 ± 659.43 (Expected: 1139.67 ± 659.43) -> PASS

[3/5] Verifying Paired Statistical Invariants (Proposed E2 vs Baseline B0)...
  * Cost Wilcoxon p-value:         2.5313e-08 (Expected: 2.5313e-08) -> PASS
  * Feasibility Wilcoxon p-value:  1.0365e-19 (Expected: 1.0365e-19) -> PASS
  * Feasibility Win Rate:          85.00% (Expected: 85.00%) -> PASS
  * Cost Win Rate:                 71.67% (Expected: 71.67%) -> PASS
  * Boundary Cohen's d_z:          -0.747 (Expected: -0.747) -> PASS
  * Cost Cohen's d_z:              -0.366 (Expected: -0.366) -> PASS

[4/5] Verifying Isolated Loss Ablations (Settings F & G)...
  * Setting F (Boundary Only):     217.83 kWh boundary, $96.97 cost -> PASS
  * Setting G (Tariff Only):       189.84 kWh boundary, $97.09 cost -> PASS

[5/5] Verifying Critical Artifact Checksums...
  * Sample SHA-256 Checksums verified: 10/10 sample files matched -> PASS

==============================================================================
  VERIFICATION RESULT: ALL SCIENTIFIC INVARIANTS CERTIFIED & PASSED (100%)
==============================================================================
```

## 3. Full Scenario-Level Evaluation
To execute the complete forward pass over all 180 test scenarios:

```bash
python scripts/run_phase4_frozen_test.py
```
