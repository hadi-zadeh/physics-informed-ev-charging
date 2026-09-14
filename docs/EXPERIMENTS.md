# Experimental Protocols & Ablation Design

This document details the configuration and protocol for all experiments.

## 1. Core Model Configurations
- **Proposed E2 (Nominal)**:
  - Architecture: `LSTMPolicy` (216,321 params)
  - Loss Objective: $\mathcal{L}_{\mathrm{BC}} + 0.10 \mathcal{L}_{\mathrm{tariff}} + 0.05 \mathcal{L}_{\mathrm{boundary}}$
  - Seeds: [1, 2, 3, 4, 5]
- **Baseline B0 (Standard Behavior Cloning)**:
  - Architecture: `LSTMPolicy` (216,321 params)
  - Loss Objective: $\mathcal{L}_{\mathrm{BC}}$ (uniform RMSE)
  - Seeds: [1, 2, 3, 4, 5]
- **Baseline B0S (Standard Controls)**:
  - Same as B0 with standard weight decay tuning and early stopping controls.
- **Ablation E1 (Attention Policy + Uniform BC)**:
  - Architecture: `AttentionPolicy` (990,917 params, MHSA temporal encoder)
  - Loss Objective: $\mathcal{L}_{\mathrm{BC}}$ (uniform RMSE)
- **Ablation E3 (Attention Policy + Composite Loss)**:
  - Architecture: `AttentionPolicy` (990,917 params)
  - Loss Objective: $\mathcal{L}_{\mathrm{BC}} + 0.10 \mathcal{L}_{\mathrm{tariff}} + 0.05 \mathcal{L}_{\mathrm{boundary}}$

## 2. Isolated Loss Ablation Protocol (Settings F & G)
Evaluated on the 87 validation scenarios across seeds [1, 2, 3]:
- **Setting F (Boundary Penalty Only)**:
  - $\lambda_{\mathrm{tariff}} = 0.00$, $\lambda_{\mathrm{boundary}} = 0.05$
  - Validation Cost: $\$96.97 \pm 0.67$
  - Boundary Correction: $217.83 \pm 30.49$ kWh
- **Setting G (Tariff Weighting Only)**:
  - $\lambda_{\mathrm{tariff}} = 0.10$, $\lambda_{\mathrm{boundary}} = 0.00$
  - Validation Cost: $\$97.09 \pm 0.14$
  - Boundary Correction: $189.84 \pm 16.13$ kWh
