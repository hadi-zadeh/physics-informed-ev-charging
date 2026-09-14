# Pre-Trained Policy Checkpoints

This directory contains trained PyTorch neural network checkpoints (`.pt`) for reproducing all experimental evaluations without retraining.

---

## Directory Structure

```text
checkpoints/
├── e2_proposed/
│   ├── e2_seed_1.pt        # Proposed E2 model (LSTM + Composite Loss, Seed 1)
│   ├── e2_seed_2.pt        # Proposed E2 model (Seed 2)
│   ├── e2_seed_3.pt        # Proposed E2 model (Seed 3)
│   ├── e2_seed_4.pt        # Proposed E2 model (Seed 4)
│   └── e2_seed_5.pt        # Proposed E2 model (Seed 5)
├── b0_baseline/
│   ├── b0_seed_1.pt        # Baseline B0 model (LSTM + Uniform RMSE, Seed 1)
│   ├── b0_seed_2.pt        # Baseline B0 model (Seed 2)
│   ├── b0_seed_3.pt        # Baseline B0 model (Seed 3)
│   ├── b0_seed_4.pt        # Baseline B0 model (Seed 4)
│   └── b0_seed_5.pt        # Baseline B0 model (Seed 5)
└── ablations/
    ├── setting_f_boundary_only_seed_1.pt   # Isolated boundary loss (λ_tariff = 0, λ_bound = 0.05)
    ├── setting_f_boundary_only_seed_2.pt
    ├── setting_f_boundary_only_seed_3.pt
    ├── setting_g_tariff_only_seed_1.pt     # Isolated tariff loss (λ_tariff = 0.10, λ_bound = 0)
    ├── setting_g_tariff_only_seed_2.pt
    └── setting_g_tariff_only_seed_3.pt
```

## Model Architecture Specifications
- **Input Dimension:** 24 timesteps × 3 channels (Price, Solar PV, Baseload Demand) + 4 real-time context scalars (Time of Day, Battery SoC, Departure Time, Charger Limit) = 76 state features.
- **Hidden Units:** 64 LSTM units.
- **Parameter Count:** 216,321 trainable parameters.
- **Output:** Continuous real-time charging/discharging power command $a(t) \in [-7.0, +7.0]$ kW.
