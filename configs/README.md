# Experiment Configurations (`configs/`)

This directory contains configuration files for all frozen experiments and isolated loss ablations:

## Schema
- **`e2/config_e2_nominal.json`**: Nominal Proposed E2 configuration ($\lambda_{\mathrm{tariff}}=0.10, \lambda_{\mathrm{boundary}}=0.05$, LSTM backbone, 216,321 parameters, evaluated across Seeds 1..5).
- **`ablations/config_setting_F.json`**: Setting F ablation ($\lambda_{\mathrm{tariff}}=0.00, \lambda_{\mathrm{boundary}}=0.05$, boundary loss only).
- **`ablations/config_setting_G.json`**: Setting G ablation ($\lambda_{\mathrm{tariff}}=0.10, \lambda_{\mathrm{boundary}}=0.00$, tariff loss only).

All hyperparameter values, optimizer parameters, learning rate schedules, normalization bounds, and LP solver tolerances are documented within each configuration.
