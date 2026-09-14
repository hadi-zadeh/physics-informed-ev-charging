# Data Provenance & Preprocessing Lineage Audit

This document certifies the end-to-end data lineage from public utility feeds to the processed benchmark tensors.

## 1. Raw Data Acquisition
- **CAISO OASIS Portal**:
  - Query: Hourly Day-Ahead and Real-Time Marginal Energy Costs.
  - Granularity: Hourly ($1$ h time-step $\Delta t = 1$ h).
  - Timeframe: Calendar years 2019 and 2020.
- **PJM Interconnection**:
  - Source: Data Miner 2 API (RTO aggregate real-time LMP).
  - Timeframe: Calendar years 2019, 2020, and out-of-sample 2021.
- **UK Power Networks**:
  - Source: Low Carbon London Open Data Portal (residential smart meter profiles).
  - Normalization: Scaled to single-family residential peak load ($5.5$ kW).

## 2. Preprocessing & Normalization
All state features are normalized to $[0, 1]$ before policy ingestion:
$$p_{\mathrm{norm}} = rac{p - p_{\min}}{p_{\max} - p_{\min}}, \quad 	ext{with } p_{\min} = 0.008, \, p_{\max} = 0.080 	ext{ \$/kWh}$$
$$\mathrm{PV}_{\mathrm{norm}} = rac{\mathrm{PV} - \mathrm{PV}_{\min}}{\mathrm{PV}_{\max} - \mathrm{PV}_{\min}}, \quad 	ext{with } \mathrm{PV}_{\min} = 0.0, \, \mathrm{PV}_{\max} = 10.0 	ext{ kW}$$
$$L_{\mathrm{norm}} = rac{L - L_{\min}}{L_{\max} - L_{\min}}, \quad 	ext{with } L_{\min} = 0.0, \, L_{\max} = 5.5 	ext{ kW}$$
$$\mathrm{SoC}_{\mathrm{norm}} = rac{\mathrm{SoC} - \mathrm{SoC}_{\min}}{\mathrm{SoC}_{\max} - \mathrm{SoC}_{\min}}, \quad 	ext{with } \mathrm{SoC}_{\min} = 4.0, \, \mathrm{SoC}_{\max} = 40.0 	ext{ kWh}$$
$$t_{\mathrm{norm}} = rac{t}{24.0}$$
