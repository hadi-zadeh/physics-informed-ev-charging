# Benchmark Data Partitions & Temporal Overlap Disclosure (`data/splits/`)

## Partition Breakdown
- **Training Set**: 546 scenarios ($7,652$ operational dwell steps).
  - Derived from CAISO 2019/2020 (days 1 to 546), PJM 2019/2020, and UKPN residential load.
- **Validation Set**: 87 scenarios ($1,221$ operational dwell steps).
  - Derived from CAISO 2020 (days 550 to 636) and PJM 2020.
- **Frozen Test Set**: 180 scenarios ($2,532$ operational dwell steps).
  - Derived from CAISO 2020 (days 640 to 730) and PJM 2021 (unseen out-of-sample future calendar year).

## 3-Day Temporal Overlap Disclosure & Sensitivity Audit
To uphold scientific transparency, the temporal index overlap between training and test sets is formally disclosed:
- **Nature of Overlap**: Global Days `547..549` (July 1–3, 2020) overlap between the tail of the 550-day training partition and the start of the 184-day test partition for the CAISO Spot Price and UKPN Load series.
- **Impact Scope**: This indexing affects exactly **3 out of 180 test scenarios** ($1.67\%$ of the frozen test set). This partitioning was inherited directly from the original published benchmark indexing of Huang et al. (*IEEE Transactions on Intelligent Transportation Systems*, 2024 / GitHub `ZhenhaoH/IL_EVCS`).
- **Complete Disjointness of Solar PV**: Solar PV generation in the test set is derived exclusively from **Calendar Year 2021** (`pjm_2021[:, 1, 28:212]`), ensuring 0 overlapping days, distinct seasonal weather patterns, and meteorological independence.
- **Sensitivity Verification**: An independent ablation omitting the 3 overlapping scenarios demonstrates that the scientific findings remain robust:
  - Boundary correction reduction without the 3 scenarios: **$32.98\%$** (compared to **$33.81\%$** across all 180 scenarios).
  - Statistical significance without the 3 scenarios: **$p = 4.40 \times 10^{-19}$** (Wilcoxon signed-rank test).
  - This analysis is documented in Section VI-A of the manuscript.
