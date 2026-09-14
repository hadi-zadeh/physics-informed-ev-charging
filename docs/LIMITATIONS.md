# Scope, Assumptions & Limitations

To ensure rigorous scientific transparency, the assumptions, experimental boundaries, and disclosures of this study are outlined below:

## 1. 3-Day Temporal Overlap Disclosure
- In the benchmark indexing inherited from Huang et al. (*IEEE Transactions on Intelligent Transportation Systems*, 2024 / GitHub `ZhenhaoH/IL_EVCS`), global Days `547..549` (July 1–3, 2020) overlap between the tail of the 550-day training partition and the start of the 184-day test partition for CAISO Spot Prices and UKPN Load.
- This affects exactly **3 out of 180 test scenarios** ($1.67\%$ of the test set).
- Solar PV generation in the test set is derived exclusively from Calendar Year 2021 (`pjm_2021`), ensuring 0 overlapping days and distinct meteorological conditions.
- Pre-existing sensitivity verification demonstrates that omitting these 3 overlapping scenarios leaves the primary findings intact: boundary correction reduction is **$32.98\%$** (vs. **$33.81\%$** across all 180 scenarios) with Wilcoxon $p = 4.40 \times 10^{-19}$.

## 2. Physical & Economical Modeling Assumptions
- **Battery Aging**: Electrochemical battery capacity degradation is not dynamically modeled within the real-time loss; the model enforces safe depth-of-discharge boundaries ($10\% \le \mathrm{SoC} \le 100\%$) and symmetric charging/discharging efficiency $\eta_c = \eta_d = 0.98$.
- **Range Anxiety Parameters**: The exponential range anxiety penalty formulation uses scale $c_1 = 0.005$ and shape parameter $c_2 = -3.0$. In the expert LP optimization, the active objective weight is set to $\alpha_{\mathrm{active}} = 0$ (economic cost minimization) while range anxiety is tracked empirically.
- **AC Power Flow & Network Constraints**: The residential prosumer model represents active power exchange with the distribution feeder constrained by the local transformer thermal limit ($E_{\max} = 100$ kW). Full AC power flow, voltage regulation, and reactive power dynamics are not modeled and represent important future work.
- **Prosumer Aggregation**: The framework is evaluated on individual residential prosumer systems. Multi-agent coordination across distribution feeders with voltage regulation constraints is identified as an important avenue for future research.

## 3. Continuous LP Exactness (Theorem 1)
- Under charging and discharging efficiencies $\eta_c, \eta_d < 1$ and positive buying tariff ($P_b > 0$), simultaneous charging and discharging is strictly sub-optimal, guaranteeing continuous LP relaxation exactness without binary integer variables.
