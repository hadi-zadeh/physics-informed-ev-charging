# Changelog

All notable changes to this research repository will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-09-09
### Added
- Complete implementation of Physics-Informed Cost-and-Boundary-Aware Imitation Learning framework (`src/`).
- Differentiable Composite Imitation Loss ($\mathcal{L}_{\mathrm{BC}} + \lambda_{\mathrm{tariff}}\mathcal{L}_{\mathrm{tariff}} + \lambda_{\mathrm{boundary}}\mathcal{L}_{\mathrm{boundary}}$).
- Deterministic 3-Step Safety Post-Processing (`src/spp/safety_projection.py`) guaranteeing 100% departure SoC fulfillment and transformer limit compliance.
- Clairvoyant LP Expert solver (`src/expert/lp_expert.py`) utilizing COIN-OR CBC / Clp simplex solver via Python-MIP 1.17.6 with Theorem 1 continuous exactness.
- 180 frozen out-of-sample test scenarios spanning CAISO, PJM, and UK Power Networks across 5 random seeds (2,532 active dwell hours).
- Isolated loss ablation results (Settings F & G) verifying individual contributions of boundary barrier and tariff weighting.
- Unified one-command reproduction suite (`reproduction/run_reproduction.py`).
- Publication-quality figures (Figures 1 through 7) and statistical hypothesis test summaries.
- Comprehensive documentation: dataset card, provenance audit, reproducibility manual, experiment specs, and limitations disclosure.
