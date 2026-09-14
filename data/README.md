# Benchmark Datasets (`data/`)

This directory contains the real-world utility scenario data and partition metadata supporting the IEEE TSG manuscript.

## Third-Party Data Provenance Notice
The underlying raw time series were collected by public electric utilities and distribution network operators:
1. **California ISO (CAISO)**: Day-Ahead and Real-Time Locational Marginal Prices and solar generation feeds (California, USA, 2019–2020). Available via [CAISO OASIS](http://oasis.caiso.com/).
2. **PJM Interconnection**: Hourly Real-Time Locational Marginal Prices (Eastern USA, 2019–2021). Available via [PJM Data Miner 2](https://dataminer2.pjm.com/).
3. **UK Power Networks**: Half-hourly residential smart meter demand profiles from the Low Carbon London project (East Anglia, UK). Available via [UKPN Open Data](https://data.ukpowernetworks.co.uk/).

*Notice*: The authors of this repository do not claim ownership of third-party public utility data. The processed benchmark slices (~1.37 MB) are redistributed strictly for non-commercial academic research and reproducibility purposes. To download the original uncurated feeds, consult `data/raw/README.md`.

## Directory Structure
- **`raw/`**: Retrieval instructions and endpoints for external public utility portals.
- **`processed/`**: Normalized, resampled, and formatted scenario arrays:
  - `california_iso/`: 2019/2020 LMP and solar PV generation profiles.
  - `pjm/`: 2019, 2020, and 2021 LMP pricing profiles.
  - `uk_power_network/`: Residential baseload demand profiles.
- **`splits/`**:
  - `README.md`: Partition breakdown and disclosure of the 3-day temporal overlap in 3 out of 180 test scenarios ($1.67\%$).
  - `splits_info.json`: Machine-readable partition metadata.
- **`DATASET_CARD.md`**: Standard dataset card documenting features, splits, and ethical considerations.
- **`DATA_PROVENANCE.md`**: Preprocessing lineage, normalization boundaries, and time-zone conversions.
