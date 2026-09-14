# Dataset Card: Solar Prosumer EV Charging Benchmark

## Dataset Summary
This benchmark comprises 813 total residential prosumer EV charging scenarios ($11,405$ operational dwell hours) derived from real-world utility feeds across three major power systems: California ISO (CAISO), PJM Interconnection, and UK Power Networks (UKPN). It provides paired multi-variate time series (electricity spot tariff, solar rooftop PV generation, and residential household baseload demand) coupled with synthetic stochastic EV arrival/departure dwell sessions and clairvoyant optimal LP dispatch schedules.

## Dataset Structure
- **Total Scenarios**: 813
- **Data Partitions**:
  - Training: 546 scenarios ($7,652$ steps)
  - Validation: 87 scenarios ($1,221$ steps)
  - Test: 180 scenarios ($2,532$ steps)
- **Features per Scenario**:
  - `price`: 24-hour historical electricity spot prices ($/kWh), bounds: $[0.008, 0.080]$
  - `pv`: 24-hour historical rooftop solar PV generation (kW), bounds: $[0.0, 10.0]$
  - `load`: 24-hour historical residential baseload demand (kW), bounds: $[0.0, 5.5]$
  - `time`: Time of day normalized by 24.0 hours
  - `soc`: Battery State of Charge (kWh), bounds: $[4.0, 40.0]$
  - `action_gt`: Clairvoyant LP expert continuous charging power action (kW), bounds: $[-7.0, 7.0]$

## Data Sources & Provenance
1. **CAISO OASIS**: Real-time and Day-Ahead Locational Marginal Prices and solar generation feeds (California, USA, 2019–2020).
2. **PJM Data Miner 2**: Hourly LMP spot electricity pricing (Eastern USA, 2019–2021).
3. **UK Power Networks**: Low Carbon London smart meter residential household load trials (East Anglia, UK).

## Redistribution & Intellectual Property
The authors do not claim intellectual property over third-party utility feeds. The processed benchmark slices are curated and redistributed under fair-use academic research provisions to support scientific reproducibility.
