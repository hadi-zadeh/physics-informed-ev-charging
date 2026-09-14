# Dataset Provenance, Time Zones & Cleaning Audit

## 1. System Coordinates & Time Zones
- **CAISO Data**:
  - Location: California, USA (Pacific Time, UTC-8 standard / UTC-7 daylight saving)
  - Preprocessing: Timestamp alignment converted to standard continuous UTC.
- **PJM Interconnection**:
  - Location: Eastern USA (Eastern Time, UTC-5 standard / UTC-4 daylight saving)
  - Preprocessing: Timestamp alignment converted to standard continuous UTC.
- **UK Power Networks**:
  - Location: Greater London / East Anglia, UK (Greenwich Mean Time, UTC+0)
  - Preprocessing: Scaled from 30-minute half-hourly readings to 1-hour intervals via average power integration.

## 2. Preprocessing Invariants
- No data imputation with forward fill exceeding 2 consecutive hours.
- Extreme price outliers ($> \$1.00$/kWh or $< -\$0.10$/kWh) clipped to standard market operating limits.
- Solar irradiance profiles synchronized with geographical solar noon.
