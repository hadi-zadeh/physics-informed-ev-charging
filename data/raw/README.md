# Raw Data Retrieval Guide (`data/raw/`)

Because raw ISO feeds span tens of gigabytes across multiple calendar years, only the processed scenario benchmark arrays (~1.4 MB) are included in this repository under `data/processed/`.

To retrieve the original full-year raw data directly from the respective utilities:

### 1. California ISO (CAISO OASIS)
- **Portal**: http://oasis.caiso.com/
- **API Endpoint**: `http://oasis.caiso.com/oasisapi/SingleZip`
- **Reports**: `PRC_LMP` (Locational Marginal Prices) and `SLD_REN_FCST` (Renewable Solar Forecast & Actuals).

### 2. PJM Interconnection
- **Portal**: https://dataminer2.pjm.com/
- **Feed**: Real-Time Hourly LMPs (RTO aggregate).

### 3. UK Power Networks
- **Portal**: https://data.ukpowernetworks.co.uk/
- **Project**: Low Carbon London (Residential Smart Meter Demand Trials).
