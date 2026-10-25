# Data used

| File | What it is | Source |
|---|---|---|
| `district_estimates.csv`, `segment_estimates.csv`, `district_profile.csv` | Survey-weighted resilience estimates with bootstrap 95% intervals, by district and by population segment | Computed from FinScope Rwanda 2024 |
| `drivers.csv`, `fairness.csv`, `calibration.csv`, `levers.csv`, `headline.json` | Model explanations, fairness audit, calibration and association results | Computed by `train.py` |
| `rwanda_districts.geojson` | 30 district boundaries (ADM2) | geoBoundaries, CC BY 4.0, https://www.geoboundaries.org |

**Raw microdata is not included.** FinScope Rwanda 2024 (NISR ref. RWA-NISR-FINSCOPE-2024-v01, Access to Finance Rwanda) is available from the NISR microdata portal: https://microdata.statistics.gov.rw . All files above are aggregates (no individual records).
