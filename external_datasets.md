# External Datasets

External datasets for the congestion pricing project (Jan 2023 – Dec 2025).
Coverage for the full timeline, including 2023, was checked against each source on 2026-09-15.

| # | Dataset | Role | Coverage | Granularity |
|---|---------|------|----------|-------------|
| 1 | MTA CBD Taxi Zones | Label trips as inside or outside the tolled area | 38 Manhattan taxi zones | Static list |
| 2 | MTA Subway Hourly Ridership | Test whether riders switched to the subway | 2023–2024 and 2025+ (two datasets) | Hourly, by station |
| 3 | NOAA Daily Weather (Central Park) | Weather in both models | Full 2023–2025 | Daily |
| 4 | MTA CBD Taxi/FHV Speeds | One line of background on traffic speeds | Oct 2019 – Aug 2025 | Monthly, 3 areas |
| 5 | TLC taxi zone lookup and shapefile | Zone names, maps, buffer ring of neighbouring zones | Static | Zone |

## Downloading from data.ny.gov (Socrata)

Datasets 1, 2 and 4 come from data.ny.gov. The API returns **only 1,000 rows by default**, so every download script must:

- set `$limit` (for example 50,000) and page with `$offset` until a request returns no rows
- sort with `$order` so pages don't overlap or skip rows
- check the final row count against a `$select=count(*)` query

## 1. MTA Central Business District Taxi Zones

- **Page:** https://data.ny.gov/d/yfdc-w5jh
- **CSV:** https://data.ny.gov/resource/yfdc-w5jh.csv
- **Notes:** Official MTA list of the 38 TLC taxi zones subject to the CBD toll. Matched about 97% of trips charged `cbd_congestion_fee` in yellow taxi data for March 2025. The same list is applied to all three years so every year uses the same rule.

## 2. MTA Subway Hourly Ridership

- **2020–2024:** https://data.ny.gov/d/wujg-7c2s (about 25.6M rows for 2023 and 27M for 2024)
- **Beginning 2025:** https://data.ny.gov/d/5wq4-mkjj (about 44.5M rows)
- **Notes:**
  - Each station has a latitude and longitude for a spatial join to taxi zones.
  - Aggregate on the server before downloading, for example by station and hour with a SoQL `$query`. Sum across `payment_method` and `fare_class_category`.
  - The two datasets must be stacked, so check that station IDs and column names match between them.
  - Do **not** use 2025 ridership as a Model 2 input. The toll affected it, so it would leak the effect into the forecast.

## 3. NOAA Daily Weather Summaries (GHCN-Daily, Central Park)

- **Station:** `USW00094728` (NY City Central Park)
- **API (CSV):** https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094728&startDate=2023-01-01&endDate=2025-12-31&dataTypes=PRCP,SNOW,SNWD,TMAX,TMIN&format=csv
- **Coverage check:** 365 days for 2023, 366 for 2024 and 365 for 2025.
- **Notes:**
  - Use daily data because NOAA hourly data (`global-hourly`) has no records after Aug 2025 for Central Park, LaGuardia or JFK.
  - Missing `SNOW` and `SNWD` values on days without snow become 0. State this in the preprocessing section.
  - Weather is the same for every zone on a given day. In Model 1 it enters only as `treated × weather`, because the date fixed effects absorb citywide weather.
- **Hourly fallback:** [Open-Meteo Historical Weather API](https://archive-api.open-meteo.com/v1/archive?latitude=40.78&longitude=-73.97&start_date=2023-01-01&end_date=2025-12-31&hourly=temperature_2m,precipitation,snowfall&timezone=GMT). Its values come from a weather model rather than station readings, and it snaps to a grid cell (it returned 40.81, -74.02 for these coordinates). Request times in GMT and convert them to New York time yourself. Use it only if the time-band results depend on hourly weather.

## 4. MTA Central Business District Taxi and For-Hire Vehicle Speeds

- **Page:** https://data.ny.gov/d/6p29-6xqn
- **CSV:** https://data.ny.gov/resource/6p29-6xqn.csv
- **Notes:** Zones are `CBD`, `CBD Adjacent` and `Greater NYC`. There are 32 monthly data points in the project timeline (Jan 2023 – Aug 2025), with none for Sep–Dec 2025. Use it for a single line of context ("traffic in the zone got X% faster") rather than as a model input. Cut it first if the report runs long.

## 5. Supporting TLC files

- **Taxi zone lookup:** https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
- **Taxi zone shapefile:** https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
- **Notes:**
  - Used to build the **buffer ring**: zones outside the CBD that touch it or are within about 1 km of it. Ring zones are left out of the control group.
  - Zone IDs 56/57 and 103/104/105 share shapes, so check for duplicates when joining and mapping.
  - Zones 264 ("Unknown") and 265 ("Outside of NYC") have no shape.
  - Reproject to a projected coordinate system (for example EPSG:2263, NY State Plane in feet) before measuring distances for the buffer ring.

## Dropped

- **MTA CRZ Vehicle Entries** (https://data.ny.gov/d/t6yz-b64h): starts 5 Jan 2025, so it has no data from before the toll and can't support the before/after comparison. Its entry points also don't map to taxi zones. Dropped to save page space.
