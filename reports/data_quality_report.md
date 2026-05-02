# Data Quality Report
**Project:** Azerbaijan Cotton Yield & Risk Prediction  
**Team:** Risk Masters  
**Date:** April 2026  

---

## 1. Raw Cotton Dataset

| Check | Result |
|---|---|
| Source file | `data/raw/cotton_dataset.xls` |
| Original format | Wide (districts as rows, years as columns) |
| Reshaped format | Long (one row per district-year) |
| Total districts in raw file | 29 |
| Years covered | 2000 – 2024 |
| Total rows after reshape | 725 |
| Null yield values | Present in 14 districts |

### Districts Dropped (had null yield values)
14 districts were removed from the dataset entirely because they contained at least one missing yield value across the 25-year period. Partial imputation was not applied — a district was either fully present or fully excluded to avoid introducing bias into the model.

### Districts Retained
15 districts were retained after cleaning. All 15 have complete yield records for 2000–2024 with no missing values.

| District | Avg Yield (t) | Min (t) | Max (t) |
|---|---|---|---|
| Barda | ~22.2 | — | — |
| Tartar | ~21.7 | — | — |
| Yevlakh | ~22.1 | — | — |
| Aghjabadi | ~18.8 | — | — |
| Beylagan | ~19.0 | — | — |
| Goranboy | ~17.4 | — | — |
| Aghdam | ~17.2 | — | — |
| Saatli | ~19.0 | — | — |
| Sabirabad | ~13.8 | — | — |
| Imishli | ~16.5 | — | — |
| Salyan | ~17.1 | — | — |
| Zardab | ~11.2 | — | — |
| Bilasuvar | ~17.6 | — | — |
| Kurdamir | ~11.6 | — | — |
| Neftchala | ~14.1 | — | — |

> Exact min/max values are available in DuckDB: `SELECT region, MIN(yield_tonnes), MAX(yield_tonnes) FROM clean_cotton GROUP BY region`

---

## 2. Raw Weather Dataset

| Check | Result |
|---|---|
| Source | Open-Meteo Archive API (no API key required) |
| Stations | 15 (one per district, direct coordinate fetch) |
| Variables fetched | `temperature_2m_mean`, `precipitation_sum`, `relative_humidity_2m_mean`, `wind_speed_10m_max` |
| Date range | 2000-01-01 – 2025-12-31 |
| Rows per station | ~9,615 daily rows |
| Total rows (raw_weather) | ~144,225 |
| Fetch method | Incremental — skips already-fetched stations |
| Rate limit handling | 20-second pause between API calls |

---

## 3. Cleaning Steps Applied

### Cotton Cleaning (`clean_cotton`)
| Step | Action |
|---|---|
| Strip whitespace | District names trimmed |
| Null encoding | `-`, `…`, `...` converted to `NaN` |
| District filtering | Districts with any null yield dropped entirely |
| Station mapping | `weather_station = region` (1-to-1, same name) |
| Final rows | 375 (15 districts × 25 years) |
| Final nulls | 0 |

### Weather Cleaning (`clean_weather`)
| Step | Action |
|---|---|
| Year filter | Only 2000–2025 retained (2026 excluded — partial year) |
| Temperature bounds | `temp_mean > 60°C` or `< -40°C` → set to NaN |
| Precipitation bounds | `precipitation < 0` → set to 0.0 |
| Humidity bounds | `humidity_mean > 100` → 100.0 / `< 0` → 0.0 |
| Wind speed bounds | `wind_speed < 0` → set to NaN |
| Gap interpolation | Linear interpolation, max 3 consecutive days per station |
| Final rows | 142,455 |
| Final nulls | 0 (after interpolation) |

---

## 4. Feature Engineering Quality

| Check | Result |
|---|---|
| Features table rows | 375 (15 districts × 25 years) |
| Feature columns | 40 weather features (10 per stage × 4 stages) |
| Total columns | 44 (4 base + 40 features) |
| Null features | 0 |
| Leakage columns in features | None — risk/label columns strictly in `features_with_risk` only |
| GDD negative values | 0 — all GDD values non-negative ✓ |

### Growth Stage Windows
| Stage | Months | Key Features |
|---|---|---|
| Planting | March – April | temp_mean, total_rain, frost_days, GDD |
| Growing | May – August | heat_stress_days, total_rain, GDD, max_dry_streak |
| Boll Forming | August – September | temp_mean, dry_days, humidity_mean |
| Harvest | September – November | rainy_days, humidity_mean, frost_days |

---

## 5. Quality Check Results (quality_checks.py)

| Check | Status | Detail |
|---|---|---|
| No nulls in features | ✓ PASSED | 0 nulls across all 44 columns |
| Train/test split integrity | ✓ PASSED | Train ≤2021, Test >2021, no overlap |
| Yield range sanity | ✓ PASSED | All values in range 0–500 tonnes |
| Year range coverage | ✓ PASSED | 2000–2024, 25 unique years |
| Feature column count | ✓ PASSED | 44 columns ≥ 44 expected minimum |
| GDD non-negative | ✓ PASSED | All 4 GDD columns ≥ 0 |
| All 15 districts present | ✓ PASSED | 15 districts, each with 25 years |

---

## 6. Train / Test Split

| Split | Years | Rows | Purpose |
|---|---|---|---|
| Training | 2000 – 2021 | 330 | Model training and walk-forward CV |
| Test | 2022 – 2024 | 45 | Held-out evaluation |
| Prediction | 2025 | 15 | Final output (no known yield) |

---

## 7. Known Limitations

| Limitation | Impact |
|---|---|
| Weather data only | Soil nutrients, irrigation, fertilizer not modelled |
| 375 observations | Small dataset — robust models and CV required |
| Post-2021 distribution shift | Test R² negative — likely reflects real-world policy/practice changes not visible in weather data |
| 4 weather variables only | Original dataset had more variables; reduced set impacts some feature proxies (e.g. heat stress estimated from temp_mean rather than temp_max) |
| Partial 2026 data | 2026 excluded from all tables — only ~4 months available at time of writing |

---

## 8. DuckDB Table Inventory

| Table | Rows | Cols | Description |
|---|---|---|---|
| `raw_cotton` | 725 | 3 | Original reshaped cotton data |
| `raw_weather` | ~144,225 | 9 | Original API weather data |
| `clean_cotton` | 375 | 4 | Cleaned, null-free cotton records |
| `clean_weather` | ~142,455 | 9 | Cleaned, interpolated weather data |
| `features` | 375 | 44 | Growth stage features (training use) |
| `features_with_risk` | 375 | 55 | Features + risk scores and labels (website/EDA use) |
| `predictions` | 15 | 9 | 2025 yield and risk predictions | 