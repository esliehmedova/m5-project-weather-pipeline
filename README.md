# Topic: Azerbaijan Cotton Yield & Risk Prediction 

## 1. Problem Definition

This project integrates 25 years of historical weather data (2000–2024) with district-level cotton production records to build an explainable prediction system. Low yield predictions are linked to specific agronomic risks across four biological growth stages: planting, growing, boll forming, and harvest. 

## 2. Team & Responsibilities 

### Roles and Responsibilities 

| Task | Member 1 (Ahmadova Esli) | Member 2 (Dirayeva Narmin) | Member 3 (Gasimova Khaver) | Member 4 (Aliyeva Ulviyye) |
|---|---|---|---|---|
| **Role** | Data Engineer | Agro-Analyst & Feature Engineer | ML Engineer | Data Analyst & Visualizer |
| **Files** | `config.py` `ingestion.py` `cleaning.py` | `features.py` | `quality_checks.py` `models.py` | `reports.py` |
| **Task 1** | Project setup & folder structure | Define growth stage windows | Run pre-modelling quality checks | EDA — heatmaps & scatter plots |
| **Task 2** | Fetch 25 years weather data via API | Calculate GDD (base 15.5°C) | Train 4 risk classifiers (one per stage) | Regional yield comparisons |
| **Task 3** | Reshape cotton data wide → long | Compute heat stress, frost, dry streak days | Train yield regressor (XGBoost/Ridge) | Generate 2025 prediction dashboard |
| **Task 4** | Clean nulls, outliers, align datasets | Build risk labels (0/1) and risk scores (0–100%) | Evaluate models (MAE, RMSE, R²) & save via joblib | Export CSV report & write final documentation |
| **Notebooks** | `day_01` `day_02` `day_03` `day_05` | `day_04` `day_07` | `day_08` | `day_06` | 

### Project Timeline (10-Day Plan)

The project follows a structured 10-day roadmap to ensure internal readiness:

| Day | Date | Brief | Focus |
|-----|------|-------|-------|
| 1 | 20 Apr | Kick-Off | Repo setup, API exploration, and project planning |
| 2 | 21 Apr | Data Ingestion | Building the ingestion module and fetching 24 years of historical data |
| 3 | 22 Apr | Database Design | Schema design and data validation queries |
| 4 | 23 Apr | Feature Engineering | Cleaning pipeline and transforming weather data into agro-features |
| 5 | 24 Apr | Automation | Orchestration, incremental loading, and quality gates |
| 6 | 27 Apr | EDA | Descriptive statistics and cross-city comparisons |
| 7 | 28 Apr | Statistical Analysis | Hypothesis testing and final feature selection |
| 8 | 29 Apr | Modeling | Training and evaluating 5 total models |
| 9 | 30 Apr | Dress Rehearsal | Full timed run-through and feedback loop |
| 10 | 01 May | Internal Readiness | Repo freeze and generation of the final PDF report |

## 3. Data Sources

## 3. Data Sources

**Source 1 — Cotton Production Dataset**
- 29 districts × 25 years (2000–2024) = 725 observations
- Values are annual cotton yield in tonnes per district
- 191 missing values across 14 districts — all districts with any missing value dropped entirely
- 15 districts retained with complete 25-year records

**Source 2 — Open-Meteo Historical Weather API**
- Daily weather fetched for all 15 districts using direct GPS coordinates (no shared station mapping)
- Variables collected: mean temperature, precipitation sum, mean relative humidity, max wind speed
- 25 years × ~365 days × 15 districts = ~136,875 daily rows
- No API key required — fully reproducible 

## 4. Features

40 features total — 10 per growth stage × 4 stages.

| Source Variable | Feature Name | Unit | Aggregation | Stage |
|---|---|---|---|---|
| `temperature_2m_mean` | `planting_temp_mean` | °C | Mean over stage window | Planting |
| `temperature_2m_mean` | `planting_heat_stress_days` | days | Count of days mean > 32°C | Planting |
| `temperature_2m_mean` | `planting_frost_days` | days | Count of days mean < 2°C | Planting |
| `temperature_2m_mean` | `planting_GDD` | °C·days | Sum of max(temp_mean − 15.5, 0) | Planting |
| `precipitation_sum` | `planting_total_rain` | mm | Sum over stage window | Planting |
| `precipitation_sum` | `planting_rainy_days` | days | Count of days > 1mm | Planting |
| `precipitation_sum` | `planting_dry_days` | days | Count of days < 1mm | Planting |
| `precipitation_sum` | `planting_max_dry_streak` | days | Longest consecutive dry day run | Planting |
| `relative_humidity_2m_mean` | `planting_humidity_mean` | % | Mean over stage window | Planting |
| `wind_speed_10m_max` | `planting_wind_mean` | km/h | Mean over stage window | Planting |
| `temperature_2m_mean` | `growing_temp_mean` | °C | Mean over stage window | Growing |
| `temperature_2m_mean` | `growing_heat_stress_days` | days | Count of days mean > 32°C | Growing |
| `temperature_2m_mean` | `growing_frost_days` | days | Count of days mean < 2°C | Growing |
| `temperature_2m_mean` | `growing_GDD` | °C·days | Sum of max(temp_mean − 15.5, 0) | Growing |
| `precipitation_sum` | `growing_total_rain` | mm | Sum over stage window | Growing |
| `precipitation_sum` | `growing_rainy_days` | days | Count of days > 1mm | Growing |
| `precipitation_sum` | `growing_dry_days` | days | Count of days < 1mm | Growing |
| `precipitation_sum` | `growing_max_dry_streak` | days | Longest consecutive dry day run | Growing |
| `relative_humidity_2m_mean` | `growing_humidity_mean` | % | Mean over stage window | Growing |
| `wind_speed_10m_max` | `growing_wind_mean` | km/h | Mean over stage window | Growing |
| `temperature_2m_mean` | `boll_forming_temp_mean` | °C | Mean over stage window | Boll Forming |
| `temperature_2m_mean` | `boll_forming_heat_stress_days` | days | Count of days mean > 32°C | Boll Forming |
| `temperature_2m_mean` | `boll_forming_frost_days` | days | Count of days mean < 2°C | Boll Forming |
| `temperature_2m_mean` | `boll_forming_GDD` | °C·days | Sum of max(temp_mean − 15.5, 0) | Boll Forming |
| `precipitation_sum` | `boll_forming_total_rain` | mm | Sum over stage window | Boll Forming |
| `precipitation_sum` | `boll_forming_rainy_days` | days | Count of days > 1mm | Boll Forming |
| `precipitation_sum` | `boll_forming_dry_days` | days | Count of days < 1mm | Boll Forming |
| `precipitation_sum` | `boll_forming_max_dry_streak` | days | Longest consecutive dry day run | Boll Forming |
| `relative_humidity_2m_mean` | `boll_forming_humidity_mean` | % | Mean over stage window | Boll Forming |
| `wind_speed_10m_max` | `boll_forming_wind_mean` | km/h | Mean over stage window | Boll Forming |
| `temperature_2m_mean` | `harvest_temp_mean` | °C | Mean over stage window | Harvest |
| `temperature_2m_mean` | `harvest_heat_stress_days` | days | Count of days mean > 32°C | Harvest |
| `temperature_2m_mean` | `harvest_frost_days` | days | Count of days mean < 2°C | Harvest |
| `temperature_2m_mean` | `harvest_GDD` | °C·days | Sum of max(temp_mean − 15.5, 0) | Harvest |
| `precipitation_sum` | `harvest_total_rain` | mm | Sum over stage window | Harvest |
| `precipitation_sum` | `harvest_rainy_days` | days | Count of days > 1mm | Harvest |
| `precipitation_sum` | `harvest_dry_days` | days | Count of days < 1mm | Harvest |
| `precipitation_sum` | `harvest_max_dry_streak` | days | Longest consecutive dry day run | Harvest |
| `relative_humidity_2m_mean` | `harvest_humidity_mean` | % | Mean over stage window | Harvest |
| `wind_speed_10m_max` | `harvest_wind_mean` | km/h | Mean over stage window | Harvest |

## 5. ML Architecture

| Model | Type | Target | Algorithm | Library |
|---|---|---|---|---|
| Planting risk classifier | Binary classification | `planting_risk_label` (0/1) | Random Forest | scikit-learn |
| Growing risk classifier | Binary classification | `growing_risk_label` (0/1) | Random Forest | scikit-learn |
| Boll forming risk classifier | Binary classification | `boll_risk_label` (0/1) | Random Forest | scikit-learn |
| Harvest risk classifier | Binary classification | `harvest_risk_label` (0/1) | Random Forest | scikit-learn |
| Yield regressor | Regression | `yield_anomaly` (tonnes) | XGBoost / Ridge | XGBoost / scikit-learn |

**Validation strategy:** Walk-forward cross-validation — each fold trains on all years before the test year and predicts only forward. Prevents any future data leaking into training.

**Key design decision:** The yield model predicts `yield_anomaly` (deviation from district historical mean) rather than raw yield. This removes district-baseline noise (~70% of total variance) so the model learns purely weather-driven year-to-year fluctuations.

## 6. Project Structure

```
m5-project-weather-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py           # Paths, locations, agronomic constants
│   ├── ingestion.py        # Weather API fetch + cotton Excel load
│   ├── cleaning.py         # Data cleaning and alignment
│   ├── features.py         # Growth stage feature engineering
│   ├── quality_checks.py   # Pre-modelling validation checks
│   ├── models.py           # ML training and 2025 predictions
│   ├── reports.py          # EDA charts and forecast dashboard
│   ├── database.py         # DuckDB storage layer helpers
│   └── pipeline.py         # Full end-to-end orchestration
├── notebooks/
│   ├── day_01_exploration.ipynb
│   ├── day_02_ingestion.ipynb
│   ├── day_03_database.ipynb
│   ├── day_04_cleaning_features.ipynb
│   ├── day_05_checkpoint.ipynb
│   ├── day_06_eda.ipynb
│   ├── day_07_statistics.ipynb
│   └── day_08_modeling.ipynb
├── data/
│   └── raw/                # Raw data (gitignored)
├── reports/
│   ├── figures/            # Saved visualisations
│   └── data_quality_report.md
└── logs/                   # Pipeline logs (gitignored)
```

## 7. Setup & Run Order

```bash
git clone <repo_url>
cd m5-project-weather-pipeline
pip install -r requirements.txt
```

Place `cotton_dataset.xls` in `data/raw/` (not tracked by git).

**Run each step individually:**
```bash
python src/ingestion.py       # Fetch weather + load cotton → DuckDB
python src/cleaning.py        # Clean both datasets → DuckDB
python src/features.py        # Build growth stage features → DuckDB
python src/quality_checks.py  # Validate before modelling
python src/models.py          # Train models + predict 2025
python src/reports.py         # Generate charts and CSV report
```

**Or run everything at once:**
```bash
python src/pipeline.py
```

## 8. Limitations

## 8. Limitations

| Limitation | Impact |
|---|---|
| Weather data only | Soil nutrients, irrigation volumes, fertilizer use not modelled |
| 375 observations | Small dataset — robust CV and simple models required to avoid overfitting |
| Post-2021 distribution shift | Test R² negative on 2022–2024 — likely reflects agricultural policy changes not visible in weather |
| 4 weather variables | Reduced variable set means heat stress proxied from temp_mean rather than temp_max |
| Partial 2026 data | 2026 excluded — only ~4 months available at time of writing |