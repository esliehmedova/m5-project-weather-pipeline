# Risk Masters - Evaluation Review

---

## Executive Summary

You’ve delivered a sophisticated cotton yield and risk prediction system with impressive methodological rigor. Your project integrates 25 years of historical data across 15 Azerbaijani districts using a novel approach: predicting yield anomalies rather than raw yields, which effectively removes ~70% of baseline variance. The use of Nested Cross-Validation with Leave-One-Year-Out (LOYO) and single-feature selection per fold demonstrates an advanced understanding of small-sample learning. Furthermore, your comprehensive data quality report and explicit acknowledgment of limitations—such as negative R² in test sets due to real-world distribution shifts—shows a level of scientific maturity rarely seen in bootcamp projects.

---

## Detailed Assessment

### 1. Pipeline Completeness

**What's Implemented:**
- End-to-end pipeline with clear module separation (8 src modules)
- Data ingestion from both Open-Meteo API and Excel cotton dataset
- DuckDB database for structured storage
- Incremental fetching with skip logic for already-fetched stations
- Rate limit handling (20-second pause between API calls)
- Automated logging with timestamps

**Strengths:**
- Handles two distinct data sources (API weather + Excel cotton production)
- Smart caching: skips already-fetched weather stations
- Comprehensive pipeline orchestration
- Rate limiting for API etiquette

**Areas for Consideration:**
- Is there a single command to run the full pipeline, or do notebooks need to be executed sequentially?
- What triggers the pipeline to run (scheduled, manual, event-based)?

---

### 2. Data Quality Analysis

**What's Implemented:**
- **Comprehensive Data Quality Report** (`reports/data_quality_report.md`)
- 7 quality checks in `src/quality_checks.py`:
  - No nulls in features
  - Train/test split integrity (no leakage)
  - Yield range sanity (0-500 tonnes)
  - Year range coverage
  - Feature column count (≥44 expected)
  - GDD non-negative validation
  - All 15 districts present
- Cleaning steps well-documented:
  - Temperature bounds: >60°C or <-40°C → NaN
  - Precipitation bounds: negative → 0
  - Humidity bounds: clip to 0-100%
  - Gap interpolation (max 3 consecutive days)
- Honest limitation disclosure in quality report

**Strengths:**
- **Best-in-class data quality documentation** - 156-line detailed report
- Districts with any nulls dropped entirely (no partial imputation bias)
- Physical plausibility checks on weather data
- Explicit leakage prevention (risk columns excluded from training table)
- Honest acknowledgment: "Test R² negative — likely reflects real-world policy/practice changes"

**Areas for Consideration:**
- Were there any outliers beyond the physical bounds that were analyzed?
- What percentage of weather data required interpolation?

---

### 3. Statistical Reasoning

**What's Implemented:**
- **Walk-forward cross-validation**: Each fold trains on all years before test year
- **Nested CV with LOYO (Leave-One-Year-Out)**: Inner loop for feature selection, outer loop for validation
- Single best feature selected per fold (parsimonious modeling)
- Ridge regression with CV for alpha selection ([0.1, 1, 10, 100])
- Anomaly-based prediction (yield - district_mean) removes ~70% variance
- Train/test split: 2000-2021 train, 2022-2024 test

**Strengths:**
- **Nested CV with single-feature selection** is methodologically sophisticated for a bootcamp project
- Recognition that small dataset (375 rows) requires robust CV
- Yield anomaly approach is clever - focuses model on weather-driven fluctuations
- Temporal validation prevents future leakage
- Ridge regularization with cross-validated alpha

**Areas for Consideration:**
- What specific hypothesis tests were performed in Day 7 statistics?
- Were the assumptions of Ridge regression checked (multicollinearity, etc.)?
- The negative test R² suggests the model doesn't generalize to post-2021 - was this analyzed further?

---

### 4. Prediction Model

**What's Implemented:**
- **5 Total Models**:
  - 4 Random Forest risk classifiers (one per growth stage: planting, growing, boll forming, harvest)
  - 1 Ridge yield regressor for anomaly prediction
- **40 Features** (10 per stage × 4 stages):
  - Temperature: mean, min, max, heat_stress_days, frost_days, GDD
  - Precipitation: total_rain, rainy_days, dry_days, max_dry_streak
  - Humidity: mean
  - Wind: mean
- **Growth Stage Windows** (DOY-based):
  - Planting: DOY 60-120 (Mar-Apr)
  - Growing: DOY 121-243 (May-Aug)
  - Boll Forming: DOY 244-288 (Sep-Oct)
  - Harvest: DOY 244-334 (Sep-Nov)
- **2025 Predictions** with risk percentages per stage
- **Model Artifacts**: 9 pickle files (risk models, yield model, scaler, encoder)

**Strengths:**
- **Agronomically-informed feature engineering** tied to cotton growth stages
- GDD (Growing Degree Days) calculation with base 15.5°C is agriculturally appropriate
- Risk scores (0-100%) per stage with clear interpretation
- Separate models for each risk type allows targeted intervention advice
- Yield predictions for 2025 with percentage changes from historical averages

**Areas for Consideration:**
- The yield model predicts anomaly, but the final yield is region_mean + anomaly. How accurate are the 2025 predictions?
- Were XGBoost and Ridge compared as mentioned in README? Which performed better?
- What are the R², MAE, RMSE values for the yield model?

---

### 5. Code Quality

**What's Implemented:**
- Modular src/ structure with 8 modules
- Consistent logging function across modules
- Type hints in some places
- Clear docstrings explaining methodology
- Configuration centralized in `src/config.py`
- Feature engineering well-documented with agronomic rationale

**Strengths:**
- **Excellent documentation in code**: "KEY night temp - high nights (>22°C) during flowering cause boll shedding"
- Centralized logging with timestamps
- Clear separation: features table (training) vs features_with_risk (website/EDA)
- Constants defined (COTTON_BASE_TEMP = 15.5°C, STAGES)
- Leakage prevention explicitly coded (exclusion of risk/deviation columns)

**Areas for Consideration:**
- Could benefit from more type hints throughout
- Some comments in features.py reference "DOY-based" but config.py shows month-based stages
- No unit tests visible

---

## Strengths

- **Nested CV with Single-Feature Selection**: Methodologically sophisticated approach for small dataset
- **Yield Anomaly Prediction**: Clever technique removing ~70% district-baseline variance
- **Agronomically-Informed Features**: 40 features tied to cotton growth stages with domain knowledge
- **Comprehensive Data Quality Report**: Best-in-class documentation with honest limitations
- **Risk Stratification**: 4 separate risk models for different growth stages enable targeted interventions
- **2025 Predictions**: Practical forward-looking output with district-level detail
- **Leakage Prevention**: Strict separation of training and risk features
- **Walk-Forward CV**: Proper temporal validation preventing future leakage

## Areas for Consideration (Research Questions)

1. **Test Set Performance**: The quality report notes "Test R² negative — likely reflects real-world policy/practice changes not visible in weather data." How negative was the R²? What does this imply about the model's generalizability?

2. **Model Comparison**: The README mentions training both XGBoost and Ridge. Which performed better? Why was Ridge selected for the final model?

3. **Risk Score Calibration**: Risk scores are percentages (0-100%). How were these calibrated? Are they actual probabilities or normalized scores?

4. **Feature Importance**: With single-feature selection per fold in nested CV, which features were most commonly selected? Do they align with agronomic knowledge?

5. **2025 Prediction Accuracy**: Since 2025 yields are not yet known, how will the model be validated? Is there a plan to compare predictions with actual 2025 yields once available?

6. **Growth Stage Definition**: The config.py shows month-based stages while features.py mentions DOY-based. Which was actually used? Why the difference?

7. **Visualization**: The figures folder is empty. Were EDA visualizations and model performance plots generated? Where are they stored?

---

## Notable Findings

### Duration of Analysis
- **Historical Data**: 25 years (2000-2024)
- **Geographic Coverage**: 15 districts (out of 29 - 14 dropped due to missing data)
- **Total Observations**: 375 district-year combinations (15 districts × 25 years)
- **Weather Data**: ~144,225 daily rows (15 stations × ~9,615 days)
- **Project Duration**: 10 days

### Interesting Methodologies
1. **Anomaly-Based Prediction**: Predicting yield deviation from district mean removes ~70% of variance, focusing on weather-driven fluctuations
2. **Nested LOYO CV**: Outer loop leaves one year out, inner loop selects single best feature - parsimonious and robust for small samples
3. **Growth Stage Feature Engineering**: 40 features organized by agronomic growth phases
4. **Risk Stratification**: Separate binary classifiers for each growth stage risk type
5. **GDD Calculation**: Growing Degree Days with base 15.5°C - standard agronomic metric
6. **Comprehensive Quality Gates**: 7 checks before model training with detailed reporting

### Data Coverage
- **Geographic**: 15 districts across Azerbaijan cotton-growing regions
- **Temporal**: 25 years (2000-2024) for training, 2025 for prediction
- **Features**: 40 weather-based features (10 per growth stage × 4 stages)
- **Sources**: 
  - Cotton: Excel dataset (29 districts, 14 dropped due to nulls)
  - Weather: Open-Meteo API (4 variables: temp_mean, precipitation, humidity, wind_max)

---

## Key Files Reviewed

| File | Purpose |
|------|---------|
| `README.md` | Comprehensive project documentation |
| `src/models.py` | Nested CV, LOYO, Ridge regression, risk classifiers (457 lines) |
| `src/features.py` | Growth stage feature engineering with GDD (295 lines) |
| `src/quality_checks.py` | 7 quality checks with logging (125 lines) |
| `src/config.py` | Paths, locations, agronomic constants (78 lines) |
| `reports/data_quality_report.md` | Excellent 156-line quality documentation |
| `reports/predictions_2025.csv` | 2025 yield and risk predictions for 15 districts |
| `models/` | 9 pickle files (risk models, yield model, scaler) |
| `requirements.txt` | 12 dependencies |

---

*Teacher Assistant: Jannat Samadov*
*Evaluation Date: May 3, 2026*
