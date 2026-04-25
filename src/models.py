# src/models.py  (this replaces the old train_models.py + predict.py)
import pandas as pd
import numpy as np
import duckdb
import joblib
import os
import sys
import warnings
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")
from datetime import datetime, date

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DB_PATH, MODELS_DIR, LOGS_DIR, TRAIN_UNTIL_YEAR,
    PREDICT_YEARS, STAGES, LOCATIONS, REGION_TO_WEATHER,
    COTTON_BASE_TEMP
)


def log(msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_feature_cols(df):
    exclude = [
        "region", "weather_station", "year", "yield_tonnes",
        "yield_deviation",
        "planting_risk_score",  "planting_risk_label",
        "growing_risk_score",   "growing_risk_label",
        "boll_risk_score",      "boll_risk_label",
        "harvest_risk_score",   "harvest_risk_label",
        "overall_risk_score",   "overall_risk_label"
    ]
    return [c for c in df.columns if c not in exclude]


def train_risk_models(con, df, feature_cols):
    log("=" * 55)
    log("STEP 8 — Training Risk Classifiers (4 models)")
    log("=" * 55)

    train = df[df["year"] <= TRAIN_UNTIL_YEAR]
    test  = df[df["year"] >  TRAIN_UNTIL_YEAR]
    X_train = train[feature_cols]
    X_test  = test[feature_cols]

    risk_models = {}
    risk_preds  = pd.DataFrame(index=df.index)

    for stage in ["planting", "growing", "boll", "harvest"]:
        label_col = f"{stage}_risk_label"

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
        clf.fit(X_train, train[label_col])
        y_pred = clf.predict(X_test)

        log(f"\n  {stage.upper()} RISK CLASSIFIER:")
        log(f"  {classification_report(test[label_col], y_pred, zero_division=0)}")

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(clf, os.path.join(MODELS_DIR, f"risk_{stage}.pkl"))
        log(f"  Saved → models/risk_{stage}.pkl ✓")

        # Generate risk probability for ALL rows
        probs = clf.predict_proba(df[feature_cols])[:, 1] * 100
        risk_preds[f"{stage}_risk_pred"] = probs
        risk_models[stage] = clf

    return risk_models, risk_preds


def train_yield_model(con, df, feature_cols, risk_preds):
    log("=" * 55)
    log("STEP 9 — Training Yield Prediction Models")
    log("=" * 55)

    # Extra imports (local to avoid breaking rest of file)
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMRegressor

    # Add risk predictions as extra features
    df_full       = pd.concat([df.reset_index(drop=True), risk_preds.reset_index(drop=True)], axis=1)
    risk_cols     = list(risk_preds.columns)
    all_feat_cols = feature_cols + risk_cols

    train = df_full[df_full["year"] <= TRAIN_UNTIL_YEAR]
    test  = df_full[df_full["year"] >  TRAIN_UNTIL_YEAR]

    X_train = train[all_feat_cols]
    X_test  = test[all_feat_cols]
    y_train = train["yield_tonnes"]
    y_test  = test["yield_tonnes"]

    # Scale ONLY for models that need it (SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    models = [

        # Linear models
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Lasso Regression", Lasso(alpha=0.01)),
        ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5)),

        # Tree-based
        ("Random Forest", RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )),

        ("Extra Trees", ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )),

        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )),

        # Boosting libraries
        ("XGBoost", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbosity=0
        )),

        ("LightGBM", LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )),

        # SVM
        ("SVR", SVR(
            kernel="rbf",
            C=100,
            gamma=0.1,
            epsilon=0.1
        )),
    ]

    for name, model in models:

        # Use scaled data ONLY for SVR
        if name == "SVR":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        mape  = np.mean(np.abs((y_test - preds) / (y_test + 1e-5))) * 100

        log(f"\n  {name}:")
        log(f"    MAE:  {mae:.2f} tonnes")
        log(f"    RMSE: {rmse:.2f} tonnes")
        log(f"    R²:   {r2:.3f}")
        log(f"    MAPE: {mape:.1f}%")

        results[name] = (model, rmse, preds)

    # Pick best model (lowest RMSE)
    best_name  = min(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]

    log(f"\n  Best model: {best_name} ✓")

    # Save everything
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "yield_model.pkl"))
    joblib.dump(all_feat_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    joblib.dump(LabelEncoder().fit(df["region"]),
                os.path.join(MODELS_DIR, "label_encoder.pkl"))

    log(f"  Saved → models/yield_model.pkl ✓")

    return best_model, all_feat_cols, best_name

def fetch_forecast_weather(lat, lon, location_name, pred_year):
    """Fetch weather for a prediction year using archive + forecast + climatology."""
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache   = requests_cache.CachedSession(".weather_cache", expire_after=3600)
    session = retry(cache, retries=5, backoff_factor=0.2)
    om      = openmeteo_requests.Client(session=session)

    VARIABLES = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "relative_humidity_2m_max",
        "relative_humidity_2m_min", "wind_speed_10m_max",
        "et0_fao_evapotranspiration"
    ]

    today    = date.today()
    all_dfs  = []

    # Historical part of prediction year
    try:
        end_hist = min(today.strftime("%Y-%m-%d"), f"{pred_year}-12-31")
        resp  = om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": f"{pred_year}-01-01",
                "end_date": end_hist,
                "daily": VARIABLES, "timezone": "Asia/Baku"
            }
        )
        daily = resp[0].Daily()
        df_h  = pd.DataFrame({
            "date":        pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s"),
                end=pd.to_datetime(daily.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "temp_max":      daily.Variables(0).ValuesAsNumpy(),
            "temp_min":      daily.Variables(1).ValuesAsNumpy(),
            "temp_mean":     daily.Variables(2).ValuesAsNumpy(),
            "precipitation": daily.Variables(3).ValuesAsNumpy(),
            "humidity_max":  daily.Variables(4).ValuesAsNumpy(),
            "humidity_min":  daily.Variables(5).ValuesAsNumpy(),
            "wind_speed":    daily.Variables(6).ValuesAsNumpy(),
            "et0":           daily.Variables(7).ValuesAsNumpy(),
        })
        all_dfs.append(df_h)
    except Exception as e:
        log(f"  Warning: archive fetch failed: {e}")

    # Forecast (only relevant for current year)
    if pred_year == today.year:
        try:
            resp  = om.weather_api(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": VARIABLES,
                    "timezone": "Asia/Baku",
                    "forecast_days": 16
                }
            )
            daily = resp[0].Daily()
            df_f  = pd.DataFrame({
                "date":        pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s"),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "temp_max":      daily.Variables(0).ValuesAsNumpy(),
                "temp_min":      daily.Variables(1).ValuesAsNumpy(),
                "temp_mean":     daily.Variables(2).ValuesAsNumpy(),
                "precipitation": daily.Variables(3).ValuesAsNumpy(),
                "humidity_max":  daily.Variables(4).ValuesAsNumpy(),
                "humidity_min":  daily.Variables(5).ValuesAsNumpy(),
                "wind_speed":    daily.Variables(6).ValuesAsNumpy(),
                "et0":           daily.Variables(7).ValuesAsNumpy(),
            })
            all_dfs.append(df_f)
        except Exception as e:
            log(f"  Warning: forecast fetch failed: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined      = pd.concat(all_dfs).drop_duplicates("date").sort_values("date")
    covered       = set(pd.to_datetime(combined["date"]).dt.date)
    all_dates     = pd.date_range(f"{pred_year}-01-01", f"{pred_year}-12-31")
    missing_dates = [d for d in all_dates if d.date() not in covered]

    # Fill missing with climatology from past years
    if missing_dates:
        from config import RAW_WEATHER_DIR
        import glob
        path = os.path.join(RAW_WEATHER_DIR, f"{location_name.lower()}_daily.csv")
        if os.path.exists(path):
            hist = pd.read_csv(path, parse_dates=["date"])
            for d in missing_dates:
                same = hist[
                    (hist["date"].dt.month == d.month) &
                    (hist["date"].dt.day   == d.day)
                ]
                if not same.empty:
                    avg      = same[["temp_max","temp_min","temp_mean",
                                     "precipitation","humidity_max",
                                     "humidity_min","wind_speed","et0"]].mean()
                    avg["date"] = pd.Timestamp(d)
                    combined = pd.concat([combined, pd.DataFrame([avg])])\
                        .drop_duplicates("date").sort_values("date")

    combined["month"]  = pd.to_datetime(combined["date"]).dt.month
    combined["year"]   = pred_year
    combined["day"]    = pd.to_datetime(combined["date"]).dt.day
    combined["region"] = location_name
    return combined


def compute_stage_features_for_predict(stage_df, stage_name):
    """Same as features.py but used for prediction rows."""
    from features import compute_stage_features
    return compute_stage_features(stage_df, stage_name)


def run_predictions(con, risk_models, yield_model, all_feat_cols):
    log("=" * 55)
    log(f"STEP 10 — Predicting for years: {PREDICT_YEARS}")
    log("=" * 55)

    historical = con.execute("""
        SELECT region, year, yield_tonnes, weather_station
        FROM clean_cotton
    """).df()

    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    all_results = []

    for pred_year in PREDICT_YEARS:
        log(f"\n  Fetching weather for {pred_year}...")
        weather_year = {}
        for station, info in LOCATIONS.items():
            weather_year[station] = fetch_forecast_weather(
                info["lat"], info["lon"], station, pred_year
            )
            log(f"    {station}: {len(weather_year[station])} days")

        log(f"\n  Running predictions for {pred_year}...")
        for region, station in REGION_TO_WEATHER.items():
            if region not in historical["region"].values:
                continue

            daily = weather_year.get(station, pd.DataFrame())
            if daily.empty:
                continue

            feat = {"region": region, "year": pred_year}
            try:
                feat["region_encoded"] = le.transform([region])[0]
            except:
                continue

            for stage_name, (start_m, end_m) in STAGES.items():
                stage_df = daily[
                    (daily["month"] >= start_m) &
                    (daily["month"] <= end_m)
                ]
                feat.update(compute_stage_features_for_predict(stage_df, stage_name))

            feat_df = pd.DataFrame([feat])
            weather_only = [c for c in all_feat_cols
                            if "risk_pred" not in c and c in feat_df.columns]
            X_weather    = feat_df[weather_only]

            risk_probs = {}
            for stage, model in risk_models.items():
                prob = model.predict_proba(X_weather)[0][1] * 100
                risk_probs[f"{stage}_risk_pred"] = prob

            X_full  = pd.DataFrame([{**feat, **risk_probs}])
            missing = [c for c in all_feat_cols if c not in X_full.columns]
            for c in missing:
                X_full[c] = 0
            X_final = X_full[all_feat_cols]

            pred_yield = float(yield_model.predict(X_final)[0])

            hist_r    = historical[historical["region"] == region]
            hist_avg  = float(hist_r["yield_tonnes"].mean())
            pct_chg   = (pred_yield - hist_avg) / hist_avg * 100

            all_results.append({
                "region":          region,
                "year":            pred_year,
                "predicted_yield": round(pred_yield, 1),
                "historical_avg":  round(hist_avg, 1),
                "pct_vs_avg":      round(pct_chg, 1),
                "planting_risk":   round(risk_probs["planting_risk_pred"], 1),
                "growing_risk":    round(risk_probs["growing_risk_pred"], 1),
                "boll_risk":       round(risk_probs["boll_risk_pred"], 1),
                "harvest_risk":    round(risk_probs["harvest_risk_pred"], 1),
                "overall_risk":    round(np.mean(list(risk_probs.values())), 1),
            })

    results_df = pd.DataFrame(all_results)
    con.execute("DROP TABLE IF EXISTS predictions")
    con.execute("CREATE TABLE predictions AS SELECT * FROM results_df")
    log("\n  predictions → DuckDB ✓")

    # Print summary table
    for yr in PREDICT_YEARS:
        yr_df = results_df[results_df["year"] == yr].sort_values(
            "predicted_yield", ascending=False
        )
        log(f"\n  {'='*65}")
        log(f"  {yr} PREDICTIONS")
        log(f"  {'='*65}")
        log(f"  {'District':<28} {'Pred':>6} {'Avg':>6} {'Chg':>7}  "
            f"{'Plant':>5} {'Grow':>5} {'Boll':>5} {'Harv':>5}")
        log(f"  {'-'*65}")
        for _, r in yr_df.iterrows():
            log(f"  {r['region']:<28} {r['predicted_yield']:>6.1f} "
                f"{r['historical_avg']:>6.1f} {r['pct_vs_avg']:>+7.1f}%  "
                f"{r['planting_risk']:>4.0f}% {r['growing_risk']:>4.0f}% "
                f"{r['boll_risk']:>4.0f}% {r['harvest_risk']:>4.0f}%")
        log(f"  {'-'*65}")
        log(f"  {'TOTAL':<28} {yr_df['predicted_yield'].sum():>6.1f} "
            f"{yr_df['historical_avg'].sum():>6.1f}")

    return results_df


def run_models():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 3 — MODEL TRAINING & PREDICTION PIPELINE")
    log("=" * 55)

    # Load feature data via SQL
    df = con.execute("SELECT * FROM features_with_risk").df()
    log(f"  Loaded features_with_risk: {df.shape}")

    # Encode region
    # REPLACE with (just move makedirs ABOVE joblib.dump)
    le = LabelEncoder()
    df["region_encoded"] = le.fit_transform(df["region"])
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    feature_cols = get_feature_cols(df)
    log(f"  Feature columns: {len(feature_cols)}")

    risk_models, risk_preds         = train_risk_models(con, df, feature_cols)
    best_model, all_feat_cols, name = train_yield_model(con, df, feature_cols, risk_preds)
    run_predictions(con, risk_models, best_model, all_feat_cols)

    log("\n" + "=" * 55)
    log("MODEL PIPELINE COMPLETE — DuckDB Tables:")
    log("=" * 55)
    for t in con.execute("SHOW TABLES").fetchall():
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        cols  = len(con.execute(f"SELECT * FROM {t[0]} LIMIT 1").description)
        log(f"  {t[0]:<25} {count:>6} rows  {cols:>3} cols")

    con.close()
    log("\n  Next: Member 4 runs src/reports.py")


if __name__ == "__main__":
    from quality_checks import run_quality_checks
    run_quality_checks()
    run_models()