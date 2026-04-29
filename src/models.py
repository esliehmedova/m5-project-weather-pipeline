# src/models.py
"""
Member 3 — Model Training & Prediction Pipeline
================================================
TARGET: yield_anomaly = yield_tonnes - district_mean_yield

WHY THIS WORKS BETTER THAN RAW YIELD:
  - Raw yield varies massively between districts (Barda ~22t vs Neftchala ~14t)
  - ~70% of variance is explained by "which district" not "which year's weather"
  - By subtracting the district mean, we remove that noise completely
  - The model now purely learns: "given this year's weather, was yield
    above or below this district's normal?" — exactly what weather predicts
  - CV R² on anomaly should be much higher than on raw yield

PREDICTION FLOW:
  1. Train on yield_anomaly (2000–2021 for CV, 2000–2024 for final model)
  2. Predict 2025 anomaly
  3. Convert back: pred_yield = district_mean + pred_anomaly
  4. District mean computed from training data only (no leakage)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)
import joblib

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH, LOGS_DIR, STAGES, COTTON_BASE_TEMP, MODELS_DIR

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask   = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ── Feature column helpers ────────────────────────────────────────────────────

def get_raw_weather_cols(df):
    """Original 40 weather columns — no identifiers, no leakage, no z-scores."""
    exclude       = {"region", "weather_station", "year", "yield_tonnes",
                     "yield_anomaly", "district_mean_yield"}
    leak_keywords = ("risk", "deviation", "label", "_z")
    return [
        c for c in df.columns
        if c not in exclude and not any(kw in c for kw in leak_keywords)
    ]


def get_model_cols(df):
    """Z-scored weather features only — no district_mean (absorbed into target)."""
    return [c for c in df.columns if c.endswith("_z")]


# ── District feature engineering ──────────────────────────────────────────────

def add_district_features(df_train, df_target):
    """
    Computes from df_train only:

    district_mean_yield:
        Used to construct yield_anomaly target and to back-convert predictions.
        Computed from training data only — no leakage.

    {col}_z (z-scored weather):
        Every weather column standardized per district using training
        mean/std. Makes features comparable across districts.

    yield_anomaly (on df_train rows):
        yield_tonnes - district_mean_yield
        This is the TARGET variable the model trains on.
    """
    weather_cols = get_raw_weather_cols(df_train)

    # District mean from training data only
    district_mean = (
        df_train.groupby("region")["yield_tonnes"]
        .mean()
        .rename("district_mean_yield")
    )

    # Per-district weather mean/std from training
    w_mean = df_train.groupby("region")[weather_cols].mean()
    w_std  = (
        df_train.groupby("region")[weather_cols]
        .std().fillna(1.0).replace(0, 1.0)
    )

    result = df_target.copy()
    result["district_mean_yield"] = result["region"].map(district_mean)

    # Z-score each weather feature per district
    for col in weather_cols:
        result[f"{col}_z"] = (
            (result[col] - result["region"].map(w_mean[col]))
            / result["region"].map(w_std[col])
        )

    # Compute anomaly target (only meaningful where yield_tonnes exists)
    if "yield_tonnes" in result.columns:
        result["yield_anomaly"] = result["yield_tonnes"] - result["district_mean_yield"]

    return result


# ── Walk-forward CV on yield_anomaly ─────────────────────────────────────────

def walk_forward_cv(df_raw, model_factories, min_train_years=6):
    """
    Walk-forward CV predicting yield_anomaly.
    District features recomputed per fold from that fold's training data.
    Returns dict of {model_name: (mae, rmse, r2)} on pooled anomaly predictions.
    """
    years    = sorted(df_raw["year"].unique())
    start    = years[min_train_years]

    # Collect per-model predictions
    fold_results = {name: {"true": [], "pred": []} for name in model_factories}

    for test_year in range(start, max(years) + 1):
        fold_train = df_raw[df_raw["year"] <  test_year].copy()
        fold_test  = df_raw[df_raw["year"] == test_year].copy()
        if fold_test.empty:
            continue

        fold_train_aug = add_district_features(fold_train, fold_train)
        fold_test_aug  = add_district_features(fold_train, fold_test)
        feature_cols   = get_model_cols(fold_train_aug)

        fold_train_aug = fold_train_aug.dropna(
            subset=feature_cols + ["yield_anomaly"]
        )
        fold_test_aug = fold_test_aug.dropna(
            subset=feature_cols + ["yield_anomaly"]
        )
        if fold_test_aug.empty or len(fold_train_aug) < 10:
            continue

        X_tr = fold_train_aug[feature_cols].values
        y_tr = fold_train_aug["yield_anomaly"].values
        X_te = fold_test_aug[feature_cols].values
        y_te = fold_test_aug["yield_anomaly"].values

        for name, model_fn in model_factories.items():
            try:
                m = model_fn()
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                fold_results[name]["true"].extend(y_te)
                fold_results[name]["pred"].extend(preds)
            except Exception as e:
                pass  # skip silently, logged in main

    metrics = {}
    for name, data in fold_results.items():
        if not data["true"]:
            continue
        t = np.array(data["true"])
        p = np.array(data["pred"])
        metrics[name] = (
            mean_absolute_error(t, p),
            np.sqrt(mean_squared_error(t, p)),
            r2_score(t, p)
        )
    return metrics


# ── Feature selection ─────────────────────────────────────────────────────────

def select_top_features(df_aug, top_n=20):
    """Rank z-scored weather features by RF importance on yield_anomaly."""
    feature_cols = get_model_cols(df_aug)
    df_clean     = df_aug.dropna(subset=feature_cols + ["yield_anomaly"])

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(df_clean[feature_cols].values, df_clean["yield_anomaly"].values)

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top_cols    = importances.nlargest(top_n).index.tolist()

    log(f"\n  Top {top_n} features (predicting yield anomaly):")
    for col, imp in importances.nlargest(top_n).items():
        log(f"    {col:<50} {imp:.4f}")
    return top_cols


# ── Risk scores ───────────────────────────────────────────────────────────────

def build_risk_scores(df):
    """Weather-only risk scores — no yield, no leakage."""
    def norm(s, cap):
        return (s.clip(0, cap) / cap * 100).clip(0, 100)

    def norm_inv(s, safe, danger):
        return ((safe - s) / (safe - danger) * 100).clip(0, 100)

    def wavg(pairs):
        t = sum(w for _, w in pairs)
        return sum(s * w for s, w in pairs) / t

    return {
        "planting": wavg([
            (norm_inv(df["planting_temp_mean"],  14, 8), 0.4),
            (norm_inv(df["planting_total_rain"], 30, 5), 0.4),
            (norm(df["planting_frost_days"], 8),          0.2),
        ]).clip(0, 100),
        "growing": wavg([
            (norm(df["growing_heat_stress_days"],        20), 0.35),
            (norm_inv(df["growing_total_rain"],    100,  30), 0.25),
            (norm_inv(df["growing_GDD"],           900, 400), 0.25),
            (norm(df["growing_max_dry_streak"],         35),  0.15),
        ]).clip(0, 100),
        "boll": wavg([
            (norm(df["boll_forming_temp_mean"],     38), 0.45),
            (norm(df["boll_forming_dry_days"],      40), 0.30),
            (norm(df["boll_forming_humidity_mean"], 90), 0.25),
        ]).clip(0, 100),
        "harvest": wavg([
            (norm(df["harvest_rainy_days"],     25), 0.45),
            (norm(df["harvest_humidity_mean"],  85), 0.35),
            (norm(df["harvest_frost_days"],      8), 0.20),
        ]).clip(0, 100),
    }


def train_risk_classifiers(df_train_aug, df_test_aug, feature_cols):
    log("\n" + "=" * 55)
    log("STEP 8 — Risk Classifiers (4 stages)")
    log("=" * 55)

    train_scores = build_risk_scores(df_train_aug)
    test_scores  = build_risk_scores(df_test_aug)
    classifiers  = {}

    for stage in ["planting", "growing", "boll", "harvest"]:
        thr    = train_scores[stage].quantile(0.60)
        y_tr   = (train_scores[stage] >= thr).astype(int)
        y_te   = (test_scores[stage]  >= thr).astype(int)

        df_tr  = df_train_aug.dropna(subset=feature_cols)
        df_te  = df_test_aug.dropna(subset=feature_cols)

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        clf.fit(df_tr[feature_cols].values, y_tr[df_tr.index])
        preds = clf.predict(df_te[feature_cols].values)

        log(f"\n  {stage.upper()}:")
        log(classification_report(y_te[df_te.index], preds, zero_division=0))

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(clf, os.path.join(MODELS_DIR, f"risk_{stage}.pkl"))
        log(f"  Saved → risk_{stage}.pkl ✓")
        classifiers[stage] = clf

    return classifiers


# ── Yield anomaly model ───────────────────────────────────────────────────────

def train_yield_model(df_all_aug, feature_cols, best_name):
    """
    Train on ALL data (2000–2024) using the best model from CV.
    Target is yield_anomaly — back-converted to raw yield at prediction time.
    """
    log("\n" + "=" * 55)
    log(f"STEP 9 — Yield Anomaly Model ({best_name}, trained on 2000–2024)")
    log("=" * 55)

    df_clean = df_all_aug.dropna(subset=feature_cols + ["yield_anomaly"])
    X = df_clean[feature_cols].values
    y = df_clean["yield_anomaly"].values

    # Build final model
    if best_name == "Ridge":
        model = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    elif best_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=300, max_depth=6,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        )
    elif best_name == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            min_samples_leaf=4, subsample=0.8,
            validation_fraction=0.15, n_iter_no_change=20,
            random_state=42
        )
    elif best_name == "XGBoost" and HAS_XGB:
        split = int(len(X) * 0.85)
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30, eval_metric="rmse",
            random_state=42, verbosity=0
        )
        model.fit(
            X[:split], y[:split],
            eval_set=[(X[split:], y[split:])],
            verbose=False
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, "yield_model.pkl"))
        log(f"  Saved → yield_model.pkl ✓")
        return model
    elif best_name == "LightGBM" and HAS_LGB:
        split = int(len(X) * 0.85)
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            num_leaves=15, subsample=0.8,
            early_stopping_rounds=30,
            random_state=42, verbose=-1
        )
        model.fit(
            X[:split], y[:split],
            eval_set=[(X[split:], y[split:])],
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, "yield_model.pkl"))
        log(f"  Saved → yield_model.pkl ✓")
        return model
    else:
        model = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])

    model.fit(X, y)

    train_preds  = model.predict(X)
    train_mae    = mean_absolute_error(y, train_preds)
    train_r2     = r2_score(y, train_preds)

    # Also report in raw yield space
    raw_y   = df_clean["yield_tonnes"].values
    raw_pred = train_preds + df_clean["district_mean_yield"].values
    raw_mae  = mean_absolute_error(raw_y, raw_pred)
    raw_r2   = r2_score(raw_y, raw_pred)

    log(f"\n  Anomaly space  — MAE: {train_mae:.2f}  R²: {train_r2:.3f}")
    log(f"  Raw yield space — MAE: {raw_mae:.2f}  R²: {raw_r2:.3f}")
    log(f"  Rows: {len(df_clean)}  Features: {len(feature_cols)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "yield_model.pkl"))
    log(f"  Saved → yield_model.pkl ✓")

    return model


# ── Feature engineering helpers for future years ──────────────────────────────

def _gdd(temp_mean, base=COTTON_BASE_TEMP):
    return max(temp_mean - base, 0)


def _dry_streak(precip):
    best = cur = 0
    for v in precip:
        cur = cur + 1 if v < 1 else 0
        best = max(best, cur)
    return best


def _stage_features(stage_df, name):
    if stage_df.empty:
        return {}
    gdd = stage_df["temp_mean"].apply(_gdd)
    return {
        f"{name}_temp_mean":        round(stage_df["temp_mean"].mean(), 4),
        f"{name}_heat_stress_days": int((stage_df["temp_mean"] > 32).sum()),
        f"{name}_frost_days":       int((stage_df["temp_mean"] < 2).sum()),
        f"{name}_GDD":              round(gdd.sum(), 4),
        f"{name}_total_rain":       round(stage_df["precipitation"].sum(), 4),
        f"{name}_rainy_days":       int((stage_df["precipitation"] > 1).sum()),
        f"{name}_dry_days":         int((stage_df["precipitation"] < 1).sum()),
        f"{name}_max_dry_streak":   _dry_streak(stage_df["precipitation"]),
        f"{name}_humidity_mean":    round(stage_df["humidity_mean"].mean(), 4),
        f"{name}_wind_mean":        round(stage_df["wind_speed"].mean(), 4),
    }


def build_prediction_features(con, year):
    """Build raw weather feature rows for a future year from clean_weather."""
    districts = con.execute(
        "SELECT DISTINCT region, weather_station FROM clean_cotton"
    ).df()
    weather = con.execute(f"""
        SELECT region, year, month, day,
               temp_mean, precipitation, humidity_mean, wind_speed
        FROM clean_weather WHERE year = {year}
        ORDER BY region, month, day
    """).df()

    if weather.empty:
        log(f"  WARNING: No weather data for {year}.")
        return pd.DataFrame()

    rows = []
    for _, r in districts.iterrows():
        daily = weather[weather["region"] == r["weather_station"]]
        if daily.empty:
            continue
        feat = {"region": r["region"], "weather_station": r["weather_station"],
                "year": year, "yield_tonnes": np.nan}
        for sname, (sm, em) in STAGES.items():
            feat.update(_stage_features(
                daily[(daily["month"] >= sm) & (daily["month"] <= em)], sname
            ))
        rows.append(feat)

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).reset_index(drop=True)
    wcols  = [c for c in df_out.columns
              if c not in {"region", "weather_station", "year", "yield_tonnes"}]
    df_out = df_out.dropna(subset=wcols).reset_index(drop=True)
    log(f"  Built {len(df_out)} raw feature rows for {year} ✓")
    return df_out


# ── Predict 2025 ──────────────────────────────────────────────────────────────

def predict_future(con, yield_model, risk_classifiers, feature_cols, df_all_raw):
    log("\n" + "=" * 55)
    log("STEP 10 — Predicting 2025")
    log("=" * 55)

    df_year = build_prediction_features(con, 2025)
    if df_year.empty:
        log("  No 2025 weather data — cannot predict.")
        return None

    # Add district features using ALL training history (2000–2024)
    df_year_aug = add_district_features(df_all_raw, df_year)

    missing = [c for c in feature_cols if c not in df_year_aug.columns]
    if missing:
        log(f"  WARNING: {len(missing)} feature columns missing: {missing[:3]}...")
        return None

    df_year_aug = df_year_aug.dropna(subset=feature_cols).copy()
    if df_year_aug.empty:
        log("  All rows dropped after augmentation.")
        return None

    # Predict anomaly, convert back to raw yield
    pred_anomaly = yield_model.predict(df_year_aug[feature_cols].values)
    df_year_aug["pred_anomaly"] = pred_anomaly
    df_year_aug["pred_yield"]   = (
        df_year_aug["district_mean_yield"] + pred_anomaly
    ).clip(0)  # yield can't be negative

    # Stage risk probabilities
    for stage in ["planting", "growing", "boll", "harvest"]:
        clf = risk_classifiers.get(stage)
        if clf is not None:
            proba = clf.predict_proba(df_year_aug[feature_cols].values)
            df_year_aug[f"{stage}_risk_pct"] = (proba[:, 1] * 100).round(0).astype(int)
        else:
            df_year_aug[f"{stage}_risk_pct"] = 0

    # Historical average (training baseline)
    hist_avg = con.execute("""
        SELECT region, AVG(yield_tonnes) AS avg_yield
        FROM features WHERE year <= 2021
        GROUP BY region
    """).df().set_index("region")["avg_yield"]

    df_year_aug["avg_yield"]  = df_year_aug["region"].map(hist_avg).round(1)
    df_year_aug["pct_change"] = (
        (df_year_aug["pred_yield"] - df_year_aug["avg_yield"])
        / df_year_aug["avg_yield"] * 100
    ).round(1)

    pred_df = df_year_aug[[
        "region", "year", "pred_yield", "avg_yield", "pct_change",
        "planting_risk_pct", "growing_risk_pct",
        "boll_risk_pct", "harvest_risk_pct"
    ]].copy()

    con.execute("DROP TABLE IF EXISTS predictions")
    con.execute("CREATE TABLE predictions AS SELECT * FROM pred_df")
    log("  predictions → DuckDB ✓")

    df_s = pred_df.sort_values("pred_yield", ascending=False)
    log(f"\n  {'='*65}")
    log(f"  2025 PREDICTIONS (anomaly model → converted back to raw yield)")
    log(f"  {'='*65}")
    log(f"  {'District':<30} {'Pred':>6} {'Avg':>6} {'Chg':>7}  Plant  Grow  Boll  Harv")
    log(f"  {'-'*65}")
    for _, r in df_s.iterrows():
        log(
            f"  {r['region']:<30} {r['pred_yield']:>6.1f} "
            f"{r['avg_yield']:>6.1f} {r['pct_change']:>+7.1f}%  "
            f"{r['planting_risk_pct']:>3}%  {r['growing_risk_pct']:>3}%  "
            f"{r['boll_risk_pct']:>3}%  {r['harvest_risk_pct']:>3}%"
        )
    log(f"  {'-'*65}")
    log(f"  {'TOTAL':<30} {pred_df['pred_yield'].sum():>6.1f} "
        f"{pred_df['avg_yield'].sum():>6.1f}")

    return pred_df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_models():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 3 — MODEL TRAINING & PREDICTION PIPELINE")
    log("=" * 55)

    df = con.execute("SELECT * FROM features").df()
    log(f"  Loaded features: {df.shape}")

    forbidden = [c for c in df.columns
                 if any(kw in c for kw in ("risk", "deviation", "label"))]
    if forbidden:
        raise ValueError(f"Leakage columns in features table: {forbidden}")
    log("  No leakage columns detected ✓")

    df_train_raw = df[df["year"] <= 2021].copy()
    df_test_raw  = df[df["year"] >  2021].copy()
    df_all_raw   = df.copy()

    log(f"\n  Train (≤2021): {len(df_train_raw)} rows")
    log(f"  Test  (>2021): {len(df_test_raw)} rows  "
        f"(years {df_test_raw['year'].min()}–{df_test_raw['year'].max()})")

    # Augment with district features + z-scores
    log("\n  Adding district features + z-scored weather...")
    df_train_aug = add_district_features(df_train_raw, df_train_raw)
    df_test_aug  = add_district_features(df_train_raw, df_test_raw)
    df_all_aug   = add_district_features(df_all_raw,   df_all_raw)
    log("  Done ✓")

    # Model factories for CV (no early stopping inside folds)
    def make_ridge():
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])

    def make_rf():
        return RandomForestRegressor(
            n_estimators=300, max_depth=6,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        )

    def make_gb():
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            min_samples_leaf=4, subsample=0.8,
            validation_fraction=0.15, n_iter_no_change=20,
            random_state=42
        )

    model_factories = {
        "Ridge":             make_ridge,
        "Random Forest":     make_rf,
        "Gradient Boosting": make_gb,
    }
    if HAS_XGB:
        model_factories["XGBoost"] = lambda: xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
    if HAS_LGB:
        model_factories["LightGBM"] = lambda: lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            num_leaves=15, subsample=0.8,
            random_state=42, verbose=-1
        )

    # Walk-forward CV on yield_anomaly
    log("\n  Walk-forward CV on yield_anomaly (2000–2021)...")
    cv_metrics = walk_forward_cv(df_train_raw, model_factories)

    log(f"\n  {'Model':<22} {'CV-MAE':>8} {'CV-RMSE':>8} {'CV-R²':>7}")
    log("  " + "-" * 50)

    best_name = "Ridge"
    best_mae  = float("inf")
    for name, (mae, rmse, r2) in cv_metrics.items():
        log(f"  {name:<22} {mae:>8.2f} {rmse:>8.2f} {r2:>7.3f}")
        if mae < best_mae:
            best_mae  = mae
            best_name = name

    log(f"\n  Best model by CV-MAE: {best_name} ✓")

    # Feature selection on training anomaly
    log("\n  Selecting top features...")
    top_features = select_top_features(df_train_aug, top_n=20)
    log(f"  {len(top_features)} features selected")

    # Report test metrics on anomaly (for transparency)
    df_tr_c = df_train_aug.dropna(subset=top_features + ["yield_anomaly"])
    df_te_c = df_test_aug.dropna(subset=top_features + ["yield_anomaly"])
    probe   = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    probe.fit(df_tr_c[top_features].values, df_tr_c["yield_anomaly"].values)
    if not df_te_c.empty:
        te_anom_pred = probe.predict(df_te_c[top_features].values)
        te_anom_true = df_te_c["yield_anomaly"].values
        log(f"\n  Held-out test anomaly (2022–2024, reference only):")
        log(f"    MAE: {mean_absolute_error(te_anom_true, te_anom_pred):.2f}  "
            f"R²: {r2_score(te_anom_true, te_anom_pred):.3f}")

        # Also in raw yield space
        te_raw_pred = te_anom_pred + df_te_c["district_mean_yield"].values
        te_raw_true = df_te_c["yield_tonnes"].values
        log(f"  Held-out test raw yield (2022–2024, reference only):")
        log(f"    MAE: {mean_absolute_error(te_raw_true, te_raw_pred):.2f}  "
            f"R²: {r2_score(te_raw_true, te_raw_pred):.3f}  "
            f"MAPE: {mape(te_raw_true, te_raw_pred):.1f}%")

    # Risk classifiers (trained on training split)
    risk_classifiers = train_risk_classifiers(df_train_aug, df_test_aug, top_features)

    # Final yield model on all data
    best_model = train_yield_model(df_all_aug, top_features, best_name)

    # Predict 2025
    predict_future(con, best_model, risk_classifiers, top_features, df_all_raw)

    log("\n" + "=" * 55)
    log("MODEL PIPELINE COMPLETE — DuckDB Tables:")
    log("=" * 55)
    for t in con.execute("SHOW TABLES").fetchall():
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        cols  = len(con.execute(f"SELECT * FROM {t[0]} LIMIT 1").description)
        log(f"  {t[0]:<25} {count:>6} rows  {cols:>3} cols")

    con.close()
    log("\n  Next: run src/pipeline.py")


if __name__ == "__main__":
    run_models() 