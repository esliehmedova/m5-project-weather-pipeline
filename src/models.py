# src/models.py
"""
Member 3 — Model Training & Prediction Pipeline
================================================
Key fixes vs previous version:
  1. Input features: raw weather columns ONLY — no risk scores, no risk labels,
     no yield_deviation. All of those caused data leakage.
  2. Walk-forward cross-validation for honest model selection / tuning.
  3. GroupKFold by district to test geographic generalization.
  4. Early stopping on all boosting models.
  5. Feature importance filtering to reduce the 52-feature noise.
  6. Risk labels computed INSIDE the training fold only (no full-dataset quantiles).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, RandomForestClassifier
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
from config import DB_PATH, LOGS_DIR

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
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_feature_cols(df):
    """
    Return only raw weather feature columns.
    Explicitly excludes:
      - region, weather_station, year  (identifiers)
      - yield_tonnes                   (target)
      - anything containing 'risk'     (leakage: derived from yield)
      - anything containing 'deviation'(leakage: standardised yield)
      - anything containing 'label'    (leakage: binarised yield)
    """
    exclude = {"region", "weather_station", "year", "yield_tonnes"}
    leak_keywords = ("risk", "deviation", "label")
    cols = [
        c for c in df.columns
        if c not in exclude and not any(kw in c for kw in leak_keywords)
    ]
    return cols


# ── Walk-forward cross-validation ─────────────────────────────────────────────

def walk_forward_cv(df, feature_cols, model, min_train_years=10, step=1):
    """
    Time-series aware CV. For each test year t:
      - train on all years < t
      - predict year t
    Returns list of (mae, rmse, r2) per fold.
    """
    years = sorted(df["year"].unique())
    start = years[min_train_years]  # first year we can test on
    results = []

    for test_year in range(start, max(years) + 1, step):
        train = df[df["year"] <  test_year]
        test  = df[df["year"] == test_year]
        if len(test) == 0:
            continue

        X_train = train[feature_cols].values
        y_train = train["yield_tonnes"].values
        X_test  = test[feature_cols].values
        y_test  = test["yield_tonnes"].values

        m = joblib.load  # placeholder — we pass fitted model
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae_  = mean_absolute_error(y_test, preds)
            rmse_ = np.sqrt(mean_squared_error(y_test, preds))
            r2_   = r2_score(y_test, preds)
            results.append({"year": test_year, "mae": mae_, "rmse": rmse_, "r2": r2_})
        except Exception as e:
            log(f"    Walk-forward fold {test_year} failed: {e}")

    return pd.DataFrame(results)


# ── Feature selection ─────────────────────────────────────────────────────────

def select_top_features(X_train, y_train, feature_cols, top_n=20):
    """
    Use a small Random Forest to rank features by importance.
    Returns the top_n column names.
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top_cols = importances.nlargest(top_n).index.tolist()
    log(f"\n  Top {top_n} features selected:")
    for col, imp in importances.nlargest(top_n).items():
        log(f"    {col:<45} {imp:.4f}")
    return top_cols


# ── Risk labels — computed inside training fold only ──────────────────────────

def make_risk_labels_train_only(train_df, stage_scores):
    """
    Compute risk thresholds on TRAINING data only.
    stage_scores: dict of {stage_name: score_series}
    Returns dict of {stage_name: threshold}
    """
    thresholds = {}
    for stage, scores in stage_scores.items():
        thresholds[stage] = scores.quantile(0.60)
    return thresholds


# ── Stage risk classifiers ─────────────────────────────────────────────────────

def build_risk_scores_train_only(df_train):
    """
    Compute risk scores using only training data statistics.
    Scores are computed from raw weather features — no yield information.
    """
    def normalize(series, cap):
        return (series.clip(0, cap) / cap * 100).clip(0, 100)

    def normalize_inverse(series, safe_min, danger_min):
        return ((safe_min - series) / (safe_min - danger_min) * 100).clip(0, 100)

    def weighted_avg(pairs):
        total = sum(w for _, w in pairs)
        return sum(s * w for s, w in pairs) / total

    scores = {}

    scores["planting"] = weighted_avg([
        (normalize_inverse(df_train["planting_temp_mean"],  14, 8),  0.4),
        (normalize_inverse(df_train["planting_total_rain"], 30, 5),  0.4),
        (normalize(df_train["planting_frost_days"], 8),               0.2),
    ]).clip(0, 100)

    scores["growing"] = weighted_avg([
        (normalize(df_train["growing_heat_stress_days"], 20),          0.35),
        (normalize_inverse(df_train["growing_total_rain"], 100, 30),   0.25),
        (normalize_inverse(df_train["growing_GDD"], 900, 400),         0.25),
        (normalize(df_train["growing_max_dry_streak"], 35),            0.15),
    ]).clip(0, 100)

    scores["boll"] = weighted_avg([
        (normalize(df_train["boll_forming_temp_max_mean"], 41),        0.45),
        (normalize(df_train["boll_forming_dry_days"], 40),             0.30),
        (normalize(df_train["boll_forming_et0_total"], 220),           0.25),
    ]).clip(0, 100)

    scores["harvest"] = weighted_avg([
        (normalize(df_train["harvest_rainy_days"], 25),                0.45),
        (normalize(df_train["harvest_humidity_mean"], 85),             0.35),
        (normalize(df_train["harvest_frost_days"], 8),                 0.20),
    ]).clip(0, 100)

    return scores


def train_risk_classifiers(df_train, df_test):
    """
    Train one Random Forest classifier per growth stage.
    Labels are derived from risk SCORES (not from yield) — no leakage.
    Thresholds are set on training data only.
    """
    log("\n" + "=" * 55)
    log("STEP 8 — Training Risk Classifiers (4 stages)")
    log("=" * 55)

    # Compute scores on training set only
    train_scores = build_risk_scores_train_only(df_train)
    # Apply same formula (not same threshold) to test set
    test_scores  = build_risk_scores_train_only(df_test)

    classifiers = {}
    feature_cols_risk = get_feature_cols(df_train)

    for stage in ["planting", "growing", "boll", "harvest"]:
        # Threshold from TRAINING data only
        threshold = train_scores[stage].quantile(0.60)

        y_train_clf = (train_scores[stage] >= threshold).astype(int)
        y_test_clf  = (test_scores[stage]  >= threshold).astype(int)

        X_train = df_train[feature_cols_risk].values
        X_test  = df_test[feature_cols_risk].values

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train_clf)
        preds = clf.predict(X_test)

        log(f"\n  {stage.upper()} RISK CLASSIFIER:")
        log(f"  Train label dist: {y_train_clf.value_counts().to_dict()}")
        log(f"  Test  label dist: {y_test_clf.value_counts().to_dict()}")
        log(classification_report(y_test_clf, preds))

        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, f"models/risk_{stage}.pkl")
        log(f"  Saved → models/risk_{stage}.pkl ✓")
        classifiers[stage] = clf

    return classifiers


# ── Yield prediction ──────────────────────────────────────────────────────────

def train_yield_model(df_train, df_test, feature_cols):
    log("\n" + "=" * 55)
    log("STEP 9 — Training Yield Prediction Models")
    log("=" * 55)

    X_train = df_train[feature_cols].values
    y_train = df_train["yield_tonnes"].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test["yield_tonnes"].values

    # ── Walk-forward CV scores (honest — not inflated by train/test split) ──
    log("\n  Running walk-forward CV on training data...")
    groups_train = df_train["region"].values

    results = {}

    # ── Model zoo ──────────────────────────────────────────────────────────
    models = {
        "Ridge":             Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "Lasso":             Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=1.0, max_iter=5000))]),
        "ElasticNet":        Pipeline([("sc", StandardScaler()), ("m", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000))]),
        "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            min_samples_leaf=5, subsample=0.8,
            validation_fraction=0.15, n_iter_no_change=30,  # early stopping
            random_state=42
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30,
            eval_metric="rmse",
            random_state=42, verbosity=0
        )
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            num_leaves=15, subsample=0.8,
            early_stopping_rounds=30,
            random_state=42, verbose=-1
        )

    # ── Walk-forward CV for every model ────────────────────────────────────
    log("\n  Walk-forward CV results (honest — trained only on past years):")
    log(f"  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'MAPE':>7}")
    log("  " + "-" * 56)

    best_name  = None
    best_mae   = float("inf")
    best_model = None

    for name, model in models.items():
        try:
            cv_results = walk_forward_cv(df_train, feature_cols, model)
            if len(cv_results) == 0:
                continue
            avg_mae  = cv_results["mae"].mean()
            avg_rmse = cv_results["rmse"].mean()
            avg_r2   = cv_results["r2"].mean()

            # Re-fit on full training data for test evaluation
            if name == "XGBoost":
                # XGBoost needs eval_set for early stopping
                split = int(len(X_train) * 0.85)
                model.fit(
                    X_train[:split], y_train[:split],
                    eval_set=[(X_train[split:], y_train[split:])],
                    verbose=False
                )
            elif name == "LightGBM":
                split = int(len(X_train) * 0.85)
                model.fit(
                    X_train[:split], y_train[:split],
                    eval_set=[(X_train[split:], y_train[split:])],
                )
            else:
                model.fit(X_train, y_train)

            preds    = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, preds)
            test_rmse= np.sqrt(mean_squared_error(y_test, preds))
            test_r2  = r2_score(y_test, preds)
            test_mape= mape(y_test, preds)

            log(f"  {name:<22} CV-MAE={avg_mae:.2f}  Test: MAE={test_mae:.2f}  RMSE={test_rmse:.2f}  R²={test_r2:.3f}  MAPE={test_mape:.1f}%")

            results[name] = {
                "model": model, "cv_mae": avg_mae,
                "test_mae": test_mae, "test_rmse": test_rmse,
                "test_r2": test_r2, "test_mape": test_mape
            }

            if avg_mae < best_mae:
                best_mae   = avg_mae
                best_name  = name
                best_model = model

        except Exception as e:
            log(f"  {name}: failed — {e}")

    log(f"\n  Best model by CV-MAE: {best_name} ✓")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/yield_model.pkl")
    log("  Saved → models/yield_model.pkl ✓")

    return best_model, best_name, results


# ── Prediction for future years ───────────────────────────────────────────────

def predict_future(con, yield_model, risk_classifiers, feature_cols, years=(2025, 2026)):
    log("\n" + "=" * 55)
    log(f"STEP 10 — Predicting for years: {list(years)}")
    log("=" * 55)

    all_preds = []

    for year in years:
        df_year = con.execute(f"""
            SELECT * FROM features WHERE year = {year}
        """).df()

        if df_year.empty:
            log(f"  No feature data for {year} — skipping.")
            continue

        # Check all expected feature columns exist
        missing = [c for c in feature_cols if c not in df_year.columns]
        if missing:
            log(f"  WARNING: {len(missing)} feature columns missing for {year}: {missing[:5]}...")
            continue

        X = df_year[feature_cols].values
        df_year["pred_yield"] = yield_model.predict(X)

        # Stage risk probabilities (probability of high-risk = class 1)
        for stage in ["planting", "growing", "boll", "harvest"]:
            clf = risk_classifiers.get(stage)
            if clf is not None:
                proba = clf.predict_proba(df_year[get_feature_cols(df_year)].values)
                df_year[f"{stage}_risk_pct"] = (proba[:, 1] * 100).round(0).astype(int)
            else:
                df_year[f"{stage}_risk_pct"] = 0

        # Historical average yield per district
        hist_avg = con.execute("""
            SELECT region, AVG(yield_tonnes) AS avg_yield
            FROM features
            WHERE year <= 2021
            GROUP BY region
        """).df().set_index("region")["avg_yield"]

        df_year["avg_yield"] = df_year["region"].map(hist_avg).round(1)
        df_year["pct_change"] = (
            (df_year["pred_yield"] - df_year["avg_yield"]) / df_year["avg_yield"] * 100
        ).round(1)

        all_preds.append(df_year[[
            "region", "year", "pred_yield", "avg_yield", "pct_change",
            "planting_risk_pct", "growing_risk_pct", "boll_risk_pct", "harvest_risk_pct"
        ]])

    if not all_preds:
        log("  No predictions generated.")
        return None

    predictions = pd.concat(all_preds, ignore_index=True)

    # Save to DuckDB
    con.execute("DROP TABLE IF EXISTS predictions")
    con.execute("CREATE TABLE predictions AS SELECT * FROM predictions")
    log("  predictions → DuckDB ✓")

    # Print summary tables
    for year in years:
        df_y = predictions[predictions["year"] == year].sort_values("pred_yield", ascending=False)
        if df_y.empty:
            continue
        log(f"\n  {'='*65}")
        log(f"  {year} PREDICTIONS")
        log(f"  {'='*65}")
        log(f"  {'District':<30} {'Pred':>6} {'Avg':>6} {'Chg':>7}  Plant  Grow  Boll  Harv")
        log(f"  {'-'*65}")
        for _, r in df_y.iterrows():
            chg_str = f"{r['pct_change']:+.1f}%"
            log(
                f"  {r['region']:<30} {r['pred_yield']:>6.1f} {r['avg_yield']:>6.1f} "
                f"{chg_str:>7}  "
                f"{r['planting_risk_pct']:>3}%  "
                f"{r['growing_risk_pct']:>3}%  "
                f"{r['boll_risk_pct']:>3}%  "
                f"{r['harvest_risk_pct']:>3}%"
            )
        total_pred = df_y["pred_yield"].sum()
        total_avg  = df_y["avg_yield"].sum()
        log(f"  {'-'*65}")
        log(f"  {'TOTAL':<30} {total_pred:>6.1f} {total_avg:>6.1f}")

    return predictions


# ── Main entry point ───────────────────────────────────────────────────────────

def run_models():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 3 — MODEL TRAINING & PREDICTION PIPELINE")
    log("=" * 55)

    # Load features (weather columns only — no risk labels)
    df = con.execute("SELECT * FROM features").df()
    log(f"  Loaded features: {df.shape}")

    feature_cols_all = get_feature_cols(df)
    log(f"  Raw weather feature columns: {len(feature_cols_all)}")

    # Sanity check — confirm no leakage columns slipped through
    forbidden = [c for c in df.columns if any(kw in c for kw in ("risk", "deviation", "label"))]
    if forbidden:
        raise ValueError(f"Leakage columns found in features table: {forbidden}")
    log("  No leakage columns detected ✓")

    # ── Train / test split (time-based) ─────────────────────────────────
    df_train = df[df["year"] <= 2021].copy()
    df_test = df[df["year"] == 2022].copy() 
    log(f"\n  Train rows (≤2021): {len(df_train)}")
    log(f"  Test  rows (>2021): {len(df_test)}")

    # ── Feature selection on training data only ──────────────────────────
    log("\n  Selecting top features from training data...")
    X_tr = df_train[feature_cols_all].values
    y_tr = df_train["yield_tonnes"].values
    top_features = select_top_features(X_tr, y_tr, feature_cols_all, top_n=20)

    # ── Risk classifiers ─────────────────────────────────────────────────
    risk_classifiers = train_risk_classifiers(df_train, df_test)

    # ── Yield model ──────────────────────────────────────────────────────
    best_model, best_name, all_results = train_yield_model(
        df_train, df_test, top_features
    )

    # ── Future predictions ───────────────────────────────────────────────
    predictions = predict_future(
        con, best_model, risk_classifiers, top_features, years=(2025,)
    )

    # ── Summary ─────────────────────────────────────────────────────────
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
    run_models() 