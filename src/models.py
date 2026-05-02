# src/models.py
"""
Member 3 — Model Training & Prediction Pipeline
================================================
APPROACH: Region mean + Nested-CV 1-feature Ridge
(Following teacher's recommendation exactly)

FLOW:
  1. Target = yield_anomaly (yield - region mean, training fold only)
  2. Outer loop = Leave-One-Year-Out (LOYO)
  3. Inner loop = pick single best feature per fold (nested CV)
  4. Final model = Ridge trained on ALL data with most-selected feature
  5. Prediction = region_mean + predicted_anomaly
  6. Risk classifiers = 3 Random Forest classifiers (one per DOY stage)

STAGES (DOY-based):
  squaring:  DOY 152–195  (Jun 1  – Jul 14)
  flowering: DOY 196–243  (Jul 15 – Aug 31)  ← critical window
  bolling:   DOY 244–288  (Sep 1  – Oct 15)

EXCLUDED FROM TRAINING (climatically distinct):
  Barda, Bilasuvar, Neftchala, Salyan
  (still predicted for 2025)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import duckdb
from collections import Counter
from datetime import datetime

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report
)
import joblib

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH, LOGS_DIR, STAGES, COTTON_BASE_TEMP, MODELS_DIR


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


# ── Feature column helper ─────────────────────────────────────────────────────

def get_weather_cols(df):
    """All raw weather feature columns — no identifiers, no leakage."""
    exclude  = {"region", "weather_station", "year", "yield_tonnes",
                "yield_anomaly", "district_mean_yield"}
    leak_kw  = ("risk", "deviation", "label")
    return [
        c for c in df.columns
        if c not in exclude and not any(k in c for k in leak_kw)
    ]


# ── Nested-CV 1-feature LOYO ──────────────────────────────────────────────────

def nested_cv_loyo(df):
    """
    Leave-One-Year-Out CV with nested inner loop for feature selection.
    Returns pooled true/pred arrays and list of selected features per fold.
    """
    years     = sorted(df["year"].unique())
    feat_cols = get_weather_cols(df)
    all_true, all_pred, selected = [], [], []

    log(f"\n  Running Nested-CV LOYO ({len(years)} outer folds)...")
    log(f"  Available features: {len(feat_cols)}")
    log(f"\n  {'Year':>6}  {'Best feature':<50}  {'Fold MAE':>9}")
    log("  " + "-" * 72)

    for test_year in years:
        outer_train = df[df["year"] != test_year].copy()
        outer_test  = df[df["year"] == test_year].copy()

        region_mean = outer_train.groupby("region")["yield_tonnes"].mean()
        outer_train["anomaly"] = outer_train["yield_tonnes"] - outer_train["region"].map(region_mean)
        outer_test["anomaly"]  = outer_test["yield_tonnes"]  - outer_test["region"].map(region_mean)

        inner_years = sorted(outer_train["year"].unique())
        best_feat, best_mae = None, float("inf")

        for feat in feat_cols:
            inner_true, inner_pred = [], []
            for inner_year in inner_years:
                itr = outer_train[outer_train["year"] != inner_year]
                ite = outer_train[outer_train["year"] == inner_year]
                if len(itr) < 5 or ite.empty:
                    continue
                m = RidgeCV(alphas=[0.1, 1, 10, 100])
                m.fit(itr[[feat]].values, itr["anomaly"].values)
                inner_true.extend(ite["anomaly"].values)
                inner_pred.extend(m.predict(ite[[feat]].values))
            if len(inner_true) < 3:
                continue
            mae = mean_absolute_error(inner_true, inner_pred)
            if mae < best_mae:
                best_mae, best_feat = mae, feat

        if best_feat is None:
            continue

        selected.append(best_feat)

        m = RidgeCV(alphas=[0.1, 1, 10, 100])
        m.fit(outer_train[[best_feat]].values, outer_train["anomaly"].values)
        pred_anomaly = m.predict(outer_test[[best_feat]].values)
        pred_yield   = pred_anomaly + outer_test["region"].map(region_mean).values
        true_yield   = outer_test["yield_tonnes"].values

        all_true.extend(true_yield)
        all_pred.extend(pred_yield)

        fold_mae = mean_absolute_error(true_yield, pred_yield)
        log(f"  {test_year:>6}  {best_feat:<50}  {fold_mae:>9.2f}")

    return np.array(all_true), np.array(all_pred), selected


# ── Risk scores ───────────────────────────────────────────────────────────────

def build_risk_scores(df):
    """Weather-only risk scores using DOY stage names — no yield, no leakage."""
    def norm(s, cap):
        return (s.clip(0, cap) / cap * 100).clip(0, 100)

    def norm_inv(s, safe, danger):
        return ((safe - s) / (safe - danger) * 100).clip(0, 100)

    def wavg(pairs):
        t = sum(w for _, w in pairs)
        return sum(s * w for s, w in pairs) / t

    return {
        "squaring": wavg([
            (norm(df["squaring_heat_stress_days"],        10), 0.30),
            (norm_inv(df["squaring_total_rain"],      80, 20), 0.25),
            (norm_inv(df["squaring_GDD"],            600, 200), 0.25),
            (norm(df["squaring_max_dry_streak"],          20), 0.20),
        ]).clip(0, 100),
        "flowering": wavg([
            (norm(df["flowering_temp_min_mean"],          30), 0.45),
            (norm(df["flowering_heat_stress_days"],        20), 0.25),
            (norm_inv(df["flowering_total_rain"],     60, 15), 0.15),
            (norm(df["flowering_et0_total"],             200), 0.15),
        ]).clip(0, 100),
        "bolling": wavg([
            (norm(df["bolling_rainy_days"],               25), 0.45),
            (norm(df["bolling_humidity_mean"],             85), 0.35),
            (norm(df["bolling_frost_days"],                 5), 0.20),
        ]).clip(0, 100),
    }


def train_risk_classifiers(df_train, df_test, feat_cols):
    log("\n" + "=" * 55)
    log("STEP 8 — Risk Classifiers (3 DOY stages)")
    log("=" * 55)

    train_scores = build_risk_scores(df_train)
    test_scores  = build_risk_scores(df_test)
    classifiers  = {}

    for stage in ["squaring", "flowering", "bolling"]:
        thr   = train_scores[stage].quantile(0.60)
        y_tr  = (train_scores[stage] >= thr).astype(int)
        y_te  = (test_scores[stage]  >= thr).astype(int)

        df_tr = df_train.dropna(subset=feat_cols)
        df_te = df_test.dropna(subset=feat_cols)

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        clf.fit(df_tr[feat_cols].values, y_tr[df_tr.index])
        preds = clf.predict(df_te[feat_cols].values)

        log(f"\n  {stage.upper()}:")
        log(classification_report(y_te[df_te.index], preds, zero_division=0))

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(clf, os.path.join(MODELS_DIR, f"risk_{stage}.pkl"))
        log(f"  Saved → risk_{stage}.pkl ✓")
        classifiers[stage] = clf

    return classifiers


# ── Feature helpers for 2025 prediction ──────────────────────────────────────

def _gdd(temp_max, temp_min, base=COTTON_BASE_TEMP):
    return max(((temp_max + temp_min) / 2) - base, 0)


def _dry_streak(precip):
    best = cur = 0
    for v in precip:
        cur = cur + 1 if v < 1 else 0
        best = max(best, cur)
    return best


def _stage_features(stage_df, name):
    """Mirrors features.py — 14 features per stage including sunshine."""
    if stage_df.empty:
        return {}
    gdd = stage_df.apply(lambda r: _gdd(r["temp_max"], r["temp_min"]), axis=1)
    return {
        f"{name}_temp_mean":        round(stage_df["temp_mean"].mean(), 4),
        f"{name}_temp_min_mean":    round(stage_df["temp_min"].mean(), 4),
        f"{name}_temp_max_mean":    round(stage_df["temp_max"].mean(), 4),
        f"{name}_heat_stress_days": int((stage_df["temp_max"] > 35).sum()),
        f"{name}_frost_days":       int((stage_df["temp_min"] < 0).sum()),
        f"{name}_GDD":              round(gdd.sum(), 4),
        f"{name}_total_rain":       round(stage_df["precipitation"].sum(), 4),
        f"{name}_rainy_days":       int((stage_df["precipitation"] > 1).sum()),
        f"{name}_dry_days":         int((stage_df["precipitation"] < 1).sum()),
        f"{name}_max_dry_streak":   _dry_streak(stage_df["precipitation"]),
        f"{name}_humidity_mean":    round(stage_df["humidity_mean"].mean(), 4),
        f"{name}_wind_mean":        round(stage_df["wind_speed"].mean(), 4),
        f"{name}_et0_total":        round(stage_df["et0"].sum(), 4),
        f"{name}_sunshine_total":   round(stage_df["sunshine"].sum(), 4),
    }


def build_prediction_features(con, year):
    """Build feature rows for 2025 directly from clean_weather using DOY windows."""
    districts = con.execute(
        "SELECT DISTINCT region, weather_station FROM clean_cotton"
    ).df()

    weather = con.execute(f"""
        SELECT region, year, doy,
               temp_mean, temp_min, temp_max,
               precipitation, humidity_mean, wind_speed, et0, sunshine
        FROM clean_weather WHERE year = {year}
        ORDER BY region, doy
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
        for sname, (start_doy, end_doy) in STAGES.items():
            stage_df = daily[(daily["doy"] >= start_doy) & (daily["doy"] <= end_doy)]
            feat.update(_stage_features(stage_df, sname))
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

def predict_future(con, best_feat, risk_classifiers, all_feat_cols, df_all):
    log("\n" + "=" * 55)
    log("STEP 10 — Predicting 2025")
    log("=" * 55)

    df_year = build_prediction_features(con, 2025)
    if df_year.empty:
        log("  No 2025 weather data — cannot predict.")
        return None

    # Region mean from ALL training data (2000–2024)
    region_mean = df_all.groupby("region")["yield_tonnes"].mean()
    df_year["district_mean_yield"] = df_year["region"].map(region_mean)

    # Final model: Ridge on all data with best feature
    df_all = df_all.copy()
    df_all["anomaly"] = df_all["yield_tonnes"] - df_all["region"].map(region_mean)
    df_clean = df_all.dropna(subset=[best_feat, "anomaly"])

    final_model = RidgeCV(alphas=[0.1, 1, 10, 100])
    final_model.fit(df_clean[[best_feat]].values, df_clean["anomaly"].values)

    pred_anomaly = final_model.predict(df_year[[best_feat]].values)
    df_year["pred_yield"] = (
        df_year["district_mean_yield"] + pred_anomaly
    ).clip(0)

    # Stage risk probabilities
    for stage in ["squaring", "flowering", "bolling"]:
        clf = risk_classifiers.get(stage)
        if clf is not None:
            proba = clf.predict_proba(df_year[all_feat_cols].values)
            df_year[f"{stage}_risk_pct"] = (proba[:, 1] * 100).round(0).astype(int)
        else:
            df_year[f"{stage}_risk_pct"] = 0

    # Historical average from features table
    hist_avg = con.execute("""
        SELECT region, AVG(yield_tonnes) AS avg_yield
        FROM features WHERE year <= 2021
        GROUP BY region
    """).df().set_index("region")["avg_yield"]

    df_year["avg_yield"]  = df_year["region"].map(hist_avg).round(1)
    df_year["pct_change"] = (
        (df_year["pred_yield"] - df_year["avg_yield"])
        / df_year["avg_yield"] * 100
    ).round(1)

    pred_df = df_year[[
        "region", "year", "pred_yield", "avg_yield", "pct_change",
        "squaring_risk_pct", "flowering_risk_pct", "bolling_risk_pct"
    ]].copy()

    con.execute("DROP TABLE IF EXISTS predictions")
    con.execute("CREATE TABLE predictions AS SELECT * FROM pred_df")
    log("  predictions → DuckDB ✓")

    df_s = pred_df.sort_values("pred_yield", ascending=False)
    log(f"\n  {'='*65}")
    log(f"  2025 PREDICTIONS  (best feature: {best_feat})")
    log(f"  {'='*65}")
    log(f"  {'District':<30} {'Pred':>6} {'Avg':>6} {'Chg':>7}  Squar  Flow  Boll")
    log(f"  {'-'*65}")
    for _, r in df_s.iterrows():
        log(
            f"  {r['region']:<30} {r['pred_yield']:>6.1f} "
            f"{r['avg_yield']:>6.1f} {r['pct_change']:>+7.1f}%  "
            f"{r['squaring_risk_pct']:>3}%  "
            f"{r['flowering_risk_pct']:>3}%  "
            f"{r['bolling_risk_pct']:>3}%"
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

    # ── Exclude climatically distinct districts from training ─────────────
    EXCLUDE = {"Barda district", "Bilasuvar district",
               "Neftchala district", "Salyan district"}

    df_model = df[~df["region"].isin(EXCLUDE)].copy()
    log(f"\n  Excluded from training ({len(EXCLUDE)} districts):")
    for d in sorted(EXCLUDE):
        log(f"    ✗ {d}")
    log(f"  Training districts: {df_model['region'].nunique()} ({len(df_model)} rows)")

    df_train = df_model[df_model["year"] <= 2021].copy()
    df_test  = df_model[df_model["year"] >  2021].copy()
    df_all   = df_model.copy()

    log(f"\n  Train (≤2021): {len(df_train)} rows")
    log(f"  Test  (>2021): {len(df_test)} rows  "
        f"(years {df_test['year'].min()}–{df_test['year'].max()})")

    # ── Nested-CV LOYO ────────────────────────────────────────────────────
    log("\n" + "=" * 55)
    log("STEP 7 — Nested-CV 1-Feature LOYO (teacher's approach)")
    log("=" * 55)

    all_true, all_pred, selected = nested_cv_loyo(df_train)

    cv_mae = mean_absolute_error(all_true, all_pred)
    cv_r2  = r2_score(all_true, all_pred)
    log(f"\n  {'='*50}")
    log(f"  NESTED-CV RESULTS (2000–2021, 11 districts)")
    log(f"  {'='*50}")
    log(f"  MAE : {cv_mae:.3f}")
    log(f"  R²  : {cv_r2:.3f}")

    feat_counts = Counter(selected)
    log(f"\n  Most selected features:")
    for feat, count in feat_counts.most_common(5):
        log(f"    {feat:<50} selected {count} times")

    best_feat = feat_counts.most_common(1)[0][0]
    log(f"\n  Best feature overall: {best_feat} ✓")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "best_feature.txt"), "w") as f:
        f.write(best_feat)

    # ── Risk classifiers ──────────────────────────────────────────────────
    all_feat_cols = get_weather_cols(df_model)
    risk_classifiers = train_risk_classifiers(df_train, df_test, all_feat_cols)

    # ── Final model + 2025 prediction ─────────────────────────────────────
    log("\n" + "=" * 55)
    log(f"STEP 9 — Final Ridge Model (feature: {best_feat})")
    log("=" * 55)

    predict_future(con, best_feat, risk_classifiers, all_feat_cols, df_all)

    log("\n" + "=" * 55)
    log("MODEL PIPELINE COMPLETE — DuckDB Tables:")
    log("=" * 55)
    for t in con.execute("SHOW TABLES").fetchall():
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        cols  = len(con.execute(f"SELECT * FROM {t[0]} LIMIT 1").description)
        log(f"  {t[0]:<25} {count:>6} rows  {cols:>3} cols")

    con.close()
    log("\n  Next: run src/reports.py")


if __name__ == "__main__":
    run_models() 