# src/features.py
# Member 2 — Feature Engineering
# Creates two tables:
#   features            → weather features + yield (used for training)
#   features_with_risk  → above + risk scores/labels (used for website)

import pandas as pd
import numpy as np
import duckdb
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DB_PATH, STAGES, COTTON_BASE_TEMP, LOGS_DIR
)


def log(msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_gdd(temp_mean, base=COTTON_BASE_TEMP):
    """
    Growing Degree Days from daily mean temperature.
    (temp_max/temp_min no longer available — using temp_mean directly)
    """
    return max(temp_mean - base, 0)


def max_dry_streak(precip_series):
    """Longest consecutive streak of dry days (precip < 1mm)."""
    max_streak = current = 0
    for val in precip_series:
        if val < 1:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def compute_stage_features(stage_df, stage_name):
    """
    Aggregates daily weather into features for one growth stage.
    Uses only the 4 columns available: temp_mean, precipitation,
    humidity_mean, wind_speed.
    Returns a dict of feature_name → value.
    """
    if stage_df.empty:
        return {}

    gdd_values = stage_df["temp_mean"].apply(
        lambda t: calculate_gdd(t)
    )

    return {
        # Temperature
        f"{stage_name}_temp_mean":        round(stage_df["temp_mean"].mean(), 4),
        f"{stage_name}_heat_stress_days": int((stage_df["temp_mean"] > 32).sum()),  # proxy via mean
        f"{stage_name}_frost_days":       int((stage_df["temp_mean"] < 2).sum()),   # proxy via mean
        f"{stage_name}_GDD":              round(gdd_values.sum(), 4),
        # Precipitation
        f"{stage_name}_total_rain":       round(stage_df["precipitation"].sum(), 4),
        f"{stage_name}_rainy_days":       int((stage_df["precipitation"] > 1).sum()),
        f"{stage_name}_dry_days":         int((stage_df["precipitation"] < 1).sum()),
        f"{stage_name}_max_dry_streak":   max_dry_streak(stage_df["precipitation"]),
        # Humidity
        f"{stage_name}_humidity_mean":    round(stage_df["humidity_mean"].mean(), 4),
        # Wind
        f"{stage_name}_wind_mean":        round(stage_df["wind_speed"].mean(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD FEATURES TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_features(con):
    log("=" * 55)
    log("STEP 6 — Building Growth Stage Features")
    log("=" * 55)

    cotton = con.execute("""
        SELECT region, year, yield_tonnes, weather_station
        FROM clean_cotton
        WHERE yield_tonnes IS NOT NULL
          AND yield_tonnes > 0
        ORDER BY region, year
    """).df()

    # Only the 4 columns that exist in clean_weather
    weather = con.execute("""
        SELECT region, year, month, day,
               temp_mean, precipitation, humidity_mean, wind_speed
        FROM clean_weather
        ORDER BY region, year, month, day
    """).df()

    log(f"  Cotton rows:  {len(cotton)}")
    log(f"  Weather rows: {len(weather)}")
    log(f"  Districts:    {cotton['region'].nunique()}")
    log(f"  Processing {len(cotton)} district-year combinations...")

    all_rows = []
    for i, row in cotton.iterrows():
        region  = row["region"]
        year    = int(row["year"])
        yield_t = row["yield_tonnes"]
        station = row["weather_station"]

        daily = weather[
            (weather["region"] == station) &
            (weather["year"]   == year)
        ]

        if daily.empty:
            log(f"  WARNING: No weather data for {station} in {year} — skipping")
            continue

        feat = {
            "region":          region,
            "weather_station": station,
            "year":            year,
            "yield_tonnes":    yield_t
        }

        for stage_name, (start_m, end_m) in STAGES.items():
            stage_df = daily[
                (daily["month"] >= start_m) &
                (daily["month"] <= end_m)
            ]
            feat.update(compute_stage_features(stage_df, stage_name))

        all_rows.append(feat)

        if (i + 1) % 100 == 0:
            log(f"  Processed {i + 1}/{len(cotton)} rows...")

    df_feat = pd.DataFrame(all_rows).sort_values(
        ["region", "year"]
    ).reset_index(drop=True)

    rows_before  = len(df_feat)
    df_feat      = df_feat.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df_feat)

    if rows_dropped > 0:
        log(f"\n  Dropped {rows_dropped} rows with null features")
        log(f"  Reason: missing weather coverage for some stage windows")

    log(f"\n  Final shape:  {df_feat.shape}")
    log(f"  Districts:    {df_feat['region'].nunique()}")
    log(f"  Years:        {df_feat['year'].min()} – {df_feat['year'].max()}")
    log(f"  Null check:   {df_feat.isnull().sum().sum()} nulls")

    con.execute("DROP TABLE IF EXISTS features")
    con.execute("CREATE TABLE features AS SELECT * FROM df_feat")
    log("  features → DuckDB ✓")

    return df_feat


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD features_with_risk TABLE (for website only)
# Risk scores and labels are computed here but NEVER used for training
# ─────────────────────────────────────────────────────────────────────────────

def build_features_with_risk(con):
    log("=" * 55)
    log("STEP 7 — Building Risk Scores (for website only)")
    log("         ⚠ These will NOT be used in model training")
    log("=" * 55)

    df = con.execute("SELECT * FROM features").df()

    def normalize(series, danger_max, cap):
        """Higher value = more risk."""
        return (series.clip(0, cap) / cap * 100).clip(0, 100)

    def normalize_inverse(series, safe_min, danger_min):
        """Lower value = more risk."""
        return ((safe_min - series) / (safe_min - danger_min) * 100).clip(0, 100)

    def weighted_avg(pairs):
        total = sum(w for _, w in pairs)
        return sum(s * w for s, w in pairs) / total

    # ── Planting Risk (Mar–Apr) ───────────────────────────────────────────────
    # Risk factors: too cold, too little rain, frost days
    df["planting_risk_score"] = weighted_avg([
        (normalize_inverse(df["planting_temp_mean"],  14, 8),  0.4),
        (normalize_inverse(df["planting_total_rain"], 30, 5),  0.4),
        (normalize(df["planting_frost_days"],          3, 8),  0.2),
    ]).clip(0, 100)

    # ── Growing Risk (May–Aug) ────────────────────────────────────────────────
    # Risk factors: heat stress days, drought, low GDD, long dry streaks
    df["growing_risk_score"] = weighted_avg([
        (normalize(df["growing_heat_stress_days"],      8,  20), 0.35),
        (normalize_inverse(df["growing_total_rain"],  100,  30), 0.25),
        (normalize_inverse(df["growing_GDD"],         900, 400), 0.25),
        (normalize(df["growing_max_dry_streak"],       15,  35), 0.15),
    ]).clip(0, 100)

    # ── Boll Forming Risk (Aug–Sep) ───────────────────────────────────────────
    # Risk factors: high temp mean (proxy for heat), dry days, high humidity
    # (et0 removed — not available; replaced with humidity as drying stress proxy)
    df["boll_risk_score"] = weighted_avg([
        (normalize(df["boll_forming_temp_mean"],     30, 38), 0.45),
        (normalize(df["boll_forming_dry_days"],      20, 40), 0.30),
        (normalize(df["boll_forming_humidity_mean"], 70, 90), 0.25),
    ]).clip(0, 100)

    # ── Harvest Risk (Sep–Nov) ────────────────────────────────────────────────
    # Risk factors: too many rainy days, high humidity, frost days
    df["harvest_risk_score"] = weighted_avg([
        (normalize(df["harvest_rainy_days"],     12, 25), 0.45),
        (normalize(df["harvest_humidity_mean"],  65, 85), 0.35),
        (normalize(df["harvest_frost_days"],      2,  8), 0.20),
    ]).clip(0, 100)

    # ── Overall Risk ──────────────────────────────────────────────────────────
    df["overall_risk_score"] = weighted_avg([
        (df["planting_risk_score"], 0.15),
        (df["growing_risk_score"],  0.40),
        (df["boll_risk_score"],     0.30),
        (df["harvest_risk_score"],  0.15),
    ]).clip(0, 100)

    # ── Risk Labels from yield deviation ─────────────────────────────────────
    region_avg = df.groupby("region")["yield_tonnes"].transform("mean")
    region_std = df.groupby("region")["yield_tonnes"].transform("std")
    df["yield_deviation"]    = (df["yield_tonnes"] - region_avg) / region_std
    df["overall_risk_label"] = (df["yield_deviation"] < -0.5).astype(int)

    for stage in ["planting", "growing", "boll", "harvest"]:
        col       = f"{stage}_risk_score"
        threshold = df[col].quantile(0.60)
        df[f"{stage}_risk_label"] = (df[col] >= threshold).astype(int)

    # ── Validate: risky years should have lower avg yield ────────────────────
    avg_by_label = df.groupby("overall_risk_label")["yield_tonnes"].mean()
    log(f"\n  Risk label validation:")
    log(f"    Safe  (label=0) avg yield: {avg_by_label.get(0, 0):.1f}t")
    log(f"    Risky (label=1) avg yield: {avg_by_label.get(1, 0):.1f}t")
    status = "PASSED ✓" if avg_by_label.get(0, 0) > avg_by_label.get(1, 0) \
             else "WARNING ✗"
    log(f"    Validation: {status}")

    con.execute("DROP TABLE IF EXISTS features_with_risk")
    con.execute("CREATE TABLE features_with_risk AS SELECT * FROM df")
    log("\n  features_with_risk → DuckDB ✓")
    log("  (used for website visualization only — NOT for model training)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_features():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 2 — FEATURE ENGINEERING PIPELINE")
    log("=" * 55)

    build_features(con)
    build_features_with_risk(con)

    log("\n" + "=" * 55)
    log("FEATURE ENGINEERING COMPLETE — DuckDB Tables:")
    log("=" * 55)
    for t in con.execute("SHOW TABLES").fetchall():
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        cols  = len(con.execute(f"SELECT * FROM {t[0]} LIMIT 1").description)
        log(f"  {t[0]:<25} {count:>6} rows  {cols:>3} cols")

    con.close()
    log("\n  Next: run src/models.py")


if __name__ == "__main__":
    run_features() 