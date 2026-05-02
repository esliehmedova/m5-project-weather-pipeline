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
from config import DB_PATH, STAGES, COTTON_BASE_TEMP, LOGS_DIR


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

def calculate_gdd(temp_max, temp_min, base=COTTON_BASE_TEMP):
    """GDD = max(((temp_max + temp_min) / 2) - base, 0)"""
    return max(((temp_max + temp_min) / 2) - base, 0)


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
    Uses DOY-filtered stage_df (not month-filtered).

    Uses all 8 columns: temp_mean, temp_min, temp_max,
    precipitation, humidity_mean, wind_speed, et0, sunshine.

    Key feature: {stage}_temp_min_mean — mean night temperature.
    For the flowering stage this is THE dominant predictor:
    high nights (>22°C) during Jul 15–Aug 31 cause boll shedding.
    """
    if stage_df.empty:
        return {}

    gdd_values = stage_df.apply(
        lambda r: calculate_gdd(r["temp_max"], r["temp_min"]), axis=1
    )

    return {
        # Temperature
        f"{stage_name}_temp_mean": round(stage_df["temp_mean"].mean(), 4),
        f"{stage_name}_temp_min_mean": round(stage_df["temp_min"].mean(), 4),   # KEY night temp
        f"{stage_name}_temp_max_mean": round(stage_df["temp_max"].mean(), 4),
        f"{stage_name}_heat_stress_days": int((stage_df["temp_max"] > 35).sum()),  # proper threshold
        f"{stage_name}_frost_days": int((stage_df["temp_min"] < 0).sum()),   # proper threshold
        f"{stage_name}_GDD": round(gdd_values.sum(), 4),
        # Precipitation
        f"{stage_name}_total_rain": round(stage_df["precipitation"].sum(), 4),
        f"{stage_name}_rainy_days": int((stage_df["precipitation"] > 1).sum()),
        f"{stage_name}_dry_days": int((stage_df["precipitation"] < 1).sum()),
        f"{stage_name}_max_dry_streak": max_dry_streak(stage_df["precipitation"]),
        # Humidity
        f"{stage_name}_humidity_mean": round(stage_df["humidity_mean"].mean(), 4),
        # Wind
        f"{stage_name}_wind_mean": round(stage_df["wind_speed"].mean(), 4),
        # Evapotranspiration
        f"{stage_name}_et0_total": round(stage_df["et0"].sum(), 4),
        # Sunshine duration — solar energy available for photosynthesis
        f"{stage_name}_sunshine_total": round(stage_df["sunshine"].sum(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD FEATURES TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_features(con):
    log("=" * 55)
    log("STEP 6 — Building Growth Stage Features (DOY-based)")
    log("=" * 55)

    cotton = con.execute("""
        SELECT region, year, yield_tonnes, weather_station
        FROM clean_cotton
        WHERE yield_tonnes IS NOT NULL
          AND yield_tonnes > 0
        ORDER BY region, year
    """).df()

    # All 8 columns + doy for DOY-based stage filtering
    weather = con.execute("""
        SELECT region, year, month, day, doy,
               temp_mean, temp_min, temp_max,
               precipitation, humidity_mean, wind_speed, et0, sunshine
        FROM clean_weather
        ORDER BY region, year, doy
    """).df()

    log(f"  Cotton rows:  {len(cotton)}")
    log(f"  Weather rows: {len(weather)}")
    log(f"  Districts: {cotton['region'].nunique()}")
    log(f"  Stage windows (DOY):")
    for stage, (start_doy, end_doy) in STAGES.items():
        log(f"{stage:<12} DOY {start_doy}–{end_doy}")
    log(f"Processing {len(cotton)} district-year combinations...")

    all_rows = []
    for i, row in cotton.iterrows():
        region  = row["region"]
        year = int(row["year"])
        yield_t = row["yield_tonnes"]
        station = row["weather_station"]

        daily = weather[
            (weather["region"] == station) &
            (weather["year"]   == year)
        ]

        if daily.empty:
            log(f" WARNING: No weather data for {station} in {year} — skipping")
            continue

        feat = {
            "region":          region,
            "weather_station": station,
            "year":            year,
            "yield_tonnes":    yield_t
        }

        # DOY-based stage filtering — replaces old month-based filtering
        for stage_name, (start_doy, end_doy) in STAGES.items():
            stage_df = daily[
                (daily["doy"] >= start_doy) &
                (daily["doy"] <= end_doy)
            ]
            feat.update(compute_stage_features(stage_df, stage_name))

        all_rows.append(feat)

        if (i + 1) % 100 == 0:
            log(f"Processed {i + 1}/{len(cotton)} rows...")

    df_feat = pd.DataFrame(all_rows).sort_values(
        ["region", "year"]
    ).reset_index(drop=True)

    rows_before  = len(df_feat)
    df_feat = df_feat.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df_feat)

    if rows_dropped > 0:
        log(f"\n  Dropped {rows_dropped} rows with null features")
        log(f"  Reason: missing weather coverage for some DOY windows")

    log(f"\n  Final shape:  {df_feat.shape}")
    log(f"Districts: {df_feat['region'].nunique()}")
    log(f"Years: {df_feat['year'].min()} – {df_feat['year'].max()}")
    log(f" Null check: {df_feat.isnull().sum().sum()} nulls")
    log(f"Features per stage: 14 × 3 stages = 42 weather features")

    con.execute("DROP TABLE IF EXISTS features")
    con.execute("CREATE TABLE features AS SELECT * FROM df_feat")
    log("  features → DuckDB ✓")

    return df_feat

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD features_with_risk TABLE (for website only)
# ─────────────────────────────────────────────────────────────────────────────

def build_features_with_risk(con):
    log("=" * 55)
    log("STEP 7 — Building Risk Scores (for website only)")
    log("These will NOT be used in model training")
    log("=" * 55)

    df = con.execute("SELECT * FROM features").df()

    def normalize(series, cap):
        """Higher value = more risk."""
        return (series.clip(0, cap) / cap * 100).clip(0, 100)

    def normalize_inverse(series, safe_min, danger_min):
        """Lower value = more risk."""
        return ((safe_min - series) / (safe_min - danger_min) * 100).clip(0, 100)

    def weighted_avg(pairs):
        total = sum(w for _, w in pairs)
        return sum(s * w for s, w in pairs) / total

    # ── Squaring Risk (DOY 152–195) ───────────────────────────────────────
    df["squaring_risk_score"] = weighted_avg([
        (normalize(df["squaring_heat_stress_days"],10), 0.30),
        (normalize_inverse(df["squaring_total_rain"], 80, 20), 0.25),
        (normalize_inverse(df["squaring_GDD"], 600, 200), 0.25),
        (normalize(df["squaring_max_dry_streak"], 20), 0.20),
    ]).clip(0, 100)

    # ── Flowering Risk (DOY 196–243) — critical stage ─────────────────────
    df["flowering_risk_score"] = weighted_avg([
        (normalize(df["flowering_temp_min_mean"], 30), 0.45),
        (normalize(df["flowering_heat_stress_days"],  20), 0.25),
        (normalize_inverse(df["flowering_total_rain"], 60, 15), 0.15),
        (normalize(df["flowering_et0_total"], 200), 0.15),
    ]).clip(0, 100)

    # ── Bolling Risk (DOY 244–288) ────────────────────────────────────────
    df["bolling_risk_score"] = weighted_avg([
        (normalize(df["bolling_rainy_days"], 25), 0.45),
        (normalize(df["bolling_humidity_mean"], 85), 0.35),
        (normalize(df["bolling_frost_days"], 5), 0.20),
    ]).clip(0, 100)

    # ── Overall Risk ──────────────────────────────────────────────────────
    df["overall_risk_score"] = weighted_avg([
        (df["squaring_risk_score"],  0.25),
        (df["flowering_risk_score"], 0.50),
        (df["bolling_risk_score"],   0.25),
    ]).clip(0, 100)

    # ── Risk Labels from yield deviation ─────────────────────────────────
    region_avg = df.groupby("region")["yield_tonnes"].transform("mean")
    region_std = df.groupby("region")["yield_tonnes"].transform("std")
    df["yield_deviation"]  = (df["yield_tonnes"] - region_avg) / region_std
    df["overall_risk_label"] = (df["yield_deviation"] < -0.5).astype(int)

    for stage in ["squaring", "flowering", "bolling"]:
        col = f"{stage}_risk_score"
        threshold = df[col].quantile(0.60)
        df[f"{stage}_risk_label"] = (df[col] >= threshold).astype(int)

    # ── Validate ──────────────────────────────────────────────────────────
    avg_by_label = df.groupby("overall_risk_label")["yield_tonnes"].mean()
    log(f"\n  Risk label validation:")
    log(f"Safe  (label=0) avg yield: {avg_by_label.get(0, 0):.1f}t")
    log(f" Risky (label=1) avg yield: {avg_by_label.get(1, 0):.1f}t")
    status = "PASSED ✓" if avg_by_label.get(0, 0) > avg_by_label.get(1, 0) \
             else "WARNING ✗"
    log(f" Validation: {status}")

    con.execute("DROP TABLE IF EXISTS features_with_risk")
    con.execute("CREATE TABLE features_with_risk AS SELECT * FROM df")
    log("\n features_with_risk → DuckDB ✓")
    log("(used for website visualization only — NOT for model training)")

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
