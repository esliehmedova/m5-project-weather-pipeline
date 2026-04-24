# src/features.py
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
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a",encoding="utf-8") as f:
        f.write(line + "\n")


def calculate_gdd(temp_max, temp_min, base=COTTON_BASE_TEMP):
    """Growing Degree Days for a single day."""
    mean = (temp_max + temp_min) / 2
    return max(mean - base, 0)


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
    """Aggregates daily weather into features for one growth stage."""
    if stage_df.empty:
        return {}

    gdd_values = stage_df.apply(
        lambda r: calculate_gdd(r["temp_max"], r["temp_min"]), axis=1
    )

    return {
        f"{stage_name}_temp_mean":        stage_df["temp_mean"].mean(),
        f"{stage_name}_temp_max_mean":    stage_df["temp_max"].mean(),
        f"{stage_name}_temp_min_mean":    stage_df["temp_min"].mean(),
        f"{stage_name}_heat_stress_days": int((stage_df["temp_max"] > 35).sum()),
        f"{stage_name}_frost_days":       int((stage_df["temp_min"] < 0).sum()),
        f"{stage_name}_GDD":              gdd_values.sum(),
        f"{stage_name}_total_rain":       stage_df["precipitation"].sum(),
        f"{stage_name}_rainy_days":       int((stage_df["precipitation"] > 1).sum()),
        f"{stage_name}_dry_days":         int((stage_df["precipitation"] < 1).sum()),
        f"{stage_name}_max_dry_streak":   max_dry_streak(stage_df["precipitation"]),
        f"{stage_name}_humidity_mean":    ((stage_df["humidity_max"] + stage_df["humidity_min"]) / 2).mean(),
        f"{stage_name}_wind_mean":        stage_df["wind_speed"].mean(),
        f"{stage_name}_et0_total":        stage_df["et0"].sum(),
    }


def build_features(con):
    log("=" * 55)
    log("STEP 6 — Building Growth Stage Features")
    log("=" * 55)

    # Read clean data using SQL
    cotton = con.execute("""
        SELECT region, year, yield_tonnes, weather_station
        FROM clean_cotton
        ORDER BY region, year
    """).df()

    weather = con.execute("""
        SELECT region, year, month, day,
               temp_max, temp_min, temp_mean,
               precipitation, humidity_max, humidity_min,
               wind_speed, et0
        FROM clean_weather
        ORDER BY region, year, month, day
    """).df()

    log(f"  Cotton rows loaded:  {len(cotton)}")
    log(f"  Weather rows loaded: {len(weather)}")
    log(f"  Processing {len(cotton)} district-year combinations...")

    all_rows = []
    for i, row in cotton.iterrows():
        region  = row["region"]
        year    = row["year"]
        yield_t = row["yield_tonnes"]
        station = row["weather_station"]

        # Get daily weather for this station and year using pandas filter
        daily = weather[
            (weather["region"] == station) &
            (weather["year"]   == year)
        ]

        if daily.empty:
            log(f"  WARNING: No weather for {station} in {year}")
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
            log(f"    Processed {i+1}/{len(cotton)} rows...")

    df_feat = pd.DataFrame(all_rows).sort_values(
        ["region", "year"]
    ).reset_index(drop=True)

    log(f"\n  Feature dataset shape: {df_feat.shape}")
    log(f"  Features per row:      {len(df_feat.columns) - 4}")

    con.execute("DROP TABLE IF EXISTS features")
    con.execute("CREATE TABLE features AS SELECT * FROM df_feat")
    log("  features → DuckDB ✓")

    return df_feat


def build_risk_labels(con):
    log("=" * 55)
    log("STEP 7 — Building Risk Labels and Scores")
    log("=" * 55)

    # Read features using SQL
    df = con.execute("SELECT * FROM features").df()
    
    ###
    df["yield_tonnes"] = pd.to_numeric(df["yield_tonnes"], errors="coerce")
    df = df.dropna(subset=["yield_tonnes"])

    def normalize(series, danger_max, cap):
        return (series.clip(0, cap) / cap * 100).clip(0, 100)

    def normalize_inverse(series, safe_min, danger_min):
        return ((safe_min - series) / (safe_min - danger_min) * 100).clip(0, 100)

    def weighted_avg(pairs):
        total = sum(w for _, w in pairs)
        return sum(s * w for s, w in pairs) / total

    # ── Planting Risk ─────────────────────────────────────────────────────
    df["planting_risk_score"] = weighted_avg([
        (normalize_inverse(df["planting_temp_mean"],  14, 8),  0.4),
        (normalize_inverse(df["planting_total_rain"], 30, 5),  0.4),
        (normalize(df["planting_frost_days"], 3, 8),           0.2),
    ]).clip(0, 100)

    # ── Growing Risk ──────────────────────────────────────────────────────
    df["growing_risk_score"] = weighted_avg([
        (normalize(df["growing_heat_stress_days"],  8,  20), 0.35),
        (normalize_inverse(df["growing_total_rain"], 100, 30), 0.25),
        (normalize_inverse(df["growing_GDD"],        900, 400), 0.25),
        (normalize(df["growing_max_dry_streak"],    15, 35), 0.15),
    ]).clip(0, 100)

    # ── Boll Forming Risk ─────────────────────────────────────────────────
    df["boll_risk_score"] = weighted_avg([
        (normalize(df["boll_forming_temp_max_mean"], 36, 41), 0.45),
        (normalize(df["boll_forming_dry_days"],      20, 40), 0.30),
        (normalize(df["boll_forming_et0_total"],     150, 220), 0.25),
    ]).clip(0, 100)

    # ── Harvest Risk ──────────────────────────────────────────────────────
    df["harvest_risk_score"] = weighted_avg([
        (normalize(df["harvest_rainy_days"],    12, 25), 0.45),
        (normalize(df["harvest_humidity_mean"], 65, 85), 0.35),
        (normalize(df["harvest_frost_days"],    2,  8),  0.20),
    ]).clip(0, 100)

    # ── Overall Risk ──────────────────────────────────────────────────────
    df["overall_risk_score"] = weighted_avg([
        (df["planting_risk_score"], 0.15),
        (df["growing_risk_score"],  0.40),
        (df["boll_risk_score"],     0.30),
        (df["harvest_risk_score"],  0.15),
    ]).clip(0, 100)

    # ── Risk Labels from yield deviation ─────────────────────────────────
    region_avg = df.groupby("region")["yield_tonnes"].transform("mean")
    region_std = df.groupby("region")["yield_tonnes"].transform("std")
    df["yield_deviation"]    = (df["yield_tonnes"] - region_avg) / region_std
    df["overall_risk_label"] = (df["yield_deviation"] < -0.5).astype(int)

    for stage in ["planting", "growing", "boll", "harvest"]:
        col       = f"{stage}_risk_score"
        threshold = df[col].quantile(0.60)
        df[f"{stage}_risk_label"] = (df[col] >= threshold).astype(int)

    # Validate
    avg_by_label = df.groupby("overall_risk_label")["yield_tonnes"].mean()
    log(f"\n  Risk label validation:")
    log(f"    Safe  (label=0) avg yield: {avg_by_label.get(0, 0):.1f} tonnes")
    log(f"    Risky (label=1) avg yield: {avg_by_label.get(1, 0):.1f} tonnes")

    if avg_by_label.get(0, 0) > avg_by_label.get(1, 0):
        log("    Validation PASSED ✓ (safe > risky as expected)")
    else:
        log("    Validation WARNING ✗ (check thresholds)")

    con.execute("DROP TABLE IF EXISTS features_with_risk")
    con.execute("CREATE TABLE features_with_risk AS SELECT * FROM df")
    log("\n  features_with_risk → DuckDB ✓")

    return df


def run_features():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 2 — FEATURE ENGINEERING PIPELINE")
    log("=" * 55)

    build_features(con)
    build_risk_labels(con)

    log("\n" + "=" * 55)
    log("FEATURE ENGINEERING COMPLETE — DuckDB Tables:")
    log("=" * 55)
    for t in con.execute("SHOW TABLES").fetchall():
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        cols  = len(con.execute(f"SELECT * FROM {t[0]} LIMIT 1").description)
        log(f"  {t[0]:<25} {count:>6} rows  {cols:>3} cols")


    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    output_path = os.path.join(BASE_DIR, "data", "features_with_risk.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = con.execute("SELECT * FROM features_with_risk").df()
    df.to_csv(output_path, index=False)

    log(f"CSV saved at {output_path} ✓")
    
    con.close()
    log("\n  Next: Member 3 runs src/quality_checks.py then models")


if __name__ == "__main__":
    run_features()