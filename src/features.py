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
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
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

    # ── IMPORTANT: only raw weather features go into this table ──────────
    # Risk scores and risk labels are NOT computed here.
    # They were causing data leakage: labels derived from yield_tonnes
    # (the target), and quantile thresholds computed on the full dataset
    # before the train/test split. Both leak test-set information into
    # training.
    # ─────────────────────────────────────────────────────────────────────

    con.execute("DROP TABLE IF EXISTS features")
    con.execute("CREATE TABLE features AS SELECT * FROM df_feat")
    log("  features → DuckDB ✓")
    log("  NOTE: No risk scores/labels — raw weather features only.")

    return df_feat


def run_features():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 2 — FEATURE ENGINEERING PIPELINE")
    log("=" * 55)

    build_features(con)

    log("\n" + "=" * 55)
    log("FEATURE ENGINEERING COMPLETE — DuckDB Tables:")
    log("=" * 55)
    for t in con.execute("SHOW TABLES").fetchall():
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        cols  = len(con.execute(f"SELECT * FROM {t[0]} LIMIT 1").description)
        log(f"  {t[0]:<25} {count:>6} rows  {cols:>3} cols")

    con.close()
    log("\n  Next: Member 3 runs src/quality_checks.py then models.py")


if __name__ == "__main__":
    run_features()