# src/cleaning.py
import pandas as pd
import numpy as np
import duckdb
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH, REGION_TO_WEATHER, LOGS_DIR


def log(msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean_cotton(con):
    log("=" * 55)
    log("STEP 3 — Cleaning Cotton Dataset")
    log("=" * 55)

    df = con.execute("""
        SELECT region, year, yield_tonnes
        FROM raw_cotton
        ORDER BY region, year
    """).df()

    # Strip whitespace
    df["region"] = df["region"].str.strip()

    # Convert dashes / ellipses to NaN
    df["yield_tonnes"] = pd.to_numeric(
        df["yield_tonnes"].replace(["-", "…", "..."], np.nan),
        errors="coerce"
    )

    log(f"   Rows loaded from raw_cotton: {len(df)}")
    log(f"   Null yields: {df['yield_tonnes'].isnull().sum()}")

    # Drop districts that have ANY null yield
    null_districts = df[df["yield_tonnes"].isnull()]["region"].unique()
    if len(null_districts) > 0:
        log(f"\n  Districts dropped (had null values):")
        for d in sorted(null_districts):
            log(f"    ✗ {d}")
        df = df[~df["region"].isin(null_districts)].copy()
    else:
        log("  No null districts found.")

    # REGION_TO_WEATHER is a set — keep only districts present in it
    # (each district maps to itself as the weather station name)
    unmapped = df[~df["region"].isin(REGION_TO_WEATHER)]["region"].unique()
    if len(unmapped) > 0:
        log(f"\n  Districts not in REGION_TO_WEATHER (dropped):")
        for d in unmapped:
            log(f"    ✗ {d}")
        df = df[df["region"].isin(REGION_TO_WEATHER)].copy()

    # weather_station == region (1-to-1 mapping)
    df["weather_station"] = df["region"]

    log(f"\n  Rows after cleaning:  {len(df)}")
    log(f"  Districts remaining:  {df['region'].nunique()}")
    log(f"  Nulls remaining:      {df['yield_tonnes'].isnull().sum()}")

    con.execute("DROP TABLE IF EXISTS clean_cotton")
    con.execute("CREATE TABLE clean_cotton AS SELECT * FROM df")
    log("  clean_cotton table saved to DuckDB ✓")

    return df


def clean_weather(con):
    log("=" * 55)
    log("STEP 4 — Cleaning Weather Dataset")
    log("=" * 55)

    # 2000–2025: training (2000–2024) + prediction input (2025)
    # 2026 excluded — only partial year available
    df = con.execute("""
        SELECT *
        FROM raw_weather
        WHERE year BETWEEN 2000 AND 2025
        ORDER BY region, date
    """).df()

    log(f"   Rows loaded from raw_weather (2000–2025): {len(df)}")

    # Only the 4 columns that config now fetches
    numeric_cols = ["temp_mean", "precipitation", "humidity_mean", "wind_speed"]

    # Fix obvious sensor / API errors
    df.loc[df["temp_mean"] > 60,       "temp_mean"]     = np.nan
    df.loc[df["temp_mean"] < -40,      "temp_mean"]     = np.nan
    df.loc[df["precipitation"] < 0,    "precipitation"] = 0.0
    df.loc[df["humidity_mean"] > 100,  "humidity_mean"] = 100.0
    df.loc[df["humidity_mean"] < 0,    "humidity_mean"] = 0.0
    df.loc[df["wind_speed"] < 0,       "wind_speed"]    = np.nan

    # Interpolate short gaps (up to 3 consecutive days) per station
    df = df.sort_values(["region", "date"])
    for col in numeric_cols:
        df[col] = df.groupby("region")[col].transform(
            lambda x: x.interpolate(method="linear", limit=3)
        )

    log("   Sensor errors fixed.")
    log("   Nulls after interpolation:")
    for col in numeric_cols:
        n = df[col].isnull().sum()
        if n > 0:
            log(f"     {col}: {n}")

    con.execute("DROP TABLE IF EXISTS clean_weather")
    con.execute("CREATE TABLE clean_weather AS SELECT * FROM df")
    log("   clean_weather table saved to DuckDB ✓")

    return df


def verify_alignment(con):
    log("=" * 55)
    log("STEP 5 — Alignment Verification")
    log("=" * 55)

    result = con.execute("""
        SELECT
            MIN(c.year)              AS cotton_min_year,
            MAX(c.year)              AS cotton_max_year,
            COUNT(DISTINCT c.region) AS cotton_districts,
            MIN(w.year)              AS weather_min_year,
            MAX(w.year)              AS weather_max_year,
            COUNT(DISTINCT w.region) AS weather_stations
        FROM clean_cotton c, clean_weather w
    """).df()

    log(f"   Cotton years:     {result['cotton_min_year'][0]} – {result['cotton_max_year'][0]}")
    log(f"   Cotton districts: {result['cotton_districts'][0]}")
    log(f"   Weather years:    {result['weather_min_year'][0]} – {result['weather_max_year'][0]}")
    log(f"   Weather stations: {result['weather_stations'][0]}")

    # Every cotton district must have a matching weather station
    missing = con.execute("""
        SELECT DISTINCT c.region, c.weather_station
        FROM clean_cotton c
        LEFT JOIN (
            SELECT DISTINCT region FROM clean_weather
        ) w ON c.weather_station = w.region
        WHERE w.region IS NULL
    """).df()

    if len(missing) > 0:
        log(f"\n  WARNING — Districts with no matching weather station:")
        for _, row in missing.iterrows():
            log(f"    ✗ {row['region']} → {row['weather_station']}")
    else:
        log("  All districts have matching weather stations ✓")


def run_cleaning():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 1 — DATA CLEANING PIPELINE")
    log("=" * 55)

    clean_cotton(con)
    clean_weather(con)
    verify_alignment(con)

    log("\n" + "=" * 55)
    log("CLEANING COMPLETE — DuckDB Tables:")
    log("=" * 55)
    tables = con.execute("SHOW TABLES").fetchall()
    for t in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        cols  = len(con.execute(f"SELECT * FROM {t[0]} LIMIT 1").description)
        log(f"  {t[0]:<25} {count:>6} rows  {cols:>3} cols")

    con.close()


if __name__ == "__main__":
    run_cleaning() 