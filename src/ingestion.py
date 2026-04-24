# src/ingestion.py
import pandas as pd
import os
import sys
import duckdb
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    LOCATIONS, WEATHER_VARIABLES, WEATHER_START_YEAR,
    WEATHER_END_YEAR, TIMEZONE, RAW_WEATHER_DIR,
    RAW_COTTON_PATH, DB_PATH, LOGS_DIR
)


def log(msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f: 
        f.write(line + "\n")


def get_openmeteo_client():
    cache_session = requests_cache.CachedSession(
        '.weather_cache', expire_after=-1
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_weather_for_location(client, name, lat, lon, start_year, end_year):
    log(f"  Fetching weather for {name} ({start_year}–{end_year})...")

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": f"{start_year}-01-01",
        "end_date":   f"{end_year}-12-31",
        "daily":      WEATHER_VARIABLES,
        "timezone":   TIMEZONE
    }

    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params
    )
    daily = responses[0].Daily()

    df = pd.DataFrame({
        "date": pd.date_range(
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

    df["region"] = name
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["day"]    = df["date"].dt.day
    df["date"]   = df["date"].astype(str)

    return df


def ingest_all_weather(con):
    os.makedirs(RAW_WEATHER_DIR, exist_ok=True)
    client  = get_openmeteo_client()
    all_dfs = []

    log("=" * 55)
    log("STEP 1 — Weather Ingestion")
    log("=" * 55)

    for name, info in LOCATIONS.items():
        df = fetch_weather_for_location(
            client, name,
            info["lat"], info["lon"],
            WEATHER_START_YEAR, WEATHER_END_YEAR
        )
        save_path = os.path.join(RAW_WEATHER_DIR, f"{name.lower()}_daily.csv")
        df.to_csv(save_path, index=False)
        log(f"    Saved {len(df)} rows → {save_path}")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save to DuckDB
    con.execute("DROP TABLE IF EXISTS raw_weather")
    con.execute("""
        CREATE TABLE raw_weather AS
        SELECT * FROM combined
    """)

    count = con.execute("SELECT COUNT(*) FROM raw_weather").fetchone()[0]
    log(f"\n  raw_weather table created in DuckDB")
    log(f"  Total rows:  {count}")
    log(f"  Stations:    {len(LOCATIONS)}")
    log(f"  Date range:  {WEATHER_START_YEAR}-01-01 → {WEATHER_END_YEAR}-12-31")

    return combined


def ingest_cotton(con):
    log("=" * 55)
    log("STEP 2 — Cotton Dataset Ingestion")
    log("=" * 55)

    df = pd.read_excel(RAW_COTTON_PATH)
    df = df.rename(columns={df.columns[0]: "region"})

    # Wide → Long reshape
    df_long = df.melt(
        id_vars="region",
        var_name="year",
        value_name="yield_tonnes"
    )
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")
    df_long = df_long.dropna(subset=["year"])
    df_long["year"] = df_long["year"].astype(int)
    df_long = df_long.sort_values(["region", "year"]).reset_index(drop=True)

    log(f"  Raw shape after reshape: {df_long.shape}")
    log(f"  Nulls in yield:          {df_long['yield_tonnes'].isnull().sum()}")
    log(f"  Districts:               {df_long['region'].nunique()}")
    log(f"  Years:                   {df_long['year'].min()} – {df_long['year'].max()}")

    # Save raw long format to DuckDB
    con.execute("DROP TABLE IF EXISTS raw_cotton")
    con.execute("""
        CREATE TABLE raw_cotton AS
        SELECT * FROM df_long
    """)

    log(f"\n  raw_cotton table created in DuckDB")

    return df_long


def run_ingestion():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 1 — DATA INGESTION PIPELINE")
    log("=" * 55)

    ingest_all_weather(con)
    ingest_cotton(con)

    # Summary
    log("\n" + "=" * 55)
    log("INGESTION COMPLETE — DuckDB Tables Created:")
    log("=" * 55)
    tables = con.execute("SHOW TABLES").fetchall()
    for t in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        cols  = len(con.execute(f"SELECT * FROM {t[0]} LIMIT 1").description)
        log(f"  {t[0]:<20} {count:>6} rows  {cols:>3} cols")

    con.close()
    log("\nDuckDB saved → " + DB_PATH)


if __name__ == "__main__":
    run_ingestion() 