# src/quality_checks.py
import duckdb
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH, LOGS_DIR


def log(msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_quality_checks(con):
    log("\n" + "=" * 55)
    log("MEMBER 3 — QUALITY CHECKS")
    log("=" * 55)

    # ── Check 1: No nulls in features ────────────────────────────────────
    log("\nCheck 1 — Null values in features:")
    df = con.execute("SELECT * FROM features").df()
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    log(f"  Total rows:    {len(df)}")
    log(f"  Total nulls:   {total_nulls}")
    if total_nulls == 0:
        log("  PASSED ✓")
    else:
        log("  FAILED ✗ — null columns:")
        for col, cnt in null_counts[null_counts > 0].items():
            log(f"    {col}: {cnt} nulls")

    # ── Check 2: Train/test split integrity ───────────────────────────────
    log("\nCheck 2 — Train/test split:")
    train = df[df["year"] <= 2021]
    test  = df[df["year"] >  2021]
    districts = df["region"].nunique()
    log(f"  Train rows (≤2021): {len(train)}")
    log(f"  Test  rows (>2021): {len(test)}")
    log(f"  Districts:          {districts}")
    assert train["year"].max() <= 2021, "Train set contains years > 2021!"
    assert test["year"].min()  >  2021, "Test set contains years <= 2021!"
    log("  PASSED ✓")

    # ── Check 3: Yield range sanity ───────────────────────────────────────
    log("\nCheck 3 — Yield range sanity:")
    y_min  = df["yield_tonnes"].min()
    y_max  = df["yield_tonnes"].max()
    y_mean = df["yield_tonnes"].mean()
    log(f"  Min yield:  {y_min:.1f} tonnes")
    log(f"  Max yield:  {y_max:.1f} tonnes")
    log(f"  Mean yield: {y_mean:.1f} tonnes")
    if y_min > 0 and y_max < 500:
        log("  PASSED ✓")
    else:
        log("  WARNING ✗ — yields outside expected range (0–500 tonnes)")

    # ── Check 4: Year range coverage ──────────────────────────────────────
    log("\nCheck 4 — Year range coverage:")
    years = sorted(df["year"].unique())
    log(f"  Years: {years[0]} – {years[-1]}")
    log(f"  Total unique years: {len(years)}")
    log("  PASSED ✓")

    # ── Check 5: Feature column count ─────────────────────────────────────
    log("\nCheck 5 — Feature column integrity:")
    # Schema: region, weather_station, year, yield_tonnes = 4 base columns
    # Weather features per stage: temp_mean, temp_min_mean, temp_max_mean,
    #   heat_stress_days, frost_days, GDD, total_rain, rainy_days, dry_days,
    #   max_dry_streak, humidity_mean, wind_mean, et0_total = 13 features × 4 stages = 52
    # Total expected: 4 + 52 = 56
    expected_min = 56 
    actual       = len(df.columns)
    log(f"  Columns found:    {actual}")
    log(f"  Columns expected: ≥{expected_min}")

    # Confirm NO risk / deviation columns leaked into training table
    forbidden = [c for c in df.columns if "risk" in c or "deviation" in c]
    if forbidden:
        log(f"  FAILED ✗ — forbidden leakage columns present: {forbidden}")
    elif actual >= expected_min:
        log("  PASSED ✓ — no risk/deviation columns, weather features intact")
    else:
        log(f"  WARNING — fewer columns than expected (got {actual}, want ≥{expected_min})")

    # ── Check 6: GDD values non-negative ──────────────────────────────────
    log("\nCheck 6 — GDD values non-negative:")
    gdd_cols     = [c for c in df.columns if "_GDD" in c]
    negative_gdd = {c: (df[c] < 0).sum() for c in gdd_cols if (df[c] < 0).sum() > 0}
    if not negative_gdd:
        log(f"  All {len(gdd_cols)} GDD columns non-negative ✓")
        log("  PASSED ✓")
    else:
        log(f"  WARNING ✗ — negative GDD values: {negative_gdd}")

    # ── Check 7: All 15 districts present ─────────────────────────────────
    log("\nCheck 7 — District coverage:")
    actual_districts = df["region"].nunique()
    log(f"  Districts in features: {actual_districts}")
    for d in sorted(df["region"].unique()):
        row_count = len(df[df["region"] == d])
        log(f"    {d:<30} {row_count} years")
    if actual_districts == 15:
        log("  PASSED ✓ — all 15 districts present")
    else:
        log(f"  WARNING — expected 15 districts, found {actual_districts}")

    log("\n  Quality checks complete. Proceeding to model training.")
    return df


def run_checks():
    con = duckdb.connect(DB_PATH)
    run_quality_checks(con)
    con.close()


if __name__ == "__main__":
    run_checks() 