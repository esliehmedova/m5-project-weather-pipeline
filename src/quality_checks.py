# src/quality_checks.py
import pandas as pd
import duckdb
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH, LOGS_DIR, TRAIN_UNTIL_YEAR


def log(msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_quality_checks():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 3 — QUALITY CHECKS")
    log("=" * 55)

    # Check 1: No nulls in features
    log("\nCheck 1 — Null values in features_with_risk:")
    nulls = con.execute("""
        SELECT COUNT(*) AS total_rows,
               COUNT(*) FILTER (WHERE yield_tonnes IS NULL) AS null_yield,
               COUNT(*) FILTER (WHERE growing_GDD IS NULL) AS null_gdd,
               COUNT(*) FILTER (WHERE overall_risk_score IS NULL) AS null_risk
        FROM features_with_risk
    """).df()
    log(f"  Total rows:   {nulls['total_rows'][0]}")
    log(f"  Null yield:   {nulls['null_yield'][0]}")
    log(f"  Null GDD:     {nulls['null_gdd'][0]}")
    log(f"  Null risk:    {nulls['null_risk'][0]}")
    log(f"  {'PASSED ✓' if nulls['null_yield'][0] == 0 else 'FAILED ✗'}")

    # Check 2: Train/test split integrity
    log("\nCheck 2 — Train/test split:")
    split = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE year <= {TRAIN_UNTIL_YEAR}) AS train_rows,
            COUNT(*) FILTER (WHERE year >  {TRAIN_UNTIL_YEAR}) AS test_rows,
            COUNT(DISTINCT region) AS districts
        FROM features_with_risk
    """).df()
    log(f"  Train rows (≤{TRAIN_UNTIL_YEAR}): {split['train_rows'][0]}")
    log(f"  Test  rows (>{TRAIN_UNTIL_YEAR}):  {split['test_rows'][0]}")
    log(f"  Districts:              {split['districts'][0]}")
    log(f"  PASSED ✓")

    # Check 3: Risk label balance
    log("\nCheck 3 — Risk label distribution:")
    labels = con.execute("""
        SELECT overall_risk_label,
               COUNT(*) AS count,
               ROUND(AVG(yield_tonnes), 2) AS avg_yield
        FROM features_with_risk
        GROUP BY overall_risk_label
        ORDER BY overall_risk_label
    """).df()
    for _, row in labels.iterrows():
        log(f"  Label {int(row['overall_risk_label'])}: "
            f"{int(row['count'])} rows, "
            f"avg yield = {row['avg_yield']} tonnes")
    log(f"  {'PASSED ✓' if labels['avg_yield'][0] > labels['avg_yield'][1] else 'WARNING ✗'}")

    # Check 4: Year range
    log("\nCheck 4 — Year range coverage:")
    years = con.execute("""
        SELECT MIN(year) AS min_year,
               MAX(year) AS max_year,
               COUNT(DISTINCT year) AS total_years
        FROM features_with_risk
    """).df()
    log(f"  Years: {years['min_year'][0]} – {years['max_year'][0]}")
    log(f"  Total unique years: {years['total_years'][0]}")
    log(f"  PASSED ✓")

    con.close()
    log("\n  Quality checks complete. Proceeding to model training.")


if __name__ == "__main__":
    run_quality_checks()