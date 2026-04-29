# src/pipeline.py
# FULL PROJECT PIPELINE (Member 1 → Member 2 → Member 3 → Member 4)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion     import run_ingestion
from cleaning      import run_cleaning
from features      import run_features
from quality_checks import run_checks
from models        import run_models
from reports       import run_reports


def run_full_pipeline():
    print("\n" + "=" * 60)
    print("FULL COTTON YIELD PIPELINE")
    print("Steps: Ingestion → Cleaning → Features → Quality → Models → Reports")
    print("=" * 60)

    print("\n[STEP 1] INGESTION")
    run_ingestion()

    print("\n[STEP 2] CLEANING")
    run_cleaning()

    print("\n[STEP 3] FEATURE ENGINEERING")
    run_features()

    print("\n[STEP 4] QUALITY CHECKS")
    run_checks()

    print("\n[STEP 5] MODEL TRAINING & PREDICTION")
    run_models()

    print("\n[STEP 6] REPORTS & VISUALIZATIONS")
    run_reports()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("Outputs:")
    print("  - All data tables in DuckDB")
    print("  - Model predictions for 2025")
    print("  - Figures saved to reports/figures/")
    print("  - CSV exported to reports/")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline() 