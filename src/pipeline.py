# src/pipeline.py
# FULL PROJECT PIPELINE (Member 1 → Member 2 → Member 3)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion import run_ingestion
from cleaning import run_cleaning
from features import run_feature_engineering
from quality_checks import run_quality_checks
from models import run_model_pipeline


def run_full_pipeline():
    print("\n" + "=" * 60)
    print("FULL COTTON YIELD PIPELINE")
    print("Steps: Ingestion → Cleaning → Features → Quality → Models")
    print("=" * 60)

    # MEMBER 1
    print("\n[STEP 1] INGESTION")
    run_ingestion()

    print("\n[STEP 2] CLEANING")
    run_cleaning()

    # MEMBER 2
    print("\n[STEP 3] FEATURE ENGINEERING")
    run_feature_engineering()

    print("\n[STEP 4] QUALITY CHECKS")
    run_quality_checks()

    # MEMBER 3
    print("\n[STEP 5] MODEL TRAINING & PREDICTION")
    run_model_pipeline()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("Outputs:")
    print("- Clean data in DuckDB")
    print("- Feature tables ready")
    print("- Model predictions for 2025–2026")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline() 