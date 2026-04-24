# src/pipeline.py
# Member 1 runs this single file to execute the full ingestion + cleaning pipeline

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion import run_ingestion
from cleaning  import run_cleaning

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("MEMBER 1 — FULL DATA PIPELINE")
    print("Running: ingestion → cleaning → DuckDB")
    print("=" * 55)

    run_ingestion()
    run_cleaning()

    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("All raw and cleaned data saved to DuckDB.")
    print("Next: Member 2 runs src/features.py")
    print("=" * 55) 