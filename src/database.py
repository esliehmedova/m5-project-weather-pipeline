# src/database.py
# ─────────────────────────────────────────────────────────────────────────────
# DATABASE / STORAGE LAYER
# Handles saving and loading all project datasets consistently
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import os
from config import (
    PROCESSED_DIR, FINAL_DIR, RAW_WEATHER_DIR,
    COTTON_LONG_PATH, WEATHER_ALL_PATH
)


def save_dataset(df: pd.DataFrame, path: str, description: str = ""):
    """Saves a DataFrame to CSV with confirmation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved {description}: {len(df)} rows × {len(df.columns)} cols → {path}")


def load_cotton() -> pd.DataFrame:
    """Loads the cleaned long-format cotton dataset."""
    df = pd.read_csv(COTTON_LONG_PATH)
    print(f"Loaded cotton: {df.shape} | years {df['year'].min()}–{df['year'].max()}")
    return df


def load_weather() -> pd.DataFrame:
    """Loads the combined daily weather dataset."""
    df = pd.read_csv(WEATHER_ALL_PATH, parse_dates=["date"])
    print(f"Loaded weather: {df.shape} | {df['region'].nunique()} stations")
    return df


def load_ml_dataset(with_risk: bool = False) -> pd.DataFrame:
    """
    Loads the final ML-ready dataset.
    with_risk=True loads the version with risk scores and labels.
    """
    filename = "ml_dataset_with_risk.csv" if with_risk else "ml_dataset.csv"
    path = os.path.join(FINAL_DIR, filename)
    df   = pd.read_csv(path)
    print(f"Loaded ML dataset ({'with risk' if with_risk else 'base'}): {df.shape}")
    return df


def load_predictions() -> pd.DataFrame:
    """Loads the 2025 predictions results table."""
    path = os.path.join("reports", "predictions_2025.csv")
    df   = pd.read_csv(path)
    print(f"Loaded predictions: {df.shape}")
    return df


def dataset_summary():
    """Prints a full summary of all available datasets."""
    print("\n" + "="*55)
    print("DATASET INVENTORY")
    print("="*55)

    files = {
        "Cotton (long format)":   COTTON_LONG_PATH,
        "Weather (all stations)": WEATHER_ALL_PATH,
        "ML dataset (base)":      os.path.join(FINAL_DIR, "ml_dataset.csv"),
        "ML dataset (with risk)": os.path.join(FINAL_DIR, "ml_dataset_with_risk.csv"),
    }

    for name, path in files.items():
        if os.path.exists(path):
            df   = pd.read_csv(path)
            size = os.path.getsize(path) / 1024
            print(f"  {name:<28} {str(df.shape):<15} {size:.1f} KB")
        else:
            print(f"  {name:<28} NOT FOUND")

    print("="*55)


if __name__ == "__main__":
    dataset_summary() 