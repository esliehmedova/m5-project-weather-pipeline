# src/database.py
# ─────────────────────────────────────────────────────────────────────────────
# DATABASE / STORAGE LAYER
# Handles saving and loading all project datasets from DuckDB
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import duckdb
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH, REPORTS_DIR, MODELS_DIR


def get_connection():
    """Returns a DuckDB connection to the project database."""
    return duckdb.connect(DB_PATH)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_raw_cotton() -> pd.DataFrame:
    """Loads the raw long-format cotton dataset from DuckDB."""
    con = get_connection()
    df  = con.execute("SELECT * FROM raw_cotton ORDER BY region, year").df()
    con.close()
    print(f"Loaded raw_cotton:    {df.shape} | years {df['year'].min()}–{df['year'].max()}")
    return df


def load_clean_cotton() -> pd.DataFrame:
    """Loads the cleaned cotton dataset from DuckDB."""
    con = get_connection()
    df  = con.execute("SELECT * FROM clean_cotton ORDER BY region, year").df()
    con.close()
    print(f"Loaded clean_cotton:  {df.shape} | districts {df['region'].nunique()}")
    return df


def load_raw_weather() -> pd.DataFrame:
    """Loads the raw daily weather dataset from DuckDB."""
    con = get_connection()
    df  = con.execute("SELECT * FROM raw_weather ORDER BY region, date").df()
    con.close()
    print(f"Loaded raw_weather:   {df.shape} | stations {df['region'].nunique()}")
    return df


def load_clean_weather() -> pd.DataFrame:
    """Loads the cleaned daily weather dataset from DuckDB."""
    con = get_connection()
    df  = con.execute("SELECT * FROM clean_weather ORDER BY region, date").df()
    con.close()
    print(f"Loaded clean_weather: {df.shape} | stations {df['region'].nunique()}")
    return df


def load_features() -> pd.DataFrame:
    """Loads the engineered feature table from DuckDB."""
    con = get_connection()
    df  = con.execute("SELECT * FROM features ORDER BY region, year").df()
    con.close()
    print(f"Loaded features:      {df.shape} | years {df['year'].min()}–{df['year'].max()}")
    return df


def load_ml_dataset(with_risk: bool = False) -> pd.DataFrame:
    """
    Loads the final ML-ready dataset from DuckDB.
    with_risk=True loads the version that includes risk scores and labels.
    """
    table = "ml_dataset_with_risk" if with_risk else "ml_dataset"
    con   = get_connection()
    df    = con.execute(f"SELECT * FROM {table} ORDER BY region, year").df()
    con.close()
    print(f"Loaded {table}: {df.shape}")
    return df


def load_predictions() -> pd.DataFrame:
    """Loads the predictions results table from DuckDB."""
    con = get_connection()
    df  = con.execute("SELECT * FROM predictions ORDER BY region, year").df()
    con.close()
    print(f"Loaded predictions:   {df.shape}")
    return df


# ── Savers ────────────────────────────────────────────────────────────────────

def save_table(df: pd.DataFrame, table_name: str, description: str = ""):
    """Saves a DataFrame to DuckDB as a named table (replaces if exists)."""
    con = get_connection()
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    con.close()
    label = description or table_name
    print(f"  Saved {label}: {count} rows × {len(df.columns)} cols → DuckDB [{table_name}]")


def save_predictions_csv(df: pd.DataFrame, filename: str = "predictions_2025.csv"):
    """Exports predictions to a CSV report file for sharing/reporting."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Predictions exported: {len(df)} rows → {path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def dataset_summary():
    """Prints a full summary of all tables currently in DuckDB."""
    print("\n" + "=" * 55)
    print("DATABASE INVENTORY — " + DB_PATH)
    print("=" * 55)

    con    = get_connection()
    tables = con.execute("SHOW TABLES").fetchall()

    if not tables:
        print("  No tables found.")
    else:
        for (name,) in tables:
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                cols  = len(con.execute(f"SELECT * FROM {name} LIMIT 1").description)
                print(f"  {name:<28} {count:>6} rows  {cols:>3} cols")
            except Exception as e:
                print(f"  {name:<28} ERROR: {e}")

    con.close()
    print("=" * 55)


if __name__ == "__main__":
    dataset_summary()  