# src/reports.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import duckdb
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH, FIGURES_DIR, REPORTS_DIR, LOGS_DIR


def log(msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Actual predictions table columns ─────────────────────────────────────────
# region, year, pred_yield, avg_yield, pct_change,
# planting_risk_pct, growing_risk_pct, boll_risk_pct, harvest_risk_pct


def run_eda(con):
    log("=" * 55)
    log("STEP 11 — Exploratory Data Analysis")
    log("=" * 55)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Chart 1: Yield over time per district ─────────────────────────────
    df_yield = con.execute("""
        SELECT year, region, yield_tonnes
        FROM clean_cotton
        ORDER BY region, year
    """).df()

    fig, ax = plt.subplots(figsize=(14, 6))
    for region, grp in df_yield.groupby("region"):
        ax.plot(grp["year"], grp["yield_tonnes"],
                marker="o", linewidth=1.5, markersize=3,
                label=region.replace(" district", ""))
    ax.set_title("Cotton Yield Over Time by District (2000–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (tonnes)")
    ax.legend(fontsize=7, ncol=3, title="District")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_yield_over_time.png"), dpi=150)
    plt.close()
    log("  Saved: 01_yield_over_time.png ✓")

    # ── Chart 2: Average yield per district ───────────────────────────────
    df_dist = con.execute("""
        SELECT region, AVG(yield_tonnes) AS avg_yield
        FROM clean_cotton
        GROUP BY region
        ORDER BY avg_yield ASC
    """).df()

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df_dist)))
    ax.barh(
        [r.replace(" district", "") for r in df_dist["region"]],
        df_dist["avg_yield"], color=colors
    )
    ax.set_title("Average Cotton Yield per District (2000–2024)")
    ax.set_xlabel("Average Yield (tonnes)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_yield_per_district.png"), dpi=150)
    plt.close()
    log("  Saved: 02_yield_per_district.png ✓")

    # ── Chart 3: Correlation heatmap (weather features vs yield) ─────────
    df_feat = con.execute("SELECT * FROM features").df()
    feat_cols = [c for c in df_feat.columns if any(
        s in c for s in ["planting_", "growing_", "boll_", "harvest_"]
    )]

    corr = (
        df_feat[feat_cols + ["yield_tonnes"]]
        .corr()[["yield_tonnes"]]
        .drop("yield_tonnes")
        .sort_values("yield_tonnes", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(5, 14))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation with Yield")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_correlation_heatmap.png"), dpi=150)
    plt.close()
    log("  Saved: 03_correlation_heatmap.png ✓")

    # ── Chart 4: Risk scores over time (from features_with_risk) ─────────
    df_risk = con.execute("""
        SELECT year,
               AVG(planting_risk_score) AS planting,
               AVG(growing_risk_score)  AS growing,
               AVG(boll_risk_score)     AS boll,
               AVG(harvest_risk_score)  AS harvest
        FROM features_with_risk
        GROUP BY year
        ORDER BY year
    """).df()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.flat, [
        ("planting", "Planting Risk (Mar–Apr)"),
        ("growing",  "Growing Risk (May–Aug)"),
        ("boll",     "Boll Forming Risk (Aug–Sep)"),
        ("harvest",  "Harvest Risk (Sep–Nov)"),
    ]):
        ax.fill_between(df_risk["year"], df_risk[col], alpha=0.3, color="orange")
        ax.plot(df_risk["year"], df_risk[col], color="red", linewidth=2)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Risk Score")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Average Stage Risk Scores Over Time (All Districts)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_risk_over_time.png"), dpi=150)
    plt.close()
    log("  Saved: 04_risk_over_time.png ✓")

    # ── Chart 5: Growing GDD & rain vs yield ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (col, label) in zip(axes, [
        ("growing_GDD",        "Growing GDD"),
        ("growing_total_rain", "Growing Rain (mm)"),
    ]):
        ax.scatter(df_feat[col], df_feat["yield_tonnes"],
                   alpha=0.4, s=20, color="steelblue")
        ax.set_xlabel(label)
        ax.set_ylabel("Yield (tonnes)")
        ax.set_title(f"{label} vs Yield")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Key Growing Season Features vs Cotton Yield")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_gdd_rain_vs_yield.png"), dpi=150)
    plt.close()
    log("  Saved: 05_gdd_rain_vs_yield.png ✓")

    # ── Chart 6: Safe vs risky yield distribution ─────────────────────────
    df_risk_feat = con.execute("SELECT * FROM features_with_risk").df()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [
        (0, "seagreen", "Safe (label=0)"),
        (1, "tomato",   "Risky (label=1)")
    ]:
        sub = df_risk_feat[
            df_risk_feat["overall_risk_label"] == label
        ]["yield_tonnes"]
        ax.hist(sub, bins=25, alpha=0.5, color=color,
                label=f"{name} (n={len(sub)})")
        ax.axvline(sub.mean(), color=color, linestyle="--", linewidth=2)
    ax.set_title("Yield Distribution: Safe vs Risky Years")
    ax.set_xlabel("Yield (tonnes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_yield_safe_vs_risky.png"), dpi=150)
    plt.close()
    log("  Saved: 06_yield_safe_vs_risky.png ✓")

    log(f"\n  EDA complete — 6 charts saved to {FIGURES_DIR}")


def run_forecast_report(con):
    log("=" * 55)
    log("STEP 12 — Forecast Dashboard & Report")
    log("=" * 55)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Actual column names: region, year, pred_yield, avg_yield, pct_change,
    # planting_risk_pct, growing_risk_pct, boll_risk_pct, harvest_risk_pct
    predictions = con.execute("SELECT * FROM predictions").df()

    # ── Chart 7: Predictions vs historical average ────────────────────────
    yr_df = predictions.sort_values("pred_yield", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    x     = np.arange(len(yr_df))
    width = 0.35
    ax.bar(x - width/2, yr_df["avg_yield"],
           width, label="Historical Avg (2000–2021)",
           color="steelblue", alpha=0.7)
    ax.bar(x + width/2, yr_df["pred_yield"],
           width, label="2025 Prediction",
           color=["seagreen" if p >= h else "tomato"
                  for p, h in zip(yr_df["pred_yield"], yr_df["avg_yield"])],
           alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r.replace(" district", "") for r in yr_df["region"]],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Yield (tonnes)")
    ax.set_title("2025 Cotton Yield Predictions vs Historical Average")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    total_pred = yr_df["pred_yield"].sum()
    total_hist = yr_df["avg_yield"].sum()
    pct        = (total_pred - total_hist) / total_hist * 100
    ax.text(0.99, 0.97,
            f"Total: {total_pred:.0f}t vs {total_hist:.0f}t ({pct:+.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "07_predictions_2025.png"), dpi=150)
    plt.close()
    log("  Saved: 07_predictions_2025.png ✓")

    # ── Chart 8: Risk heatmap ─────────────────────────────────────────────
    risk_mat = predictions.set_index("region")[[
        "planting_risk_pct", "growing_risk_pct",
        "boll_risk_pct", "harvest_risk_pct"
    ]].copy()
    risk_mat.columns = ["Planting", "Growing", "Boll", "Harvest"]
    risk_mat.index   = [i.replace(" district", "") for i in risk_mat.index]
    risk_mat = risk_mat.sort_values("Growing", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(risk_mat, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=0, vmax=100, ax=ax, linewidths=0.4,
                cbar_kws={"label": "Risk %"}, annot_kws={"size": 8})
    ax.set_title("2025 Stage Risk Heatmap — All Districts")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "08_risk_heatmap_2025.png"), dpi=150)
    plt.close()
    log("  Saved: 08_risk_heatmap_2025.png ✓")

    # ── Chart 9: Predicted yield vs pct change bubble chart ───────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["seagreen" if p >= 0 else "tomato" for p in predictions["pct_change"]]
    scatter = ax.scatter(
        predictions["pred_yield"],
        predictions["pct_change"],
        c=colors,
        s=120, alpha=0.8, edgecolors="white", linewidths=0.5
    )
    for _, row in predictions.iterrows():
        ax.annotate(
            row["region"].replace(" district", ""),
            (row["pred_yield"], row["pct_change"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(3, 3), textcoords="offset points"
        )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Predicted Yield 2025 (tonnes)")
    ax.set_ylabel("Change vs Historical Average (%)")
    ax.set_title("2025 Predicted Yield vs % Change from Historical Average")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "09_yield_vs_pct_change.png"), dpi=150)
    plt.close()
    log("  Saved: 09_yield_vs_pct_change.png ✓")

    # ── Chart 10: Summary dashboard ───────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Panel 1: yield over time all districts
    ax1 = fig.add_subplot(gs[0, :])
    df_yield = con.execute("""
        SELECT year, region, yield_tonnes
        FROM clean_cotton ORDER BY region, year
    """).df()
    for region, grp in df_yield.groupby("region"):
        ax1.plot(grp["year"], grp["yield_tonnes"],
                 marker="o", markersize=2, linewidth=1.5, alpha=0.7,
                 label=region.replace(" district", ""))
    ax1.axvline(2021, color="gray", linestyle="--",
                alpha=0.6, label="Train/test split")
    ax1.set_title("Historical Yield by District + Train/Test Split")
    ax1.set_ylabel("Yield (tonnes)")
    ax1.legend(fontsize=6, ncol=5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: 2025 risk heatmap
    ax2     = fig.add_subplot(gs[1, 0])
    r_mat   = predictions.set_index("region")[[
        "planting_risk_pct", "growing_risk_pct",
        "boll_risk_pct", "harvest_risk_pct"
    ]].copy()
    r_mat.columns = ["Plant", "Grow", "Boll", "Harv"]
    r_mat.index   = [i.replace(" district", "") for i in r_mat.index]
    r_mat = r_mat.sort_values("Grow", ascending=False)
    sns.heatmap(r_mat, annot=True, fmt=".0f", cmap="RdYlGn_r",
                vmin=0, vmax=100, ax=ax2, linewidths=0.3,
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 7})
    ax2.set_title("2025 Stage Risk by District", fontsize=11)
    ax2.tick_params(labelsize=7)

    # Panel 3: predicted yield bar
    ax3     = fig.add_subplot(gs[1, 1])
    p_sorted = predictions.sort_values("pred_yield", ascending=True)
    bar_colors = ["seagreen" if p >= h else "tomato"
                  for p, h in zip(p_sorted["pred_yield"], p_sorted["avg_yield"])]
    ax3.barh(
        [r.replace(" district", "") for r in p_sorted["region"]],
        p_sorted["pred_yield"],
        color=bar_colors, alpha=0.8
    )
    ax3.axvline(predictions["avg_yield"].mean(), color="navy",
                linestyle="--", linewidth=1.5,
                label=f"Avg historical ({predictions['avg_yield'].mean():.1f}t)")
    ax3.set_title("2025 Predicted Yield per District", fontsize=11)
    ax3.set_xlabel("Yield (tonnes)")
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "Azerbaijan Cotton Yield Prediction 2025 — Summary Dashboard",
        fontsize=15, fontweight="bold"
    )
    plt.savefig(
        os.path.join(FIGURES_DIR, "10_summary_dashboard.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    log("  Saved: 10_summary_dashboard.png ✓")

    # ── Export predictions CSV ────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(REPORTS_DIR, "predictions_2025.csv")
    predictions.to_csv(csv_path, index=False)
    log(f"  Saved: predictions_2025.csv → {csv_path} ✓")


def print_final_summary(con):
    log("\n" + "=" * 65)
    log("FINAL PROJECT SUMMARY")
    log("=" * 65)

    stats = con.execute("""
        SELECT
            COUNT(DISTINCT region) AS districts,
            MIN(year)              AS min_year,
            MAX(year)              AS max_year
        FROM clean_cotton
    """).df()

    predictions = con.execute("SELECT * FROM predictions").df()

    total_pred = predictions["pred_yield"].sum()
    total_hist = predictions["avg_yield"].sum()
    pct_total  = (total_pred - total_hist) / total_hist * 100

    best_row   = predictions.loc[predictions["pred_yield"].idxmax()]
    worst_row  = predictions.loc[predictions["pct_change"].idxmin()]

    avg_risk = predictions[[
        "planting_risk_pct", "growing_risk_pct",
        "boll_risk_pct", "harvest_risk_pct"
    ]].mean().mean()

    log(f"""
  DATA
    Districts:        {stats['districts'][0]} regions
    Historical years: {stats['min_year'][0]} – {stats['max_year'][0]}
    Database:         DuckDB (cotton_project.duckdb)

  PIPELINE TABLES
    raw_cotton          → original reshaped data
    raw_weather         → original API data
    clean_cotton        → nulls removed, districts mapped
    clean_weather       → sensor errors fixed, 2000–2025
    features            → 40 weather features per district/year
    features_with_risk  → + risk scores and labels
    predictions         → 2025 forecasts (15 districts)

  MODEL
    Target:           yield_anomaly (deviation from district mean)
    Final model:      trained on all 2000–2024 data
    Features:         z-scored weather + district mean yield
    Note:             Weather explains modest year-to-year variance;
                      district baselines dominate absolute yield levels.

  2025 PREDICTIONS
    Total predicted:  {total_pred:.0f} tonnes
    Historical total: {total_hist:.0f} tonnes
    Overall change:   {pct_total:+.1f}%
    Avg stage risk:   {avg_risk:.1f}%
    Best district:    {best_row['region']} ({best_row['pred_yield']:.1f}t)
    Most at risk:     {worst_row['region']} ({worst_row['pct_change']:+.1f}% vs avg)
""")

    log("  District breakdown:")
    log(f"  {'District':<30} {'Pred':>6} {'Avg':>6} {'Chg':>7}  Plant  Grow  Boll  Harv")
    log("  " + "-" * 65)
    for _, r in predictions.sort_values("pred_yield", ascending=False).iterrows():
        log(
            f"  {r['region']:<30} {r['pred_yield']:>6.1f} "
            f"{r['avg_yield']:>6.1f} {r['pct_change']:>+7.1f}%  "
            f"{r['planting_risk_pct']:>3}%  {r['growing_risk_pct']:>3}%  "
            f"{r['boll_risk_pct']:>3}%  {r['harvest_risk_pct']:>3}%"
        )

    log("\n" + "=" * 65)
    log("PROJECT COMPLETE ✓")
    log("=" * 65)


def run_reports():
    con = duckdb.connect(DB_PATH)

    log("\n" + "=" * 55)
    log("MEMBER 4 — EDA, VISUALIZATION & REPORTING")
    log("=" * 55)

    run_eda(con)
    run_forecast_report(con)
    print_final_summary(con)

    con.close()


if __name__ == "__main__":
    run_reports()  