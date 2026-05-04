"""
Generate fatigue vs. performance visualizations from nba_contextual.db.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "nba_contextual.db"
VIZ_DIR = ROOT / "visualizations"

HIGH_ALTITUDE_ARENAS = {"Ball Arena", "Vivint Arena"}


def load_master_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT
        p.performance_id,
        p.game_id,
        p.player_id,
        p.player_team,
        p.is_home,
        p.miles_traveled,
        p.days_rest,
        p.is_back_to_back,
        p.altitude_impact,
        p.mp,
        p.pts,
        p.fg_pct,
        p.adv_usg_pct,
        p.adv_ts_pct,
        g.game_date,
        g.home_team,
        g.visitor_team,
        g.arena_name
    FROM Performances p
    JOIN Games g ON p.game_id = g.game_id
    """
    return pd.read_sql_query(query, conn)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = [
        "miles_traveled",
        "days_rest",
        "is_back_to_back",
        "altitude_impact",
        "pts",
        "adv_usg_pct",
        "adv_ts_pct",
    ]
    corr_df = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = corr_df.corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
    )
    plt.title("Fatigue vs Performance Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "01_correlation_heatmap.png", dpi=300)
    plt.close()


def save_back_to_back_boxplot(df: pd.DataFrame) -> None:
    plot_df = df[["is_back_to_back", "pts"]].copy()
    plot_df["is_back_to_back"] = pd.to_numeric(
        plot_df["is_back_to_back"], errors="coerce"
    ).fillna(0)
    plot_df["pts"] = pd.to_numeric(plot_df["pts"], errors="coerce")
    plot_df = plot_df.dropna(subset=["pts"])
    plot_df["b2b_label"] = plot_df["is_back_to_back"].map(
        {0: "Not B2B (0)", 1: "Back-to-Back (1)"}
    )

    plt.figure(figsize=(9, 6))
    sns.boxplot(
        data=plot_df,
        x="b2b_label",
        hue="b2b_label",
        y="pts",
        palette=["#4c72b0", "#dd8452"],
        dodge=False,
        legend=False,
    )
    plt.title('Back-to-Back "Schedule Loss" Test: Points Distribution')
    plt.xlabel("Back-to-Back Status")
    plt.ylabel("Points Scored")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "02_back_to_back_penalty_boxplot.png", dpi=300)
    plt.close()


def save_distance_vs_efficiency_scatter(df: pd.DataFrame) -> None:
    plot_df = df[["miles_traveled", "adv_ts_pct", "mp"]].copy()
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    plot_df = plot_df[plot_df["mp"] >= 15].dropna(subset=["miles_traveled", "adv_ts_pct"])

    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=plot_df,
        x="miles_traveled",
        y="adv_ts_pct",
        scatter_kws={"alpha": 0.25, "s": 20},
        line_kws={"color": "#d62728", "linewidth": 2},
    )
    plt.title("Distance Traveled vs True Shooting % (MP >= 15)")
    plt.xlabel("Miles Traveled")
    plt.ylabel("True Shooting Percentage (adv_ts_pct)")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "03_distance_vs_efficiency_scatter.png", dpi=300)
    plt.close()


def save_altitude_bar_chart(df: pd.DataFrame) -> None:
    plot_df = df[["is_home", "arena_name", "pts", "fg_pct"]].copy()
    plot_df["is_home"] = pd.to_numeric(plot_df["is_home"], errors="coerce").fillna(0)
    plot_df["pts"] = pd.to_numeric(plot_df["pts"], errors="coerce")
    plot_df["fg_pct"] = pd.to_numeric(plot_df["fg_pct"], errors="coerce")
    # Visitor team performances only.
    plot_df = plot_df[plot_df["is_home"] == 0].dropna(subset=["pts", "fg_pct", "arena_name"])

    plot_df["altitude_group"] = plot_df["arena_name"].apply(
        lambda arena: "High Altitude (Denver/Utah)"
        if arena in HIGH_ALTITUDE_ARENAS
        else "Standard Altitude"
    )

    agg = (
        plot_df.groupby("altitude_group", as_index=False)[["pts", "fg_pct"]]
        .mean()
        .melt(id_vars="altitude_group", var_name="metric", value_name="value")
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg, x="altitude_group", y="value", hue="metric", palette="Set2")
    plt.title("Visitor Team Performance in High Altitude vs Standard Arenas")
    plt.xlabel("Arena Type")
    plt.ylabel("Average Value")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "04_altitude_check_bar_chart.png", dpi=300)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df = load_master_dataframe(conn)

    print(f"Loaded master dataframe with {len(df):,} rows.")
    print("Generating plots...")

    save_correlation_heatmap(df)
    print("  Saved: 01_correlation_heatmap.png")

    save_back_to_back_boxplot(df)
    print("  Saved: 02_back_to_back_penalty_boxplot.png")

    save_distance_vs_efficiency_scatter(df)
    print("  Saved: 03_distance_vs_efficiency_scatter.png")

    save_altitude_bar_chart(df)
    print("  Saved: 04_altitude_check_bar_chart.png")

    print(f"Done. Visualizations are in: {VIZ_DIR}")


if __name__ == "__main__":
    main()
