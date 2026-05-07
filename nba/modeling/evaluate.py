# Small manual check that each table actually works as intended

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "datasets" / "game_db.db"

TABLES = ["Arenas", "Teams", "Players", "Games", "Performances"]


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            None,
            "display.max_colwidth",
            36,
            "display.float_format",
            lambda x: f"{x:.4f}" if abs(x) < 1e6 else f"{x:.2f}",
        ):
            for table in TABLES:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 5",
                    conn,
                )
                sep = "=" * 88
                print(f"\n{sep}\n{table} — 5 random rows ({len(df.columns)} columns)\n{sep}")
                print(df.to_string(index=False))
        print()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
