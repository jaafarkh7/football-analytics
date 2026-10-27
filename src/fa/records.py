"""Keep the things that must survive between runs as CSV files in the repo.

Scheduled runs (GitHub Actions) start on a fresh machine every time, so anything that
lives only in data/football.duckdb would be lost. These tables are therefore mirrored in
records/*.csv, which the workflow commits back to the repo:

    records/forecasts.csv     every forecast, timestamped (the track record)
    records/season_sims.csv   every season simulation (for "biggest movers")

sync_in()  merges the CSV into the local database (both sides kept, duplicates dropped)
sync_out() writes the merged table back to the CSV

Merging both ways means nothing is lost when you run on your laptop and in the cloud.
Committing records/forecasts.csv before kickoff also gives every forecast a public,
timestamped git commit.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from fa.config import DB_PATH, ROOT

RECORDS = ROOT / "records"
TABLES = {
    "forecasts": {"key": ["created_at", "model", "match_id"],
                  "dates": ["created_at", "date", "kickoff_utc"]},
    "season_sims": {"key": ["created_at", "league", "team"],
                    "dates": ["created_at", "as_of"]},
}


def _table_exists(con, name: str) -> bool:
    return con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                       [name]).fetchall()[0][0] > 0


def _read_csv(name: str) -> pd.DataFrame | None:
    path = RECORDS / f"{name}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    for c in TABLES[name]["dates"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
    return df


def _merge(frames: list[pd.DataFrame], name: str) -> pd.DataFrame:
    df = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True)
    for c in TABLES[name]["dates"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
    key = [k for k in TABLES[name]["key"] if k in df]
    return df.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)


def sync_in(db_path=None) -> None:
    """Merge records/*.csv into the local database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path or DB_PATH))
    try:
        for name in TABLES:
            csv_df = _read_csv(name)
            if csv_df is None:
                continue
            db_df = con.execute(f"SELECT * FROM {name}").df() if _table_exists(con, name) else None
            merged = _merge([db_df, csv_df], name)
            con.register("merged_df", merged)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM merged_df")
            con.unregister("merged_df")
    finally:
        con.close()


def sync_out(db_path=None) -> None:
    """Write the database tables back to records/*.csv (merging with what's there)."""
    con = duckdb.connect(str(db_path or DB_PATH), read_only=True)
    try:
        for name in TABLES:
            if not _table_exists(con, name):
                continue
            merged = _merge([_read_csv(name), con.execute(f"SELECT * FROM {name}").df()], name)
            RECORDS.mkdir(parents=True, exist_ok=True)
            merged.to_csv(RECORDS / f"{name}.csv", index=False)
    finally:
        con.close()
