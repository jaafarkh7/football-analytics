"""Run with:  PYTHONPATH=src python -m pytest -q"""
import duckdb
import pandas as pd
import pytest

import fa.records as rec


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "RECORDS", tmp_path / "records")
    monkeypatch.setattr(rec, "DB_PATH", tmp_path / "db.duckdb")
    return tmp_path / "db.duckdb"


def _fc(created, match, p=0.5):
    return {"created_at": pd.Timestamp(created), "model": "blend_v1", "match_id": match,
            "date": pd.Timestamp("2026-10-31"), "kickoff_utc": pd.Timestamp("2026-10-31 15:00"),
            "p_home": p, "p_draw": 0.3, "p_away": 0.7 - p}


def _write_db(db, rows):
    con = duckdb.connect(str(db))
    con.register("df", pd.DataFrame(rows))
    con.execute("CREATE OR REPLACE TABLE forecasts AS SELECT * FROM df")
    con.close()


def _read_db(db):
    con = duckdb.connect(str(db), read_only=True)
    df = con.execute("SELECT * FROM forecasts ORDER BY created_at").df()
    con.close()
    return df


def test_round_trip_keeps_rows_from_both_sides(isolated):
    _write_db(isolated, [_fc("2026-10-27 10:00", "laptop")])
    rec.sync_out()                                             # laptop -> CSV
    csv = pd.read_csv(rec.RECORDS / "forecasts.csv")
    csv = pd.concat([csv, pd.DataFrame([_fc("2026-10-28 06:00", "cloud")])])
    csv.to_csv(rec.RECORDS / "forecasts.csv", index=False)      # a cloud run added a row
    _write_db(isolated, [_fc("2026-10-27 10:00", "laptop"), _fc("2026-10-29 09:00", "laptop2")])
    rec.sync_in()
    assert list(_read_db(isolated).match_id) == ["laptop", "cloud", "laptop2"]


def test_duplicates_are_dropped_and_timestamps_survive(isolated):
    _write_db(isolated, [_fc("2026-10-27 10:00:00.123456", "m1")])
    rec.sync_out()
    rec.sync_in()
    rec.sync_in()
    df = _read_db(isolated)
    assert len(df) == 1
    assert df.created_at.iloc[0] == pd.Timestamp("2026-10-27 10:00:00.123456")
    assert df.kickoff_utc.iloc[0] == pd.Timestamp("2026-10-31 15:00")


def test_sync_in_without_csv_or_table_is_harmless(isolated):
    rec.sync_in()
    rec.sync_out()
    assert not (rec.RECORDS / "forecasts.csv").exists()


def test_sync_in_into_fresh_database(tmp_path, monkeypatch):
    """A new database (as on GitHub Actions) + saved records: duckdb 2.x must not complain
    about an unfinished query result."""
    import fa.records as rec
    monkeypatch.setattr(rec, "RECORDS", tmp_path / "records")
    (tmp_path / "records").mkdir()
    pd.DataFrame([dict(created_at="2026-10-27 06:30:00", model="blend_v1", match_id="m1",
                       league="EPL", date="2026-11-01", home_team="A", away_team="B",
                       p_home=0.5, p_draw=0.3, p_away=0.2)]).to_csv(
        tmp_path / "records" / "forecasts.csv", index=False)
    db = tmp_path / "fresh.duckdb"
    duckdb.connect(str(db)).execute("CREATE TABLE matches (x INT)").close()
    rec.sync_in(db)
    assert duckdb.connect(str(db)).execute("SELECT count(*) FROM forecasts").fetchall()[0][0] == 1
