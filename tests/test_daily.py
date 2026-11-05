"""Run with:  PYTHONPATH=src python -m pytest -q"""
import pandas as pd

from fa.daily import already_done, mark_done


def test_a_day_is_marked_done_once(tmp_path):
    log = tmp_path / "daily_done.csv"
    day = pd.Timestamp("2026-10-29 08:30", tz="Asia/Beirut")
    assert not already_done(day, log)
    mark_done(day, log)
    assert already_done(day, log)
    assert not already_done(day + pd.Timedelta(days=1), log)
    mark_done(day + pd.Timedelta(days=1), log)
    assert log.read_text().count("\n") == 3                    # header + two days
