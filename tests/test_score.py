"""Run with:  PYTHONPATH=src python -m pytest -q"""
import duckdb
import numpy as np
import pandas as pd

import fa.predict as fp
from fa.predict import kickoff_utc_from_local
from fa.score import attach_results, build_report, eligible_forecasts, summarize


def _fc(created, kickoff, match="EPL_20261031_A_B", p=(0.5, 0.3, 0.2), date="2026-10-31",
        model="blend_v1", home="A", away="B"):
    return {"created_at": pd.Timestamp(created), "kickoff_utc": pd.Timestamp(kickoff) if kickoff else pd.NaT,
            "model": model, "match_id": match, "league": "EPL", "date": pd.Timestamp(date),
            "home_team": home, "away_team": away, "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "top_score": "2-1", "p_over25": 0.6, "p_btts": 0.55,
            "market_home": np.nan, "market_draw": np.nan, "market_away": np.nan}


def test_latest_forecast_before_kickoff_wins():
    fc = pd.DataFrame([
        _fc("2026-10-27 10:00", "2026-10-31 15:00", p=(0.4, 0.3, 0.3)),
        _fc("2026-10-30 10:00", "2026-10-31 15:00", p=(0.6, 0.2, 0.2)),   # latest before kickoff
        _fc("2026-10-31 16:00", "2026-10-31 15:00", p=(0.9, 0.05, 0.05)), # after kickoff: ignored
    ])
    ok = eligible_forecasts(fc)
    assert len(ok) == 1 and ok.p_home.iloc[0] == 0.6


def test_old_forecasts_without_kickoff_need_two_hours_before_match_day():
    fc = pd.DataFrame([
        _fc("2026-10-30 21:00", None, match="m1"),                        # 3 h before the day: ok
        _fc("2026-10-30 23:00", None, match="m2"),                        # 1 h before: too late
    ])
    assert list(eligible_forecasts(fc).match_id) == ["m1"]


def test_results_join_tolerates_a_one_day_date_difference():
    fc = pd.DataFrame([_fc("2026-10-27", "2026-10-31 23:30", date="2026-11-01")])
    res = pd.DataFrame({"league": ["EPL", "EPL"], "home_team": ["A", "A"], "away_team": ["B", "B"],
                        "date": pd.to_datetime(["2026-10-31", "2027-03-01"]),
                        "home_goals": [2, 0], "away_goals": [1, 0], "result": ["H", "D"]})
    m = attach_results(fc, res)
    assert len(m) == 1 and m.result.iloc[0] == "H"          # not the March rematch


def test_summary_and_report():
    rows = [_fc("2026-10-27", "2026-10-31 15:00", match=f"m{i}", home=f"H{i}", away=f"A{i}")
            for i in range(4)]
    df = pd.DataFrame(rows).assign(home_goals=[2, 0, 1, 3], away_goals=[1, 0, 2, 0],
                                   result=["H", "D", "A", "H"])
    s = summarize(df)
    assert s["n"] == 4 and s["accuracy"] == 0.5 and s["exact_score"] == 0.25
    assert s["over25_acc"] == 0.75                                     # 3, 0, 3, 3 goals vs "over"
    text = build_report(df, pending=3, generated=pd.Timestamp("2026-11-02"))
    assert "Scored matches: **4**" in text and "Biggest surprises" in text


def test_kickoff_conversion_handles_summer_and_winter_time():
    d = pd.Series(pd.to_datetime(["2026-05-14", "2026-12-12"]))
    t = pd.Series(["15:00", "15:00"])
    k = kickoff_utc_from_local(d, t, "Europe/London")
    assert list(k) == [pd.Timestamp("2026-05-14 14:00"), pd.Timestamp("2026-12-12 15:00")]


def test_old_forecasts_table_is_migrated_without_losing_rows(tmp_path, monkeypatch):
    db = tmp_path / "f.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE forecasts (created_at TIMESTAMP NOT NULL, model VARCHAR NOT NULL,
        match_id VARCHAR NOT NULL, league VARCHAR, date DATE, home_team VARCHAR, away_team VARCHAR,
        p_home DOUBLE, p_draw DOUBLE, p_away DOUBLE, xg_home DOUBLE, xg_away DOUBLE,
        top_score VARCHAR, top_score_p DOUBLE, market_home DOUBLE, market_draw DOUBLE,
        market_away DOUBLE)""")
    con.execute("INSERT INTO forecasts (created_at, model, match_id) VALUES ('2026-10-27', 'blend_v1', 'old')")
    con.close()
    monkeypatch.setattr(fp, "DB_PATH", db)
    new = pd.DataFrame([_fc("2026-10-28", "2026-10-31 15:00", match="new")]).drop(
        columns=["created_at", "model"]).assign(xg_home=1.5, xg_away=1.0, top_score_p=0.1)
    fp.save(new, pd.Timestamp("2026-10-28").to_pydatetime())
    rows = duckdb.connect(str(db)).execute(
        "SELECT match_id, kickoff_utc FROM forecasts ORDER BY match_id").df()
    assert list(rows.match_id) == ["new", "old"]
    assert rows.kickoff_utc.iloc[0] == pd.Timestamp("2026-10-31 15:00")
