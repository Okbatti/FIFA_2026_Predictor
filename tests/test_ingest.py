import json
import pandas as pd
from pathlib import Path
from predictor.ingest import parse_matches, merge_results, parse_historical, COLUMNS

FIX = Path(__file__).parent / "fixtures" / "sample_matches.json"

def test_parse_matches_schema():
    raw = json.loads(FIX.read_text())
    df = parse_matches(raw)
    assert list(df.columns) == ["date","home","away","home_goals","away_goals","neutral","stage","status"]
    assert len(df) == 2

def test_parse_finished_vs_scheduled():
    df = parse_matches(json.loads(FIX.read_text()))
    finished = df[df.status == "FINISHED"].iloc[0]
    assert finished.home_goals == 2 and finished.away_goals == 1
    sched = df[df.status == "SCHEDULED"].iloc[0]
    assert pd.isna(sched.home_goals)

def test_merge_results_dedupes_on_keys():
    df = parse_matches(json.loads(FIX.read_text()))
    merged = merge_results(df, df)  # merging with itself must not duplicate
    assert len(merged) == len(df)

def test_parse_historical_schema_and_name_mapping():
    raw = pd.DataFrame([
        {"date": "2024-03-01", "home_team": "Brazil", "away_team": "DR Congo",
         "home_score": 2, "away_score": 1, "neutral": True},
        {"date": "2024-03-05", "home_team": "Czech Republic", "away_team": "Cape Verde",
         "home_score": 0, "away_score": 0, "neutral": False},
    ])
    out = parse_historical(raw)
    assert list(out.columns) == COLUMNS
    assert (out.status == "FINISHED").all()
    assert (out.stage == "HISTORICAL").all()
    # historical-dataset names reconciled to football-data.org spellings
    names = set(out.home) | set(out.away)
    assert {"Congo DR", "Czechia", "Cape Verde Islands"} <= names
    assert "DR Congo" not in names and "Czech Republic" not in names

def test_parse_historical_drops_unplayed_rows():
    raw = pd.DataFrame([
        {"date": "2024-03-01", "home_team": "A", "away_team": "B",
         "home_score": 1, "away_score": 0, "neutral": True},
        {"date": "2099-01-01", "home_team": "A", "away_team": "B",
         "home_score": None, "away_score": None, "neutral": True},
    ])
    out = parse_historical(raw)
    assert len(out) == 1
