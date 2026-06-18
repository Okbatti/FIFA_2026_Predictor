import json
import pandas as pd
from pathlib import Path
from predictor.ingest import parse_matches, merge_results

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
