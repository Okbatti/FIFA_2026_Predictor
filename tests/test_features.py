import pandas as pd
from predictor.features import build_features
from predictor.elo import EloModel

def _matches():
    return pd.DataFrame([
        {"date": pd.Timestamp("2025-01-01"), "home":"A","away":"B",
         "home_goals":2,"away_goals":0,"neutral":True,"stage":"X","status":"FINISHED"},
        {"date": pd.Timestamp("2025-01-08"), "home":"B","away":"A",
         "home_goals":1,"away_goals":1,"neutral":True,"stage":"X","status":"FINISHED"},
        {"date": pd.Timestamp("2025-01-15"), "home":"A","away":"B",
         "home_goals":3,"away_goals":1,"neutral":True,"stage":"X","status":"FINISHED"},
    ])

def test_build_features_columns():
    feats, elo = build_features(_matches())
    for col in ["elo_diff","home_form","away_form","rest_diff","home_goals","away_goals"]:
        assert col in feats.columns
    assert isinstance(elo, EloModel)

def test_elo_diff_reflects_dominant_team():
    feats, elo = build_features(_matches())
    # After A wins twice and draws, A's rating should exceed B's
    assert elo.rating("A") > elo.rating("B")

def test_only_finished_rows_become_features():
    m = _matches()
    m.loc[len(m)] = {"date": pd.Timestamp("2025-02-01"), "home":"A","away":"B",
                     "home_goals":None,"away_goals":None,"neutral":True,"stage":"X","status":"SCHEDULED"}
    feats, _ = build_features(m)
    assert len(feats) == 3
