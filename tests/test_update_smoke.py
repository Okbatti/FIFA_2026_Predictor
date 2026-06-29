import pandas as pd
from predictor import config
from scripts.update import run_pipeline

def _matches():
    rows=[]
    teams=["A","B","C","D"]
    import numpy as np; rng=np.random.default_rng(0)
    base=pd.Timestamp("2025-01-01")
    for i in range(200):
        h,a=rng.choice(teams,2,replace=False)
        rows.append({"date":base+pd.Timedelta(days=i),"home":h,"away":a,
                     "home_goals":int(rng.poisson(1.4)),"away_goals":int(rng.poisson(1.1)),
                     "neutral":True,"stage":"GROUP_STAGE","status":"FINISHED"})
    # one upcoming knockout fixture
    rows.append({"date":base+pd.Timedelta(days=300),"home":"A","away":"B",
                 "home_goals":None,"away_goals":None,"neutral":True,
                 "stage":"SEMI_FINALS","status":"SCHEDULED"})
    return pd.DataFrame(rows)

def test_pipeline_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(config, "SIM_N", 200)
    artifacts = run_pipeline(_matches())
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "cup_odds.parquet").exists()
    assert (tmp_path / "next_games.parquet").exists()
    assert (tmp_path / "prev_games.parquet").exists()
    assert (tmp_path / "meta.json").exists()
    assert "log_loss" in artifacts["metrics"]
    # finished games get a predicted scoreline next to the actual result
    prev = artifacts["prev_games"]
    assert len(prev) > 0
    for col in ["actual_home", "actual_away", "pred_home", "pred_away"]:
        assert col in prev.columns

def test_pipeline_handles_object_dtype_dates(tmp_path, monkeypatch):
    # Live data merged with an empty results.csv frame can leave date as object dtype;
    # run_pipeline must coerce it rather than crashing on the .dt accessor.
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(config, "SIM_N", 200)
    m = _matches()
    m["date"] = m["date"].astype(object)
    assert m["date"].dtype == object
    artifacts = run_pipeline(m)
    assert "log_loss" in artifacts["metrics"]
