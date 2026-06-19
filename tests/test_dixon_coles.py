import numpy as np
import pandas as pd
from predictor.models.dixon_coles import DixonColes, tau

def test_tau_only_affects_low_scores():
    assert tau(3, 2, 1.0, 1.0, rho=-0.1) == 1.0   # high scores untouched
    assert tau(0, 0, 1.0, 1.0, rho=-0.1) != 1.0

def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["A","B","C","D"]
    strength = {"A":1.6,"B":1.2,"C":0.9,"D":0.6}  # attack scale
    rows=[]
    for _ in range(n):
        h,a = rng.choice(teams,2,replace=False)
        lh = strength[h]/strength[a]; la = strength[a]/strength[h]
        rows.append({"date":pd.Timestamp("2025-01-01"),"home":h,"away":a,
                     "home_goals":rng.poisson(lh),"away_goals":rng.poisson(la),
                     "neutral":True,"stage":"X","status":"FINISHED"})
    return pd.DataFrame(rows)

def test_fit_recovers_ordering():
    dc = DixonColes().fit(_synthetic())
    s = dc.team_strength()  # attack - defense composite, higher = better
    assert s["A"] > s["D"]

def test_predict_returns_positive_rates():
    dc = DixonColes().fit(_synthetic())
    lh, la = dc.predict_lambdas("A","D")
    assert lh > 0 and la > 0
    assert lh > la  # stronger home team scores more on average
