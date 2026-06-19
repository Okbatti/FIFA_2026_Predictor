import numpy as np
import pandas as pd
from predictor.models.ml_model import GoalsML

FEATURES = ["elo_diff","home_form","away_form","rest_diff"]

def _feats(n=300, seed=1):
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 150, n)
    home_form = rng.normal(1.3, 0.5, n); away_form = rng.normal(1.3, 0.5, n)
    rest_diff = rng.integers(-3, 4, n)
    # home goals rise with elo_diff
    hg = rng.poisson(np.clip(1.3 + elo_diff/300, 0.1, None))
    ag = rng.poisson(np.clip(1.3 - elo_diff/300, 0.1, None))
    return pd.DataFrame({"elo_diff":elo_diff,"home_form":home_form,"away_form":away_form,
                         "rest_diff":rest_diff,"home_goals":hg,"away_goals":ag})

def test_predict_positive_rates():
    m = GoalsML(FEATURES).fit(_feats())
    lh, la = m.predict_lambdas({"elo_diff":200,"home_form":1.5,"away_form":1.0,"rest_diff":1})
    assert lh > 0 and la > 0

def test_learns_elo_signal():
    m = GoalsML(FEATURES).fit(_feats())
    strong = m.predict_lambdas({"elo_diff":300,"home_form":1.3,"away_form":1.3,"rest_diff":0})
    weak   = m.predict_lambdas({"elo_diff":-300,"home_form":1.3,"away_form":1.3,"rest_diff":0})
    assert strong[0] > weak[0]
