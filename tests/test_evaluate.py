import numpy as np
from predictor.evaluate import log_loss, brier, calibration_bins, pick_best_weight

def test_log_loss_perfect_is_zero():
    preds = [{"home":1.0,"draw":0.0,"away":0.0}]
    assert log_loss(preds, ["home"]) < 1e-6

def test_log_loss_penalizes_wrong():
    confident_wrong = log_loss([{"home":0.0,"draw":0.0,"away":1.0}], ["home"])
    hedged = log_loss([{"home":0.34,"draw":0.33,"away":0.33}], ["home"])
    assert confident_wrong > hedged

def test_brier_range():
    b = brier([{"home":0.5,"draw":0.3,"away":0.2}], ["home"])
    assert 0.0 <= b <= 2.0

def test_calibration_bins_shape():
    preds = [{"home":p,"draw":0.0,"away":1-p} for p in np.linspace(0,1,20)]
    actuals = ["home" if i % 2 == 0 else "away" for i in range(20)]
    bins = calibration_bins(preds, actuals, n_bins=5)
    assert len(bins) == 5

def test_pick_best_weight_returns_valid():
    # two candidate prediction sets keyed by weight; outcome favors w=1.0 set
    def predictor(w):
        if w >= 0.9:
            return [{"home":0.9,"draw":0.05,"away":0.05}]
        return [{"home":0.4,"draw":0.3,"away":0.3}]
    best = pick_best_weight(predictor, ["home"], grid=[0.3,0.5,0.7,1.0])
    assert best == 1.0
