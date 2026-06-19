import numpy as np
from predictor.models.ensemble import blend_lambdas, score_grid, outcome_probs, top_scorelines

def test_blend_is_convex():
    assert blend_lambdas((2.0,1.0),(1.0,2.0),w=0.5) == (1.5,1.5)
    assert blend_lambdas((2.0,1.0),(1.0,2.0),w=1.0) == (2.0,1.0)

def test_score_grid_sums_to_one():
    g = score_grid(1.4, 1.1, rho=-0.05, max_goals=10)
    assert abs(g.sum() - 1.0) < 1e-6

def test_outcome_probs_sum_to_one_and_favor_stronger():
    p = outcome_probs(score_grid(2.0, 0.7, rho=-0.05))
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
    assert p["home"] > p["away"]

def test_top_scorelines_sorted():
    g = score_grid(1.5, 1.2, rho=-0.05)
    tops = top_scorelines(g, k=3)
    assert len(tops) == 3
    assert tops[0][1] >= tops[1][1] >= tops[2][1]

def test_extreme_lambdas_stay_valid():
    # Underdetermined fits can emit huge/tiny lambdas; the grid must still be a
    # valid probability distribution (sums to 1, all finite, in [0,1]).
    for lh, la in [(500.0, 0.0001), (0.0, 0.0), (1e6, 1e6)]:
        g = score_grid(lh, la, rho=-0.05)
        assert np.isfinite(g).all()
        assert (g >= 0).all()
        assert abs(g.sum() - 1.0) < 1e-9
        p = outcome_probs(g)
        assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
