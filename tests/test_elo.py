import math
from predictor.elo import EloModel

def test_expected_score_symmetry():
    elo = EloModel()
    e_a = elo.expected(1500, 1500)
    assert math.isclose(e_a, 0.5, abs_tol=1e-9)

def test_higher_rating_higher_expected():
    elo = EloModel()
    assert elo.expected(1700, 1500) > 0.5

def test_win_raises_rating():
    elo = EloModel(k=30.0, home_adv=0.0)
    elo.set_rating("A", 1500)
    elo.set_rating("B", 1500)
    elo.update("A", "B", home_goals=2, away_goals=0, neutral=True)
    assert elo.rating("A") > 1500
    assert elo.rating("B") < 1500

def test_margin_increases_swing():
    elo1 = EloModel(k=30.0, home_adv=0.0); elo1.set_rating("A",1500); elo1.set_rating("B",1500)
    elo2 = EloModel(k=30.0, home_adv=0.0); elo2.set_rating("A",1500); elo2.set_rating("B",1500)
    elo1.update("A","B",1,0,neutral=True)
    elo2.update("A","B",5,0,neutral=True)
    assert elo2.rating("A") > elo1.rating("A")

def test_unknown_team_gets_default():
    elo = EloModel(default=1500)
    assert elo.rating("Nowhere") == 1500
