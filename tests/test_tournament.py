import pandas as pd

from predictor.tournament import standings, simulate_tournament, STAGE_NAMES


def test_standings_orders_by_points_then_gd():
    rows = [
        ("A", "B", 3, 0),  # A win
        ("A", "C", 1, 1),  # A draw
        ("B", "C", 2, 0),  # B win
    ]
    order = standings(rows)
    # A: 4 pts, B: 3 pts, C: 1 pt
    assert order[0] == "A"
    assert order.index("B") < order.index("C")


def _wc_like(n_groups=12):
    """Build a finished 12-group stage + empty knockout fixtures, all neutral.
    Group X teams named X1..X4; X1 strongest by construction (beats everyone)."""
    rows = []
    date = pd.Timestamp("2026-06-11", tz="UTC")
    for gi in range(n_groups):
        letter = "ABCDEFGHIJKL"[gi]
        teams = [f"{letter}{i}" for i in range(1, 5)]
        # round robin, lower index wins
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for hi, ai in pairs:
            hg, ag = (2, 0) if hi < ai else (0, 2)
            rows.append({
                "date": date, "home": teams[hi], "away": teams[ai],
                "home_goals": hg, "away_goals": ag, "neutral": True,
                "stage": "GROUP_STAGE", "status": "FINISHED", "group": f"GROUP_{letter}",
            })
    # 32 undetermined knockout fixtures
    for stage, count in [("LAST_32", 16), ("LAST_16", 8), ("QUARTER_FINALS", 4),
                         ("SEMI_FINALS", 2), ("FINAL", 1)]:
        for _ in range(count):
            rows.append({
                "date": date, "home": None, "away": None,
                "home_goals": None, "away_goals": None, "neutral": True,
                "stage": stage, "status": "TIMED", "group": None,
            })
    return pd.DataFrame(rows)


def _equal_lambdas(home, away):
    return 1.3, 1.3


def test_returns_empty_without_group_structure():
    df = pd.DataFrame({"group": [None, None]})
    assert simulate_tournament(df, _equal_lambdas, n=10) == {}


def test_probabilities_are_valid_and_sum_to_one_title():
    df = _wc_like()
    probs = simulate_tournament(df, _equal_lambdas, n=300, seed=1)
    # all 48 teams present
    assert len(probs) == 48
    # exactly one champion across all teams
    total_titles = sum(p["win"] for p in probs.values())
    assert abs(total_titles - 1.0) < 1e-9
    # every team has all stage keys, each a valid probability
    for p in probs.values():
        for s in STAGE_NAMES[:-1]:
            assert 0.0 <= p[s] <= 1.0
        assert 0.0 <= p["win"] <= 1.0
        # monotonic: reaching a later stage implies reaching earlier ones
        assert p["R32"] >= p["R16"] >= p["QF"] >= p["SF"] >= p["FINAL"] >= p["win"]


def test_stronger_teams_qualify_more():
    df = _wc_like()
    probs = simulate_tournament(df, _equal_lambdas, n=300, seed=2)
    # group winners (X1) finished top; they should reach R32 far more than X4
    assert probs["A1"]["R32"] > probs["A4"]["R32"]
