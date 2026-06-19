from predictor.simulate import simulate_bracket, knockout_winner, make_sampler
import numpy as np

def test_knockout_no_draw():
    # deterministic sampler: home always wins 1-0
    sampler = lambda h, a: (1, 0)
    w = knockout_winner("A","B", sampler)
    assert w == "A"

def test_penalties_break_ties():
    sampler = lambda h, a: (1, 1)  # always draw in regulation+ET
    # tie-break uses rng; with fixed seed result is deterministic
    w = knockout_winner("A","B", sampler, rng_seed=42)
    assert w in ("A","B")

def test_simulate_probabilities_sum_per_stage():
    bracket = {
        "rounds": [
            [("A","B"), ("C","D")],   # semis
            [],                        # final (filled by winners)
        ]
    }
    sampler = lambda h, a: (2, 0)  # home always advances
    probs = simulate_bracket(bracket, sampler, n=500)
    # exactly one champion across teams
    total_titles = sum(v["win"] for v in probs.values())
    assert abs(total_titles - 1.0) < 1e-9
    assert probs["A"]["win"] == 1.0  # A is always home and always wins

def test_make_sampler_draws_goals():
    rng = np.random.default_rng(0)
    s = make_sampler(lambda h,a: (2.0, 0.5), rng)
    draws = [s("A","B") for _ in range(2000)]
    mean_h = sum(d[0] for d in draws)/len(draws)
    mean_a = sum(d[1] for d in draws)/len(draws)
    assert mean_h > mean_a            # reflects lambda_home > lambda_away
    assert 1.6 < mean_h < 2.4         # ~2.0


def test_partial_bracket_no_phantom_stage_credit():
    # 8 teams, 3 rounds: QF -> SF -> FINAL. Home always wins.
    bracket = {"rounds": [
        [("A","B"),("C","D"),("E","F"),("G","H")],  # QF
        [], [],                                       # SF, FINAL filled by winners
    ]}
    sampler = lambda h,a: (2,0)
    probs = simulate_bracket(bracket, sampler, n=300)
    # Only QF, SF, FINAL, win stages should exist — never R16
    assert "R16" not in probs["A"]
    assert set(probs["A"].keys()) == {"QF","SF","FINAL","win"}
    # Teams that only ever play as 'away' (B,D,F,H) are eliminated in QF
    assert probs["B"]["QF"] == 1.0
    assert probs["B"]["SF"] == 0.0
    assert probs["B"]["win"] == 0.0
    # A is always home, wins everything
    assert probs["A"]["win"] == 1.0
    # exactly one champion
    assert abs(sum(v["win"] for v in probs.values()) - 1.0) < 1e-9
