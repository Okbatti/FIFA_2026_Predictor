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
    s = make_sampler(lambda h,a: (1.5, 1.0), rng)
    hg, ag = s("A","B")
    assert hg >= 0 and ag >= 0
