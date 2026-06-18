from __future__ import annotations
import numpy as np
from collections import defaultdict

STAGE_NAMES = ["R16","QF","SF","FINAL","WIN"]

def knockout_winner(home: str, away: str, sampler, rng_seed: int | None = None) -> str:
    hg, ag = sampler(home, away)
    if hg > ag:
        return home
    if ag > hg:
        return away
    # extra time: one more sample at reduced rate handled inside sampler if desired;
    # here regulation tie -> penalties (slight coin flip)
    rng = np.random.default_rng(rng_seed)
    return home if rng.random() < 0.5 else away

def _simulate_once(bracket: dict, sampler, rng) -> dict[str, str]:
    """Returns the furthest stage each participating team reached in one sim."""
    reached: dict[str, str] = {}
    rounds = [list(r) for r in bracket["rounds"]]
    n_rounds = len(rounds)
    # name the stage each round corresponds to, counting back from the final
    stage_for_round = STAGE_NAMES[-(n_rounds+1):-1]  # excludes WIN
    current = rounds[0]
    for ridx in range(n_rounds):
        stage = stage_for_round[ridx] if ridx < len(stage_for_round) else "FINAL"
        winners = []
        for home, away in current:
            for t in (home, away):
                reached.setdefault(t, stage)  # at least reached this stage's matches
                reached[t] = stage
            w = knockout_winner(home, away, sampler, rng_seed=int(rng.integers(1e9)))
            winners.append(w)
        # build next round by pairing winners
        if ridx + 1 < n_rounds:
            nxt = [(winners[i], winners[i+1]) for i in range(0, len(winners)-1, 2)]
            current = nxt
        else:
            champion = winners[0]
            reached[champion] = "WIN"
    return reached

def simulate_bracket(bracket: dict, sampler, n: int = 20000, seed: int = 0) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    order = {s: i for i, s in enumerate(STAGE_NAMES)}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _ in range(n):
        reached = _simulate_once(bracket, sampler, rng)
        for team, stage in reached.items():
            # credit team for reaching this stage and all earlier stages
            for s in STAGE_NAMES[: order[stage] + 1]:
                counts[team][s] += 1
    probs: dict[str, dict] = {}
    for team, c in counts.items():
        probs[team] = {s.lower() if s == "WIN" else s: c.get(s,0)/n for s in STAGE_NAMES}
        probs[team]["win"] = c.get("WIN",0)/n
    return probs


def make_sampler(predict_lambdas, rng: np.random.Generator):
    """predict_lambdas(home, away) -> (lh, la). Returns a sampler drawing Poisson goals."""
    def sampler(home: str, away: str):
        lh, la = predict_lambdas(home, away)
        return int(rng.poisson(lh)), int(rng.poisson(la))
    return sampler
