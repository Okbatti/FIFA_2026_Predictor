"""Full WC2026 tournament Monte Carlo: simulate remaining group games to decide
standings and qualification (top two per group + eight best third-placed teams),
seed the round of 32, then simulate the knockouts to estimate each team's
probability of reaching every stage and winning the title.

The 48-team knockout begins at the round of 32 (R32).
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np
import pandas as pd

from predictor.simulate import knockout_winner

STAGE_NAMES = ["R32", "R16", "QF", "SF", "FINAL", "WIN"]

# Round-of-32 bracket template. Each entry is one R32 match: (slotA, slotB).
# A slot is ("W", group) group winner, ("R", group) runner-up, or ("T", rank)
# the rank-th best (0 = best) of the eight qualifying third-placed teams.
# Counts are exact for the 48-team format: 12 winners, 12 runners-up, 8 thirds.
# No match pairs a group's winner against its own runner-up.
#
# NOTE: this is a plausible, self-consistent skeleton, NOT guaranteed to match
# FIFA's official slotting of the eight third-placed teams (that uses a fixed
# combination table). Title and deep-stage odds are robust to the exact third-
# place assignment; if you need precise per-stage opponents, replace this list
# with the official bracket.
BRACKET_TEMPLATE = [
    (("W", "A"), ("T", 0)),
    (("W", "B"), ("T", 1)),
    (("W", "C"), ("T", 2)),
    (("W", "D"), ("T", 3)),
    (("W", "E"), ("T", 4)),
    (("W", "F"), ("T", 5)),
    (("W", "G"), ("T", 6)),
    (("W", "H"), ("T", 7)),
    (("W", "I"), ("R", "B")),
    (("W", "J"), ("R", "A")),
    (("W", "K"), ("R", "D")),
    (("W", "L"), ("R", "C")),
    (("R", "E"), ("R", "F")),
    (("R", "G"), ("R", "H")),
    (("R", "I"), ("R", "J")),
    (("R", "K"), ("R", "L")),
]


def _group_letter(group_value: str) -> str:
    """'GROUP_A' -> 'A'."""
    return str(group_value).split("_")[-1]


def standings(rows: list[tuple]) -> list[tuple]:
    """Rank a group from played results.

    `rows` is a list of (home, away, home_goals, away_goals). Returns team names
    ordered best-first by (points, goal difference, goals for). Caller breaks
    residual ties (e.g. with a random jitter) before this if needed.
    """
    pts: dict[str, int] = defaultdict(int)
    gf: dict[str, int] = defaultdict(int)
    ga: dict[str, int] = defaultdict(int)
    teams: set[str] = set()
    for home, away, hg, ag in rows:
        teams.add(home)
        teams.add(away)
        gf[home] += hg
        ga[home] += ag
        gf[away] += ag
        ga[away] += hg
        if hg > ag:
            pts[home] += 3
        elif ag > hg:
            pts[away] += 3
        else:
            pts[home] += 1
            pts[away] += 1
    return sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)


def _team_record(rows: list[tuple], team: str) -> tuple[int, int, int]:
    """(points, goal_difference, goals_for) for one team across `rows`."""
    pts = gf = ga = 0
    for home, away, hg, ag in rows:
        if team == home:
            gf += hg; ga += ag
            pts += 3 if hg > ag else (1 if hg == ag else 0)
        elif team == away:
            gf += ag; ga += hg
            pts += 3 if ag > hg else (1 if ag == hg else 0)
    return pts, gf - ga, gf


def _make_sampler(predict_lambdas, rng):
    def sampler(home, away):
        lh, la = predict_lambdas(home, away)
        return int(rng.poisson(lh)), int(rng.poisson(la))
    return sampler


def simulate_tournament(matches: pd.DataFrame, predict_lambdas, n: int = 20000,
                        seed: int = 0, template=BRACKET_TEMPLATE) -> dict[str, dict]:
    """Monte Carlo the remainder of the tournament from current results.

    Returns {team: {R32, R16, QF, SF, FINAL: prob, win: prob}} for every team in
    the group stage. Returns {} if the data lacks a complete 12-group structure
    (e.g. synthetic test data), so callers can treat cup odds as unavailable.
    """
    if "group" not in matches.columns:
        return {}
    gmatches = matches[matches["group"].notna()].copy()
    if gmatches.empty:
        return {}
    gmatches["letter"] = gmatches["group"].map(_group_letter)
    letters = sorted(gmatches["letter"].unique())
    if len(letters) < 12:
        return {}

    # Precompute per-group fixture lists: (home, away, hg, ag, played)
    groups: dict[str, list[tuple]] = {}
    all_teams: set[str] = set()
    for letter, sub in gmatches.groupby("letter"):
        fixtures = []
        for r in sub.itertuples():
            played = (r.status == "FINISHED") and pd.notna(r.home_goals) and pd.notna(r.away_goals)
            hg = int(r.home_goals) if played else 0
            ag = int(r.away_goals) if played else 0
            fixtures.append((r.home, r.away, hg, ag, played))
            all_teams.update([r.home, r.away])
        groups[letter] = fixtures

    rng = np.random.default_rng(seed)
    sampler = _make_sampler(predict_lambdas, rng)
    order_idx = {s: i for i, s in enumerate(STAGE_NAMES)}
    counts: dict[str, dict[str, int]] = {t: defaultdict(int) for t in all_teams}

    for _ in range(n):
        # 1) groups -> placements with records (rebuild rows to rank thirds)
        placements: dict[str, list[str]] = {}
        records: dict[str, tuple] = {}
        for letter, fixtures in groups.items():
            rows = []
            for home, away, hg, ag, played in fixtures:
                if played:
                    rows.append((home, away, hg, ag))
                else:
                    sh, sa = sampler(home, away)
                    rows.append((home, away, sh, sa))
            order = sorted(
                standings(rows),
                key=lambda t: (*_team_record(rows, t), rng.random()),
                reverse=True,
            )
            placements[letter] = order
            for t in order:
                records[t] = _team_record(rows, t)

        # 2) qualification: top two per group + eight best thirds
        winners = {lt: placements[lt][0] for lt in placements}
        runners = {lt: placements[lt][1] for lt in placements}
        thirds = [(placements[lt][2], lt) for lt in placements if len(placements[lt]) >= 3]
        thirds.sort(key=lambda tl: (*records[tl[0]], rng.random()), reverse=True)
        best_thirds = [t for t, _ in thirds[:8]]

        # 3) seed the round of 32 from the template
        def resolve(slot):
            kind, key = slot
            if kind == "W":
                return winners[key]
            if kind == "R":
                return runners[key]
            return best_thirds[key]

        current = [(resolve(a), resolve(b)) for a, b in template]

        # 4) knockouts: R32 -> R16 -> QF -> SF -> FINAL -> champion
        reached: dict[str, str] = {}
        for stage in ["R32", "R16", "QF", "SF", "FINAL"]:
            nxt = []
            for home, away in current:
                reached[home] = stage
                reached[away] = stage
                w = knockout_winner(home, away, sampler, rng_seed=int(rng.integers(1_000_000_000)))
                nxt.append(w)
            if stage == "FINAL":
                reached[nxt[0]] = "WIN"
            else:
                current = [(nxt[i], nxt[i + 1]) for i in range(0, len(nxt), 2)]

        # 5) credit each team for its furthest stage and all earlier ones
        for team, stage in reached.items():
            for s in STAGE_NAMES[: order_idx[stage] + 1]:
                counts[team][s] += 1

    probs: dict[str, dict] = {}
    for team in all_teams:
        c = counts[team]
        probs[team] = {s: c.get(s, 0) / n for s in STAGE_NAMES if s != "WIN"}
        probs[team]["win"] = c.get("WIN", 0) / n
    return probs
