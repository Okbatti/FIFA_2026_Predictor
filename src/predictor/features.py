from __future__ import annotations
import pandas as pd
from predictor.elo import EloModel

def _rolling_form(history: list[int], window: int = 5) -> float:
    if not history:
        return 0.0
    h = history[-window:]
    return sum(h) / len(h)

def build_features(matches: pd.DataFrame, seed_ratings: dict[str, float] | None = None):
    """Walk matches chronologically. For each FINISHED match emit pre-match features
    using only prior info, then update Elo. Returns (features_df, fitted_EloModel)."""
    elo = EloModel()
    if seed_ratings:
        for t, r in seed_ratings.items():
            elo.set_rating(t, r)

    goals_for: dict[str, list[int]] = {}
    last_date: dict[str, pd.Timestamp] = {}
    rows = []

    df = matches.sort_values("date")
    for _, m in df.iterrows():
        if m["status"] != "FINISHED" or pd.isna(m["home_goals"]):
            continue
        home, away = m["home"], m["away"]
        adv = 0.0 if m["neutral"] else elo.home_adv
        elo_diff = (elo.rating(home) + adv) - elo.rating(away)
        rest_home = (m["date"] - last_date.get(home, m["date"])).days
        rest_away = (m["date"] - last_date.get(away, m["date"])).days
        rows.append({
            "elo_diff": elo_diff,
            "home_form": _rolling_form(goals_for.get(home, [])),
            "away_form": _rolling_form(goals_for.get(away, [])),
            "rest_diff": rest_home - rest_away,
            "home_goals": int(m["home_goals"]),
            "away_goals": int(m["away_goals"]),
        })
        # update state AFTER capturing pre-match features (no leakage)
        goals_for.setdefault(home, []).append(int(m["home_goals"]))
        goals_for.setdefault(away, []).append(int(m["away_goals"]))
        last_date[home] = m["date"]
        last_date[away] = m["date"]
        elo.update(home, away, int(m["home_goals"]), int(m["away_goals"]), neutral=bool(m["neutral"]))

    return pd.DataFrame(rows), elo
