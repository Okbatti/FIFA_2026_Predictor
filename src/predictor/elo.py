from __future__ import annotations
from predictor import config

class EloModel:
    def __init__(self, k: float = config.ELO_K, home_adv: float = config.ELO_HOME_ADV,
                 default: float = 1500.0):
        self.k = k
        self.home_adv = home_adv
        self.default = default
        self._ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self._ratings.get(team, self.default)

    def set_rating(self, team: str, value: float) -> None:
        self._ratings[team] = value

    @staticmethod
    def expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _margin_mult(self, goal_diff: int) -> float:
        # FIFA-style margin-of-victory multiplier
        return (abs(goal_diff) + 1) ** 0.5 if goal_diff != 0 else 1.0

    def update(self, home: str, away: str, home_goals: int, away_goals: int,
               neutral: bool = False) -> None:
        adv = 0.0 if neutral else self.home_adv
        r_home, r_away = self.rating(home) + adv, self.rating(away)
        exp_home = self.expected(r_home, r_away)
        if home_goals > away_goals:
            score_home = 1.0
        elif home_goals < away_goals:
            score_home = 0.0
        else:
            score_home = 0.5
        mult = self._margin_mult(home_goals - away_goals)
        delta = self.k * mult * (score_home - exp_home)
        self._ratings[home] = self.rating(home) + delta
        self._ratings[away] = self.rating(away) - delta

    def ratings(self) -> dict[str, float]:
        return dict(self._ratings)
