from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def tau(hg: int, ag: int, lh: float, la: float, rho: float) -> float:
    if hg == 0 and ag == 0:
        return 1 - lh * la * rho
    if hg == 0 and ag == 1:
        return 1 + lh * rho
    if hg == 1 and ag == 0:
        return 1 + la * rho
    if hg == 1 and ag == 1:
        return 1 - rho
    return 1.0

class DixonColes:
    def __init__(self, half_life_days: float = 365.0):
        self.half_life_days = half_life_days
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def _weights(self, dates: pd.Series) -> np.ndarray:
        age = (dates.max() - dates).dt.days.to_numpy()
        return np.exp(-np.log(2) * age / self.half_life_days)

    def fit(self, matches: pd.DataFrame) -> "DixonColes":
        df = matches[matches.status == "FINISHED"].dropna(subset=["home_goals","away_goals"]).copy()
        self.teams = sorted(set(df.home) | set(df.away))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        w = self._weights(df.date)
        hg = df.home_goals.to_numpy(int); ag = df.away_goals.to_numpy(int)
        hi = df.home.map(idx).to_numpy(); ai = df.away.map(idx).to_numpy()
        neutral = df.neutral.to_numpy(bool)

        # params: attack[n], defense[n], home_adv, rho  (attack[0] fixed via sum-to-zero)
        def unpack(p):
            atk = np.concatenate([[ -p[:n-1].sum() ], p[:n-1]])  # sum-to-zero identifiability
            dfn = p[n-1:2*n-1]
            return atk, np.concatenate([[0.0], dfn]), p[-2], p[-1]

        def negll(p):
            atk, dfn, hadv, rho = unpack(p)
            log_lh = atk[hi] - dfn[ai] + np.where(neutral, 0.0, hadv)
            log_la = atk[ai] - dfn[hi]
            lh = np.exp(log_lh); la = np.exp(log_la)
            t = np.array([tau(h,a,l1,l2,rho) for h,a,l1,l2 in zip(hg,ag,lh,la)])
            t = np.clip(t, 1e-6, None)
            ll = np.log(t) + hg*log_lh - lh + ag*log_la - la
            return -(w * ll).sum()

        x0 = np.zeros(2*n - 1 + 2); x0[-2] = 0.25; x0[-1] = -0.05
        res = minimize(negll, x0, method="L-BFGS-B")
        atk, dfn, hadv, rho = unpack(res.x)
        self.attack = dict(zip(self.teams, atk))
        self.defense = dict(zip(self.teams, dfn))
        self.home_adv = float(hadv); self.rho = float(rho)
        return self

    def predict_lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        ah, dh = self.attack.get(home, 0.0), self.defense.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defense.get(away, 0.0)
        adv = 0.0 if neutral else self.home_adv
        lh = np.exp(ah - da + adv)
        la = np.exp(aa - dh)
        return float(lh), float(la)

    def team_strength(self) -> dict[str, float]:
        return {t: self.attack.get(t,0.0) - self.defense.get(t,0.0) for t in self.teams}
