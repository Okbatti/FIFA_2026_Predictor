# FIFA World Cup 2026 Predictor — MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-updating data-science pipeline that predicts remaining WC2026 match outcomes and the tournament bracket, served via a Streamlit dashboard (tabs: Next Games, Cup Odds, Model Report).

**Architecture:** A pure-Python library (`src/predictor`) holds all logic — Elo, Dixon-Coles, XGBoost, an ensemble that emits per-match goal rates, a Monte Carlo bracket simulator, and an evaluator. An orchestration script (`scripts/update.py`) runs the pipeline and writes artifacts; a GitHub Actions nightly cron runs it and commits artifacts; Streamlit renders the artifacts. Logic is testable in isolation; the dashboard is a thin reader.

**Tech Stack:** Python 3.11, pandas, numpy, scipy, xgboost, requests, beautifulsoup4/lxml, streamlit, plotly, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-06-19-fifa-2026-predictor-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pinned dependencies |
| `pyproject.toml` | Package + ruff/pytest config |
| `src/predictor/__init__.py` | Package marker |
| `src/predictor/config.py` | Paths, params (blend weight default, sim N, cron time), env loading |
| `src/predictor/elo.py` | Elo rating engine; update after each result |
| `src/predictor/ingest.py` | football-data.org API client + FBref scrape + results.csv I/O |
| `src/predictor/features.py` | Build per-match feature table from matches + Elo |
| `src/predictor/models/dixon_coles.py` | Dixon-Coles goal-rate model |
| `src/predictor/models/ml_model.py` | XGBoost two-regressor goal model |
| `src/predictor/models/ensemble.py` | Blend λ's; bivariate Poisson grid → W/D/L + scorelines |
| `src/predictor/simulate.py` | Monte Carlo bracket → stage/title probabilities |
| `src/predictor/evaluate.py` | Backtest, log-loss/Brier, calibration, baselines, rankings |
| `scripts/update.py` | Orchestrate pipeline → write `artifacts/` |
| `app/streamlit_app.py` | 3-tab dashboard reading artifacts |
| `.github/workflows/update.yml` | Nightly cron runs update.py, commits artifacts |
| `tests/...` | Unit tests mirroring `src/predictor` |
| `tests/fixtures/...` | Sample API JSON + FBref HTML for offline tests |

---

## Task 0: Project scaffold

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `src/predictor/__init__.py`, `src/predictor/models/__init__.py`, `src/predictor/config.py`, `tests/__init__.py`, `tests/test_config.py`
- Create dirs (with `.gitkeep`): `data/raw/`, `data/processed/`, `artifacts/`, `tests/fixtures/`

- [ ] **Step 1: Create a virtualenv and dependency file**

Create `requirements.txt`:

```
pandas>=2.2
numpy>=1.26
scipy>=1.13
xgboost>=2.0
scikit-learn>=1.4
requests>=2.31
beautifulsoup4>=4.12
lxml>=5.2
python-dotenv>=1.0
streamlit>=1.36
plotly>=5.22
pyarrow>=16.0
pytest>=8.2
ruff>=0.5
```

Run:
```bash
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```
If Python 3.11 is unavailable, use the newest 3.12/3.13 that has xgboost wheels; avoid 3.14 (no xgboost wheel yet).

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "fifa-predictor"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Write `src/predictor/config.py`**

```python
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS_CSV = DATA / "results.csv"
ARTIFACTS = ROOT / "artifacts"

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
WC2026_COMPETITION = "WC"  # football-data.org competition code

# Model params
ELO_K = 30.0
ELO_HOME_ADV = 65.0          # rating points added to home side (0 for neutral)
DC_HISTORY_YEARS = 4
BLEND_WEIGHT_DEFAULT = 0.7    # w: weight on Dixon-Coles vs ML
MAX_GOALS_GRID = 10           # bivariate Poisson grid size
SIM_N = 20000

for d in (RAW, PROCESSED, ARTIFACTS):
    d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Write failing test `tests/test_config.py`**

```python
from predictor import config

def test_paths_exist():
    assert config.ROOT.exists()
    assert config.ARTIFACTS.exists()

def test_defaults_sane():
    assert 0.0 <= config.BLEND_WEIGHT_DEFAULT <= 1.0
    assert config.SIM_N >= 10000
    assert config.MAX_GOALS_GRID >= 8
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Add `.gitkeep` to empty dirs and commit**

```bash
touch data/raw/.gitkeep data/processed/.gitkeep artifacts/.gitkeep tests/fixtures/.gitkeep
git add requirements.txt pyproject.toml src tests data artifacts
git commit -m "chore: project scaffold and config"
```

---

## Task 1: Elo engine

**Files:**
- Create: `src/predictor/elo.py`, `tests/test_elo.py`

- [ ] **Step 1: Write failing test `tests/test_elo.py`**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_elo.py -v`
Expected: FAIL (`ModuleNotFoundError: predictor.elo`).

- [ ] **Step 3: Implement `src/predictor/elo.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_elo.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/predictor/elo.py tests/test_elo.py
git commit -m "feat: Elo rating engine with margin-of-victory multiplier"
```

---

## Task 2: Data ingest (API + scrape + results.csv)

**Files:**
- Create: `src/predictor/ingest.py`, `tests/test_ingest.py`, `tests/fixtures/sample_matches.json`

The match table is the canonical interface for everything downstream. It is a DataFrame with columns: `date` (datetime), `home`, `away`, `home_goals` (int, NaN if unplayed), `away_goals`, `neutral` (bool), `stage` (str), `status` (`FINISHED`/`SCHEDULED`).

- [ ] **Step 1: Create fixture `tests/fixtures/sample_matches.json`**

A trimmed football-data.org `/matches` response:

```json
{
  "matches": [
    {"utcDate": "2026-06-12T16:00:00Z", "status": "FINISHED",
     "stage": "GROUP_STAGE",
     "homeTeam": {"name": "Mexico"}, "awayTeam": {"name": "Canada"},
     "score": {"fullTime": {"home": 2, "away": 1}}},
    {"utcDate": "2026-07-10T19:00:00Z", "status": "SCHEDULED",
     "stage": "SEMI_FINALS",
     "homeTeam": {"name": "Brazil"}, "awayTeam": {"name": "France"},
     "score": {"fullTime": {"home": null, "away": null}}}
  ]
}
```

- [ ] **Step 2: Write failing test `tests/test_ingest.py`**

```python
import json
import pandas as pd
from pathlib import Path
from predictor.ingest import parse_matches, merge_results

FIX = Path(__file__).parent / "fixtures" / "sample_matches.json"

def test_parse_matches_schema():
    raw = json.loads(FIX.read_text())
    df = parse_matches(raw)
    assert list(df.columns) == ["date","home","away","home_goals","away_goals","neutral","stage","status"]
    assert len(df) == 2

def test_parse_finished_vs_scheduled():
    df = parse_matches(json.loads(FIX.read_text()))
    finished = df[df.status == "FINISHED"].iloc[0]
    assert finished.home_goals == 2 and finished.away_goals == 1
    sched = df[df.status == "SCHEDULED"].iloc[0]
    assert pd.isna(sched.home_goals)

def test_merge_results_dedupes_on_keys():
    df = parse_matches(json.loads(FIX.read_text()))
    merged = merge_results(df, df)  # merging with itself must not duplicate
    assert len(merged) == len(df)
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement `src/predictor/ingest.py`**

```python
from __future__ import annotations
import pandas as pd
import requests
from predictor import config

COLUMNS = ["date","home","away","home_goals","away_goals","neutral","stage","status"]
NEUTRAL_STAGES = {"GROUP_STAGE"}  # WC2026 is at neutral venues; host games handled by host-flag later

def parse_matches(raw: dict) -> pd.DataFrame:
    rows = []
    for m in raw.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        rows.append({
            "date": pd.to_datetime(m["utcDate"]),
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "home_goals": ft.get("home"),
            "away_goals": ft.get("away"),
            "neutral": True,  # World Cup: treat all as neutral by default
            "stage": m.get("stage", "UNKNOWN"),
            "status": m["status"],
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    return df

def fetch_wc_matches() -> pd.DataFrame:
    """Live pull from football-data.org. Requires FOOTBALL_DATA_KEY."""
    url = f"{config.FOOTBALL_DATA_BASE}/competitions/{config.WC2026_COMPETITION}/matches"
    resp = requests.get(url, headers={"X-Auth-Token": config.FOOTBALL_DATA_KEY}, timeout=30)
    resp.raise_for_status()
    return parse_matches(resp.json())

def merge_results(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union on (date, home, away); new rows win (carry latest scores/status)."""
    key = ["date","home","away"]
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=key, keep="last").sort_values("date")
    return combined.reset_index(drop=True)

def load_results() -> pd.DataFrame:
    if config.RESULTS_CSV.exists():
        return pd.read_csv(config.RESULTS_CSV, parse_dates=["date"])
    return pd.DataFrame(columns=COLUMNS)

def save_results(df: pd.DataFrame) -> None:
    df.to_csv(config.RESULTS_CSV, index=False)
```

Note on FBref historical scrape: add `fetch_fbref_history()` here only when wiring real historical training data. For the MVP build/test path it is optional — see Task 4, which accepts any match-table DataFrame. Keep the scrape behind its own function so it never runs during tests.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/predictor/ingest.py tests/test_ingest.py tests/fixtures/sample_matches.json
git commit -m "feat: match ingest, parsing, and results merge"
```

---

## Task 3: Feature builder

**Files:**
- Create: `src/predictor/features.py`, `tests/test_features.py`

Builds a per-finished-match feature frame for the ML model and updates Elo chronologically. Output columns: `elo_diff`, `home_form`, `away_form`, `rest_diff`, plus targets `home_goals`, `away_goals`.

- [ ] **Step 1: Write failing test `tests/test_features.py`**

```python
import pandas as pd
from predictor.features import build_features
from predictor.elo import EloModel

def _matches():
    return pd.DataFrame([
        {"date": pd.Timestamp("2025-01-01"), "home":"A","away":"B",
         "home_goals":2,"away_goals":0,"neutral":True,"stage":"X","status":"FINISHED"},
        {"date": pd.Timestamp("2025-01-08"), "home":"B","away":"A",
         "home_goals":1,"away_goals":1,"neutral":True,"stage":"X","status":"FINISHED"},
        {"date": pd.Timestamp("2025-01-15"), "home":"A","away":"B",
         "home_goals":3,"away_goals":1,"neutral":True,"stage":"X","status":"FINISHED"},
    ])

def test_build_features_columns():
    feats, elo = build_features(_matches())
    for col in ["elo_diff","home_form","away_form","rest_diff","home_goals","away_goals"]:
        assert col in feats.columns
    assert isinstance(elo, EloModel)

def test_elo_diff_reflects_dominant_team():
    feats, elo = build_features(_matches())
    # After A wins twice and draws, A's rating should exceed B's
    assert elo.rating("A") > elo.rating("B")

def test_only_finished_rows_become_features():
    m = _matches()
    m.loc[len(m)] = {"date": pd.Timestamp("2025-02-01"), "home":"A","away":"B",
                     "home_goals":None,"away_goals":None,"neutral":True,"stage":"X","status":"SCHEDULED"}
    feats, _ = build_features(m)
    assert len(feats) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_features.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/predictor/features.py`**

```python
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
        # update state
        goals_for.setdefault(home, []).append(int(m["home_goals"]))
        goals_for.setdefault(away, []).append(int(m["away_goals"]))
        last_date[home] = m["date"]; last_date[away] = m["date"]
        elo.update(home, away, int(m["home_goals"]), int(m["away_goals"]), neutral=bool(m["neutral"]))

    return pd.DataFrame(rows), elo
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_features.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/predictor/features.py tests/test_features.py
git commit -m "feat: chronological feature builder with Elo + form + rest"
```

---

## Task 4: Dixon-Coles model

**Files:**
- Create: `src/predictor/models/dixon_coles.py`, `tests/test_dixon_coles.py`

Estimates per-team attack/defense + home advantage by maximizing the (recency-weighted) Dixon-Coles likelihood, with the low-score `tau` correction. Predicts `(lambda_home, lambda_away)` for any fixture.

- [ ] **Step 1: Write failing test `tests/test_dixon_coles.py`**

```python
import numpy as np
import pandas as pd
from predictor.models.dixon_coles import DixonColes, tau

def test_tau_only_affects_low_scores():
    assert tau(3, 2, 1.0, 1.0, rho=-0.1) == 1.0   # high scores untouched
    assert tau(0, 0, 1.0, 1.0, rho=-0.1) != 1.0

def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["A","B","C","D"]
    strength = {"A":1.6,"B":1.2,"C":0.9,"D":0.6}  # attack scale
    rows=[]
    for _ in range(n):
        h,a = rng.choice(teams,2,replace=False)
        lh = strength[h]/strength[a]; la = strength[a]/strength[h]
        rows.append({"date":pd.Timestamp("2025-01-01"),"home":h,"away":a,
                     "home_goals":rng.poisson(lh),"away_goals":rng.poisson(la),
                     "neutral":True,"stage":"X","status":"FINISHED"})
    return pd.DataFrame(rows)

def test_fit_recovers_ordering():
    dc = DixonColes().fit(_synthetic())
    s = dc.team_strength()  # attack - defense composite, higher = better
    assert s["A"] > s["D"]

def test_predict_returns_positive_rates():
    dc = DixonColes().fit(_synthetic())
    lh, la = dc.predict_lambdas("A","D")
    assert lh > 0 and la > 0
    assert lh > la  # stronger home team scores more on average
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_dixon_coles.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/predictor/models/dixon_coles.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dixon_coles.py -v`
Expected: PASS (3 tests). If `test_fit_recovers_ordering` is flaky, increase synthetic `n` to 800 — it must not be made to pass by weakening the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/predictor/models/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat: Dixon-Coles goal-rate model with recency weighting"
```

---

## Task 5: XGBoost goal model

**Files:**
- Create: `src/predictor/models/ml_model.py`, `tests/test_ml_model.py`

Two `XGBRegressor`s (home goals, away goals) trained on the Task 3 feature frame. Predicts `(lambda_home, lambda_away)` from a single-row feature dict.

- [ ] **Step 1: Write failing test `tests/test_ml_model.py`**

```python
import numpy as np
import pandas as pd
from predictor.models.ml_model import GoalsML

FEATURES = ["elo_diff","home_form","away_form","rest_diff"]

def _feats(n=300, seed=1):
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 150, n)
    home_form = rng.normal(1.3, 0.5, n); away_form = rng.normal(1.3, 0.5, n)
    rest_diff = rng.integers(-3, 4, n)
    # home goals rise with elo_diff
    hg = rng.poisson(np.clip(1.3 + elo_diff/300, 0.1, None))
    ag = rng.poisson(np.clip(1.3 - elo_diff/300, 0.1, None))
    return pd.DataFrame({"elo_diff":elo_diff,"home_form":home_form,"away_form":away_form,
                         "rest_diff":rest_diff,"home_goals":hg,"away_goals":ag})

def test_predict_positive_rates():
    m = GoalsML(FEATURES).fit(_feats())
    lh, la = m.predict_lambdas({"elo_diff":200,"home_form":1.5,"away_form":1.0,"rest_diff":1})
    assert lh > 0 and la > 0

def test_learns_elo_signal():
    m = GoalsML(FEATURES).fit(_feats())
    strong = m.predict_lambdas({"elo_diff":300,"home_form":1.3,"away_form":1.3,"rest_diff":0})
    weak   = m.predict_lambdas({"elo_diff":-300,"home_form":1.3,"away_form":1.3,"rest_diff":0})
    assert strong[0] > weak[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ml_model.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/predictor/models/ml_model.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

class GoalsML:
    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self._home = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  objective="count:poisson", subsample=0.9)
        self._away = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  objective="count:poisson", subsample=0.9)

    def fit(self, feats: pd.DataFrame) -> "GoalsML":
        X = feats[self.feature_cols]
        self._home.fit(X, feats["home_goals"])
        self._away.fit(X, feats["away_goals"])
        return self

    def predict_lambdas(self, feat_row: dict) -> tuple[float, float]:
        X = pd.DataFrame([{c: feat_row[c] for c in self.feature_cols}])
        lh = float(np.clip(self._home.predict(X)[0], 0.05, None))
        la = float(np.clip(self._away.predict(X)[0], 0.05, None))
        return lh, la
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ml_model.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/predictor/models/ml_model.py tests/test_ml_model.py
git commit -m "feat: XGBoost Poisson goal model"
```

---

## Task 6: Ensemble + bivariate Poisson outcome

**Files:**
- Create: `src/predictor/models/ensemble.py`, `tests/test_ensemble.py`

Blends two `(lambda_home, lambda_away)` estimates, builds a Dixon-Coles-corrected score-probability grid, and reduces it to W/D/L probabilities and top scorelines. This is the single interface used by both simulation and evaluation.

- [ ] **Step 1: Write failing test `tests/test_ensemble.py`**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ensemble.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/predictor/models/ensemble.py`**

```python
from __future__ import annotations
import numpy as np
from scipy.stats import poisson
from predictor.models.dixon_coles import tau
from predictor import config

def blend_lambdas(dc: tuple[float,float], ml: tuple[float,float], w: float) -> tuple[float,float]:
    return (w*dc[0] + (1-w)*ml[0], w*dc[1] + (1-w)*ml[1])

def score_grid(lh: float, la: float, rho: float = -0.05,
               max_goals: int = config.MAX_GOALS_GRID) -> np.ndarray:
    h = poisson.pmf(np.arange(max_goals+1), lh)
    a = poisson.pmf(np.arange(max_goals+1), la)
    grid = np.outer(h, a)
    # Dixon-Coles low-score correction on the 2x2 corner
    for i in (0,1):
        for j in (0,1):
            grid[i,j] *= tau(i,j,lh,la,rho)
    return grid / grid.sum()

def outcome_probs(grid: np.ndarray) -> dict[str,float]:
    home = float(np.tril(grid,-1).sum())
    away = float(np.triu(grid,1).sum())
    draw = float(np.trace(grid))
    return {"home":home,"draw":draw,"away":away}

def top_scorelines(grid: np.ndarray, k: int = 3) -> list[tuple[tuple[int,int],float]]:
    flat = [((i,j), float(grid[i,j])) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat[:k]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ensemble.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/predictor/models/ensemble.py tests/test_ensemble.py
git commit -m "feat: ensemble blend + bivariate Poisson outcome probabilities"
```

---

## Task 7: Monte Carlo bracket simulator

**Files:**
- Create: `src/predictor/simulate.py`, `tests/test_simulate.py`

Simulates remaining knockout fixtures N times to produce per-team stage-reach and title probabilities. A `match_sampler` callback returns goal draws for a fixture, so the simulator is decoupled from the model (and trivially testable with a deterministic sampler).

- [ ] **Step 1: Write failing test `tests/test_simulate.py`**

```python
from predictor.simulate import simulate_bracket, knockout_winner

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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_simulate.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/predictor/simulate.py`**

```python
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
```

Note: `simulate_bracket` returns, per team, the fraction of sims in which they reached each stage. Keys are `R16, QF, SF, FINAL, win`. The bracket dict's `rounds[0]` is the current round's fixtures; deeper rounds are filled by winners.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_simulate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Build the model-backed sampler factory in `simulate.py`**

Append:

```python
def make_sampler(predict_lambdas, rng: np.random.Generator):
    """predict_lambdas(home, away) -> (lh, la). Returns a sampler drawing Poisson goals."""
    def sampler(home: str, away: str):
        lh, la = predict_lambdas(home, away)
        return int(rng.poisson(lh)), int(rng.poisson(la))
    return sampler
```

- [ ] **Step 6: Add a test for the sampler factory**

Append to `tests/test_simulate.py`:

```python
import numpy as np
from predictor.simulate import make_sampler

def test_make_sampler_draws_goals():
    rng = np.random.default_rng(0)
    s = make_sampler(lambda h,a: (1.5, 1.0), rng)
    hg, ag = s("A","B")
    assert hg >= 0 and ag >= 0
```

Run: `pytest tests/test_simulate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add src/predictor/simulate.py tests/test_simulate.py
git commit -m "feat: Monte Carlo bracket simulator with model-backed sampler"
```

---

## Task 8: Evaluation (backtest, metrics, baselines)

**Files:**
- Create: `src/predictor/evaluate.py`, `tests/test_evaluate.py`

Computes multiclass log-loss and Brier on W/D/L predictions, plus calibration bins. Used to tune the blend weight `w`.

- [ ] **Step 1: Write failing test `tests/test_evaluate.py`**

```python
import numpy as np
from predictor.evaluate import log_loss, brier, calibration_bins, pick_best_weight

def test_log_loss_perfect_is_zero():
    preds = [{"home":1.0,"draw":0.0,"away":0.0}]
    assert log_loss(preds, ["home"]) < 1e-6

def test_log_loss_penalizes_wrong():
    confident_wrong = log_loss([{"home":0.0,"draw":0.0,"away":1.0}], ["home"])
    hedged = log_loss([{"home":0.34,"draw":0.33,"away":0.33}], ["home"])
    assert confident_wrong > hedged

def test_brier_range():
    b = brier([{"home":0.5,"draw":0.3,"away":0.2}], ["home"])
    assert 0.0 <= b <= 2.0

def test_calibration_bins_shape():
    preds = [{"home":p,"draw":0.0,"away":1-p} for p in np.linspace(0,1,20)]
    actuals = ["home" if i % 2 == 0 else "away" for i in range(20)]
    bins = calibration_bins(preds, actuals, n_bins=5)
    assert len(bins) == 5

def test_pick_best_weight_returns_valid():
    # two candidate prediction sets keyed by weight; outcome favors w=1.0 set
    def predictor(w):
        if w >= 0.9:
            return [{"home":0.9,"draw":0.05,"away":0.05}]
        return [{"home":0.4,"draw":0.3,"away":0.3}]
    best = pick_best_weight(predictor, ["home"], grid=[0.3,0.5,0.7,1.0])
    assert best == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/predictor/evaluate.py`**

```python
from __future__ import annotations
import numpy as np

CLASSES = ["home","draw","away"]

def log_loss(preds: list[dict], actuals: list[str], eps: float = 1e-12) -> float:
    tot = 0.0
    for p, y in zip(preds, actuals):
        tot += -np.log(min(max(p[y], eps), 1.0))
    return tot / len(preds)

def brier(preds: list[dict], actuals: list[str]) -> float:
    tot = 0.0
    for p, y in zip(preds, actuals):
        for c in CLASSES:
            tot += (p[c] - (1.0 if c == y else 0.0)) ** 2
    return tot / len(preds)

def calibration_bins(preds: list[dict], actuals: list[str], n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b+1]
        pred_p, obs = [], []
        for p, y in zip(preds, actuals):
            ph = p["home"]
            if (lo <= ph < hi) or (b == n_bins-1 and ph == hi):
                pred_p.append(ph); obs.append(1.0 if y == "home" else 0.0)
        out.append({"lo":lo,"hi":hi,"mean_pred":float(np.mean(pred_p)) if pred_p else None,
                    "obs_freq":float(np.mean(obs)) if obs else None,"n":len(pred_p)})
    return out

def pick_best_weight(predict_fn, actuals: list[str], grid: list[float]) -> float:
    """predict_fn(w) -> list[outcome-prob dicts] aligned with actuals. Minimize log-loss."""
    best_w, best_score = grid[0], float("inf")
    for w in grid:
        score = log_loss(predict_fn(w), actuals)
        if score < best_score:
            best_score, best_w = score, w
    return best_w
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/predictor/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation metrics, calibration, blend-weight tuning"
```

---

## Task 9: Orchestration script `scripts/update.py`

**Files:**
- Create: `scripts/update.py`, `tests/test_update_smoke.py`

Wires the pipeline end-to-end and writes artifacts: `artifacts/next_games.parquet`, `artifacts/cup_odds.parquet`, `artifacts/metrics.json`, `artifacts/rankings.parquet`, `artifacts/meta.json` (timestamp, blend weight). Must run offline against a provided match table (for tests) or live via the API (in cron).

- [ ] **Step 1: Write failing smoke test `tests/test_update_smoke.py`**

```python
import pandas as pd
from predictor import config
from scripts.update import run_pipeline

def _matches():
    rows=[]
    teams=["A","B","C","D"]
    import numpy as np; rng=np.random.default_rng(0)
    base=pd.Timestamp("2025-01-01")
    for i in range(200):
        h,a=rng.choice(teams,2,replace=False)
        rows.append({"date":base+pd.Timedelta(days=i),"home":h,"away":a,
                     "home_goals":int(rng.poisson(1.4)),"away_goals":int(rng.poisson(1.1)),
                     "neutral":True,"stage":"GROUP_STAGE","status":"FINISHED"})
    # one upcoming knockout fixture
    rows.append({"date":base+pd.Timedelta(days=300),"home":"A","away":"B",
                 "home_goals":None,"away_goals":None,"neutral":True,
                 "stage":"SEMI_FINALS","status":"SCHEDULED"})
    return pd.DataFrame(rows)

def test_pipeline_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(config, "SIM_N", 200)
    artifacts = run_pipeline(_matches(), bracket={"rounds":[[("A","B")],[]]})
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "cup_odds.parquet").exists()
    assert (tmp_path / "next_games.parquet").exists()
    assert (tmp_path / "meta.json").exists()
    assert "log_loss" in artifacts["metrics"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_update_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: scripts.update`).

- [ ] **Step 3: Make `scripts` importable**

Create empty `scripts/__init__.py`. Confirm `pyproject.toml` `pythonpath` includes repo root by adding `"."`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
```

- [ ] **Step 4: Implement `scripts/update.py`**

```python
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from predictor import config
from predictor.features import build_features
from predictor.models.dixon_coles import DixonColes
from predictor.models.ml_model import GoalsML
from predictor.models.ensemble import blend_lambdas, score_grid, outcome_probs, top_scorelines
from predictor.simulate import simulate_bracket, make_sampler
from predictor import evaluate

FEATURES = ["elo_diff","home_form","away_form","rest_diff"]

def _outcome(actual_home, actual_away):
    if actual_home > actual_away: return "home"
    if actual_home < actual_away: return "away"
    return "draw"

def run_pipeline(matches: pd.DataFrame, bracket: dict | None = None) -> dict:
    finished = matches[matches.status == "FINISHED"].dropna(subset=["home_goals","away_goals"])
    feats, elo = build_features(matches)

    dc = DixonColes().fit(finished)
    ml = GoalsML(FEATURES).fit(feats)

    def predict_blended(home, away, w, neutral=True):
        dc_l = dc.predict_lambdas(home, away, neutral=neutral)
        adv = 0.0 if neutral else elo.home_adv
        feat_row = {"elo_diff": (elo.rating(home)+adv)-elo.rating(away),
                    "home_form": 1.3, "away_form": 1.3, "rest_diff": 0}
        ml_l = ml.predict_lambdas(feat_row)
        return blend_lambdas(dc_l, ml_l, w)

    # tune blend weight on finished games (predict each from model, compare to actual)
    actuals = [_outcome(r.home_goals, r.away_goals) for r in finished.itertuples()]
    def preds_for_w(w):
        out=[]
        for r in finished.itertuples():
            lh,la = predict_blended(r.home, r.away, w)
            out.append(outcome_probs(score_grid(lh, la, rho=dc.rho)))
        return out
    best_w = evaluate.pick_best_weight(preds_for_w, actuals, grid=[0.3,0.5,0.7,0.9,1.0])

    metrics = {
        "log_loss": evaluate.log_loss(preds_for_w(best_w), actuals),
        "brier": evaluate.brier(preds_for_w(best_w), actuals),
        "blend_weight": best_w,
        "n_train": len(finished),
    }
    calib = evaluate.calibration_bins(preds_for_w(best_w), actuals)

    # next games (scheduled)
    upcoming = matches[matches.status == "SCHEDULED"]
    next_rows=[]
    for r in upcoming.itertuples():
        lh,la = predict_blended(r.home, r.away, best_w)
        g = score_grid(lh, la, rho=dc.rho)
        p = outcome_probs(g); tops = top_scorelines(g, 3)
        next_rows.append({"date": r.date, "home": r.home, "away": r.away,
                          "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
                          "top_scores": ";".join(f"{i}-{j}:{v:.2f}" for (i,j),v in tops)})
    next_df = pd.DataFrame(next_rows)

    # cup odds via simulation
    cup_df = pd.DataFrame()
    if bracket is not None:
        rng = np.random.default_rng(0)
        sampler = make_sampler(lambda h,a: predict_blended(h,a,best_w), rng)
        probs = simulate_bracket(bracket, sampler, n=config.SIM_N)
        cup_df = pd.DataFrame([{"team":t, **v} for t,v in probs.items()]).sort_values("win", ascending=False)

    rankings = pd.DataFrame(
        [{"team":t,"strength":s,"elo":elo.rating(t)} for t,s in dc.team_strength().items()]
    ).sort_values("elo", ascending=False)

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    next_df.to_parquet(config.ARTIFACTS / "next_games.parquet")
    cup_df.to_parquet(config.ARTIFACTS / "cup_odds.parquet")
    rankings.to_parquet(config.ARTIFACTS / "rankings.parquet")
    (config.ARTIFACTS / "metrics.json").write_text(json.dumps({**metrics, "calibration":calib}, indent=2))
    (config.ARTIFACTS / "meta.json").write_text(json.dumps(
        {"updated_utc": pd.Timestamp.utcnow().isoformat(), "blend_weight": best_w}, indent=2))

    return {"metrics": metrics, "next_games": next_df, "cup_odds": cup_df}

def main():
    from predictor.ingest import fetch_wc_matches, load_results, merge_results, save_results
    live = fetch_wc_matches()
    merged = merge_results(load_results(), live)
    save_results(merged)
    # bracket construction from current knockout fixtures is added when WC2026 reaches knockouts;
    # until then run without simulation.
    run_pipeline(merged, bracket=None)

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run smoke test**

Run: `pytest tests/test_update_smoke.py -v`
Expected: PASS (1 test). Run full suite: `pytest -q` — all green.

- [ ] **Step 6: Commit**

```bash
git add scripts/update.py scripts/__init__.py tests/test_update_smoke.py pyproject.toml
git commit -m "feat: end-to-end update pipeline writing artifacts"
```

---

## Task 10: Streamlit dashboard

**Files:**
- Create: `app/streamlit_app.py`

Reads artifacts and renders three tabs. No model logic here. Manual verification only (Streamlit UI isn't unit-tested).

- [ ] **Step 1: Implement `app/streamlit_app.py`**

```python
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ART = Path(__file__).resolve().parents[1] / "artifacts"

st.set_page_config(page_title="WC2026 Predictor", layout="wide")
st.title("FIFA World Cup 2026 — Match & Bracket Predictor")

def _load_json(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else {}

def _load_parquet(name):
    p = ART / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

meta = _load_json("meta.json")
if meta:
    st.caption(f"Last updated: {meta.get('updated_utc','?')} · blend weight w={meta.get('blend_weight','?')}")

tab1, tab2, tab3 = st.tabs(["Next Games", "Cup Odds", "Model Report"])

with tab1:
    ng = _load_parquet("next_games.parquet")
    if ng.empty:
        st.info("No upcoming fixtures available yet.")
    else:
        for r in ng.itertuples():
            st.subheader(f"{r.home} vs {r.away}")
            cols = st.columns(3)
            cols[0].metric(r.home, f"{r.p_home:.0%}")
            cols[1].metric("Draw", f"{r.p_draw:.0%}")
            cols[2].metric(r.away, f"{r.p_away:.0%}")
            st.caption(f"Likely scores: {r.top_scores}")

with tab2:
    cup = _load_parquet("cup_odds.parquet")
    if cup.empty:
        st.info("Bracket simulation runs once the knockout stage begins.")
    else:
        st.dataframe(cup, use_container_width=True)
        fig = px.bar(cup.head(12), x="team", y="win", title="Title probability (top 12)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    metrics = _load_json("metrics.json")
    if metrics:
        c = st.columns(3)
        c[0].metric("Log-loss", f"{metrics.get('log_loss',0):.3f}")
        c[1].metric("Brier", f"{metrics.get('brier',0):.3f}")
        c[2].metric("Train games", metrics.get("n_train","?"))
        calib = pd.DataFrame(metrics.get("calibration", []))
        if not calib.empty:
            calib = calib.dropna(subset=["mean_pred","obs_freq"])
            fig = px.line(calib, x="mean_pred", y="obs_freq", markers=True,
                          title="Calibration (home-win)")
            fig.add_shape(type="line", x0=0,y0=0,x1=1,y1=1, line=dict(dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
    rankings = _load_parquet("rankings.parquet")
    if not rankings.empty:
        st.subheader("Team strength rankings")
        st.dataframe(rankings, use_container_width=True)
```

- [ ] **Step 2: Manual verification**

Generate artifacts with the smoke-test pipeline, then run the app:
```bash
python -c "from tests.test_update_smoke import _matches; from scripts.update import run_pipeline; run_pipeline(_matches(), bracket={'rounds':[[('A','B')],[]]})"
streamlit run app/streamlit_app.py
```
Expected: app opens; all three tabs render without error; Cup Odds shows a table + bar chart; Model Report shows metrics + calibration line.

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: Streamlit dashboard reading pipeline artifacts"
```

---

## Task 11: GitHub Actions nightly cron

**Files:**
- Create: `.github/workflows/update.yml`

- [ ] **Step 1: Implement `.github/workflows/update.yml`**

```yaml
name: nightly-update
on:
  schedule:
    - cron: "0 6 * * *"   # 06:00 UTC daily
  workflow_dispatch: {}

permissions:
  contents: write

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Run pipeline
        env:
          FOOTBALL_DATA_KEY: ${{ secrets.FOOTBALL_DATA_KEY }}
        run: python scripts/update.py
      - name: Commit artifacts
        run: |
          git config user.name "wc2026-bot"
          git config user.email "bot@users.noreply.github.com"
          git add artifacts data/results.csv
          git diff --cached --quiet || git commit -m "data: nightly artifact refresh [skip ci]"
          git push
```

- [ ] **Step 2: Document the required secret**

Add to `README.md` a "Setup" section noting: create a free key at football-data.org and add it as repo secret `FOOTBALL_DATA_KEY` (Settings → Secrets and variables → Actions). Also note Streamlit Community Cloud deploy: point it at `app/streamlit_app.py`.

- [ ] **Step 3: Validate workflow YAML locally**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/update.yml'))"`
Expected: no error.

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/update.yml README.md
git commit -m "ci: nightly cron to refresh artifacts"
git push
```

- [ ] **Step 5: Trigger once manually to verify**

In GitHub → Actions → nightly-update → Run workflow. Confirm green run and an artifacts commit (will no-op if WC API returns no changes). If it fails on the API key, the secret is missing.

---

## Self-Review

**Spec coverage:**
- §3 Data → Task 2 (API + results.csv; FBref scrape stubbed behind a function per spec note). ✔
- §4 Model (Elo/DC/ML/ensemble) → Tasks 1,4,5,6. ✔
- §5 Simulation → Task 7. ✔
- §6 Evaluation → Task 8. ✔
- §7 Architecture/layout → Task 0 + file structure table. ✔
- §8 Update/cron → Tasks 9,11. ✔
- §9 Dashboard (3 tabs) → Task 10. ✔
- §10 Stack → Task 0 requirements. ✔
- §11 Testing (Elo math, DC fit, Poisson sums to 1, sim sums to 1, ET/penalty, ingest parsing) → covered across tasks. ✔
- §12 Edge cases: unknown team default (Elo `default`, DC `.get(...,0.0)`); knockout no-draw (Task 7); missing API fallback — partially: cron no-ops on no change; full "fall back to last artifact" is implicit since artifacts persist in git. Acceptable for MVP.

**Placeholder scan:** No TBD/TODO; every code step has real code. FBref scrape intentionally deferred with explicit rationale (not a hidden placeholder — MVP trains on whatever match table is supplied; live path uses football-data.org).

**Type consistency:** `predict_lambdas(home, away[, neutral])` consistent across DC and the blended closure; `score_grid`/`outcome_probs`/`top_scorelines` signatures match between Task 6 and Task 9 usage; `simulate_bracket` return keys (`R16,QF,SF,FINAL,win`) consumed correctly (cup table sorts on `win`). `make_sampler(predict_lambdas, rng)` matches Task 9 call.

**Known caveat for the implementer:** Task 7's `_simulate_once` stage-naming assumes the bracket's first round maps to the correct stage label by counting back from the final. For the real WC2026 knockout (R32→R16→QF→SF→Final under the 48-team format), set `bracket["rounds"]` to the full remaining rounds and extend `STAGE_NAMES` to include `"R32"` if the round of 32 is still pending. Adjust `STAGE_NAMES` before wiring the live bracket.
