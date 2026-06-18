# FIFA World Cup 2026 Match Predictor — Design

**Date:** 2026-06-19
**Status:** Approved (design); pending spec review
**Repo:** https://github.com/Okbatti/FIFA_2026_Predictor

## 1. Goal

A data science project that predicts the outcomes of remaining FIFA World Cup 2026
matches and refreshes automatically after every game. It is both a learning/portfolio
piece and a live, hands-off dashboard.

The model predicts **goal rates** for each side, from which it derives Win/Draw/Loss
probabilities, likely scorelines, and — by Monte Carlo simulation of the remaining
bracket — each team's probability of reaching each stage and winning the cup.

## 2. Scope

### MVP (this spec)
- **A — Next Games:** W/D/L probabilities + top likely scorelines per upcoming fixture.
- **B — Cup Odds:** each team's probability to reach R16 / QF / SF / Final / Win, via
  Monte Carlo bracket simulation (N = 10k–50k).
- **D — Model Report:** backtest accuracy (log-loss, Brier) vs baselines, calibration
  plot, team strength rankings, last-updated timestamp.

### Deferred (later phases, out of scope here)
- **C — Visual bracket tree** (Phase 2): renders MVP sim output as a knockout tree.
- **E — Pick history** (Phase 3): logs past predictions vs actual, tracks hit rate.
  Logging hooks may be added early so data accumulates; display is later.

### Non-goals
- No betting integration / real-money features.
- No user accounts or multi-user state.
- No paid infrastructure — everything runs on free tiers.

## 3. Data

Hybrid source strategy:

- **Live results & fixtures:** [football-data.org](https://www.football-data.org)
  free tier (10 req/min) — WC2026 fixtures, results, JSON. Drives auto-update.
- **Historical training data:** scrape FBref / Wikipedia for past international
  matches (last ~4 years), team ratings, and xG where available. Richer features.
- **Elo seed:** pre-tournament ratings from eloratings.net / FIFA world ranking.

Storage:
- `data/results.csv` — append-only WC2026 results.
- `data/raw/` — raw API pulls + scrapes (largely gitignored).
- `data/processed/` — cleaned match table + team features.

Fallbacks: if API data is missing on a run, fall back to last committed artifact.
Teams absent from the historical set get an Elo prior + confederation-average features.

## 4. Model (the DS core)

An **ensemble** producing per-match goal rates `(λ_home, λ_away)`.

### 4.1 Elo engine (`elo.py`)
- Per-team rating, updated after **every** result — the backbone of "updates after
  each game."
- Margin-of-victory multiplier; home/neutral-venue adjustment.
- Seeded from pre-tournament ratings.
- Rating diff → expected goal supremacy; also exported as a **feature** for both models.

### 4.2 Dixon-Coles (`models/dixon_coles.py`)
- Per-team attack/defense strengths fit on recency-weighted historical internationals
  plus WC2026 games as they land.
- Outputs `λ_home`, `λ_away` (Poisson rates) with the low-score correlation
  correction term `τ`.
- Stable and interpretable; strong when WC2026 data is still thin (early knockouts).

### 4.3 XGBoost (`models/ml_model.py`)
- Two regressors → expected goals for each side.
- Features: Elo diff, rolling recent form (goals for/against), rest days,
  neutral-venue flag, confederation, FBref xG where available.
- Catches nonlinear form/rest effects.

### 4.4 Ensemble (`models/ensemble.py`)
- Blend the two rate estimates: `λ = w·λ_DC + (1−w)·λ_ML` for each side.
- Weight `w` tuned by backtest log-loss.
- Cold-start: `w` starts high (trust Dixon-Coles + Elo prior) when WC2026 games are
  few, shifts toward ML as it earns trust.
- Final `(λ_home, λ_away)` → bivariate Poisson score grid → P(scoreline) → W/D/L.

## 5. Simulation (`simulate.py`)

- Input: current bracket state (remaining fixtures, who is in) + ensemble model.
- Per game: draw a scoreline from the bivariate Poisson grid.
- Knockout ties: simulate extra-time (scaled λ) then penalty shootout (near coin-flip
  with a slight Elo tilt). Draws disallowed in knockouts.
- Advance the winner; repeat to the final. Run **N = 10k–50k** times.
- Aggregate → per-team stage-reach and title probabilities (dashboard tab B).
- Vectorized with numpy; re-runs on demand in the dashboard (<2s target).

## 6. Evaluation (`evaluate.py`)

- **Backtest:** walk-forward over past internationals + completed WC2026 games —
  predict before each, score after.
- **Metrics:** log-loss + Brier on W/D/L, against baselines:
  - de-vigged bookmaker odds (when available),
  - Elo-only,
  - bet-on-favorite.
- **Calibration:** reliability plot (predicted prob vs observed frequency).
- Used to tune blend weight `w` and validate before trusting output.
- Produces team strength ranking table (Dixon-Coles + Elo).

## 7. Architecture / Repo layout

```
fifa-predictor/
├── data/
│   ├── raw/                 # API pulls + FBref scrapes (mostly gitignored)
│   ├── processed/           # match table, team features
│   └── results.csv          # WC2026 results, append-only
├── src/predictor/
│   ├── ingest.py            # football-data API + FBref scrape
│   ├── features.py          # Elo, form, rest-days, attack/def features
│   ├── elo.py               # Elo engine
│   ├── models/
│   │   ├── dixon_coles.py
│   │   ├── ml_model.py
│   │   └── ensemble.py
│   ├── simulate.py          # Monte Carlo bracket → cup odds
│   ├── evaluate.py          # backtest, calibration, baselines
│   └── config.py            # paths, params, blend weights
├── app/streamlit_app.py     # tabs: Next Games | Cup Odds | Model Report
├── scripts/update.py        # ingest → features → train → sim → save
├── artifacts/               # model.pkl, cup_odds, next_games, metrics.json
├── tests/
├── requirements.txt
└── .github/workflows/update.yml
```

Logic lives in `src/predictor` (testable); Streamlit is a thin renderer.

## 8. Update mechanism (free, hands-off)

- **Nightly cron** via GitHub Actions (`update.yml`): scheduled (~06:00 UTC) + manual
  dispatch. Steps: checkout → install → `python scripts/update.py` → commit & push
  artifacts if changed. API key in GitHub Secrets.
- `update.py`: pull API results → append `results.csv` → rebuild features + Elo →
  fit Dixon-Coles + train XGBoost → backtest (refresh `w`) → run sim → save artifacts.
- **Streamlit Community Cloud** watches the repo and auto-redeploys on push.
- Dashboard loads cached artifacts (fast) and can re-run a light Monte Carlo on demand.

All free: GitHub Actions (free for the repo), Streamlit Community Cloud,
football-data.org free tier.

## 9. Dashboard (`app/streamlit_app.py`)

Three tabs:
- **Next Games:** upcoming fixtures with W/D/L % and top-3 likely scorelines.
- **Cup Odds:** sortable team × stage probability table + title-odds bar chart.
- **Model Report:** backtest metrics vs baselines, calibration plot, strength
  rankings, last-updated timestamp.

Artifacts cached; a "re-simulate" button triggers a fresh Monte Carlo run.

## 10. Tech stack

Python 3.11; pandas, numpy, scipy (DC optimization), xgboost, statsmodels (optional),
streamlit, plotly, requests, beautifulsoup4/lxml (scraping), pytest, ruff.
Dependencies pinned in `requirements.txt`. Local secrets in `.env`; CI secrets in
GitHub Secrets. Central params in `config.py`.

## 11. Testing

Unit tests on: Elo update math, Dixon-Coles fit on toy data, Poisson grid sums to 1,
simulation probabilities sum to 1, extra-time/penalty logic, ingest parsing with
fixture data.

## 12. Edge cases

- Missing/late API data → fall back to last committed artifact.
- Unknown team (not in history) → Elo prior + confederation-average features.
- Knockout draws disallowed → ET then penalties in sims.
- Thin early-tournament data → blend leans on Dixon-Coles + Elo prior.

## 13. Build order

1. **Phase 1 (MVP):** ingest + Elo + Dixon-Coles + XGBoost + ensemble + simulate +
   evaluate + Streamlit (tabs A, B, D) + cron. Ships a working live predictor.
2. **Phase 2:** visual bracket tree (tab C).
3. **Phase 3:** pick-history tracking and hit-rate display (tab E).
