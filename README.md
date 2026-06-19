# FIFA World Cup 2026 Predictor

A self-updating match and bracket predictor for the 2026 FIFA World Cup. The pipeline ingests live results from the football-data.org API, trains an ensemble of Elo ratings, a Dixon-Coles Poisson model, and an XGBoost classifier, then runs a Monte Carlo bracket simulation to produce win-probability rankings for every team. A Streamlit dashboard visualises upcoming fixtures, team rankings, and tournament odds. The nightly GitHub Actions workflow refreshes all artifacts automatically, and Streamlit Community Cloud redeploys on every push.

---

## Setup

### Prerequisites

- Python 3.11
- macOS users: XGBoost requires the OpenMP runtime. Install it before creating the virtual environment or the first `import xgboost` will fail:

```bash
brew install libomp
```

### Create the virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API key

The pipeline fetches match data from [football-data.org](https://www.football-data.org). A free tier key is sufficient.

1. Register at https://www.football-data.org and copy your API key.
2. **Local use:** create a `.env` file in the repo root:

   ```
   FOOTBALL_DATA_KEY=your_key_here
   ```

3. **GitHub Actions:** add the key as a repository secret named `FOOTBALL_DATA_KEY` (Settings → Secrets and variables → Actions → New repository secret). The nightly workflow reads it from there automatically.

---

## Running locally

### Update artifacts (requires API key)

```bash
python scripts/update.py
```

This fetches the latest results, retrains the models, runs the Monte Carlo simulation, and writes output files to `artifacts/`.

### Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Automated nightly refresh

`.github/workflows/update.yml` runs every day at **06:00 UTC** (and can be triggered manually via the GitHub Actions UI). It:

1. Installs dependencies.
2. Runs `python scripts/update.py` using the `FOOTBALL_DATA_KEY` secret.
3. Commits refreshed files under `artifacts/` and `data/results.csv` back to the branch with the message `data: nightly artifact refresh [skip ci]`.
4. Pushes the commit, which triggers a Streamlit Community Cloud redeploy.

---

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repository.
3. Set the main file path to `app/streamlit_app.py`.
4. Add `FOOTBALL_DATA_KEY` as an app secret in the Streamlit Cloud settings.
5. The app redeploys automatically whenever the nightly workflow pushes new artifacts.

---

## Tests

```bash
pytest
```

---

## Design documentation

The design spec and implementation plan live under `docs/superpowers/`:

- `docs/superpowers/specs/2026-06-19-fifa-2026-predictor-design.md` — architecture and feature design
- `docs/superpowers/plans/2026-06-19-fifa-2026-predictor-mvp.md` — phased MVP build plan
