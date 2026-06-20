from __future__ import annotations
import pandas as pd
import requests
from predictor import config

COLUMNS = ["date","home","away","home_goals","away_goals","neutral","stage","status","group"]

def parse_matches(raw: dict) -> pd.DataFrame:
    rows = []
    for m in raw.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        home = m.get("homeTeam") or {}
        away = m.get("awayTeam") or {}
        rows.append({
            "date": pd.to_datetime(m["utcDate"]),
            "home": home.get("name"),
            "away": away.get("name"),
            "home_goals": ft.get("home"),
            "away_goals": ft.get("away"),
            "neutral": True,  # World Cup: treat all as neutral by default
            "stage": m.get("stage", "UNKNOWN"),
            "status": m["status"],
            "group": m.get("group"),  # e.g. GROUP_A; null for knockouts
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

# Reconcile historical-dataset team names to football-data.org spellings so the
# WC2026 fixtures map onto their historical strength estimates.
HIST_NAME_MAP = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
    "Czech Republic": "Czechia",
}

def parse_historical(df: pd.DataFrame) -> pd.DataFrame:
    """Map a raw martj42 international-results frame (date, home_team, away_team,
    home_score, away_score, neutral) to the canonical match schema."""
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], utc=True),
        "home": df["home_team"].replace(HIST_NAME_MAP),
        "away": df["away_team"].replace(HIST_NAME_MAP),
        "home_goals": pd.to_numeric(df["home_score"], errors="coerce"),
        "away_goals": pd.to_numeric(df["away_score"], errors="coerce"),
        "neutral": df["neutral"].astype(bool),
        "stage": "HISTORICAL",
        "status": "FINISHED",
    }, columns=COLUMNS)
    return out.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)

def fetch_historical_results(since_year: int = config.HIST_SINCE_YEAR) -> pd.DataFrame:
    """Load international results since `since_year`. Downloads the maintained CSV
    (no auth) and caches it; falls back to the cache if the network is unavailable."""
    try:
        raw = pd.read_csv(config.HIST_RESULTS_URL)
        config.HIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(config.HIST_CACHE, index=False)
    except Exception:
        if not config.HIST_CACHE.exists():
            raise
        raw = pd.read_csv(config.HIST_CACHE)
    parsed = parse_historical(raw)
    return parsed[parsed["date"].dt.year >= since_year].reset_index(drop=True)

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
