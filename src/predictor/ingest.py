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
