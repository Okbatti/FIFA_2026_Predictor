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
