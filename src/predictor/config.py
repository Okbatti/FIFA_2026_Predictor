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

# Historical international results (no-auth, maintained CSV) used to train the
# models so team strengths are identifiable before WC2026 accumulates games.
HIST_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
HIST_SINCE_YEAR = 2021  # training window for international history
HIST_CACHE = RAW / "international_results.csv"

# Model params
ELO_K = 30.0
ELO_HOME_ADV = 65.0          # rating points added to home side (0 for neutral)
DC_HISTORY_YEARS = 4
BLEND_WEIGHT_DEFAULT = 0.7    # w: weight on Dixon-Coles vs ML
MAX_GOALS_GRID = 10           # bivariate Poisson grid size
# Tournament Monte Carlo draws. 5000 gives title-odds resolution ~0.6% while
# keeping the nightly run ~8 min (the sim is a pure-Python loop over groups +
# knockouts; raising this scales runtime roughly linearly).
SIM_N = 5000

for d in (RAW, PROCESSED, ARTIFACTS):
    d.mkdir(parents=True, exist_ok=True)
