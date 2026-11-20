"""Central place for paths and constants, so no module hard-codes a file location."""

from pathlib import Path

# .../ds_amrutha_parola  (this file lives in src/sentiment_quant/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "csv_files"
TRADES_CSV = DATA_DIR / "historical_data.csv"
SENTIMENT_CSV = DATA_DIR / "fear_greed_index.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_JSON = REPORTS_DIR / "results.json"

# Ordered from most fearful to most greedy; used for every table and plot.
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

# Crypto trades every calendar day, so annualise daily stats with 365, not 252.
PERIODS_PER_YEAR = 365

# Fixed seed so bootstrap / permutation results are reproducible run to run.
RANDOM_SEED = 42
