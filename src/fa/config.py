"""Project-wide settings. Change leagues/seasons here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "football.duckdb"
REPORTS_DIR = ROOT / "reports"

# Our league code -> (football-data.co.uk code, openfootball code)
LEAGUES = {
    "EPL": ("E0", "en.1"),
    "LALIGA": ("SP1", "es.1"),
    "SERIEA": ("I1", "it.1"),
    "BUNDESLIGA": ("D1", "de.1"),
    "LIGUE1": ("F1", "fr.1"),
}

# Seasons are named by their starting year: 2015 -> 2015-16
FIRST_SEASON = 2014
LAST_SEASON = 2025          # last *completed* season used for evaluation

# Time-based split (never shuffle football data!)
TRAIN_SEASONS = range(FIRST_SEASON, 2024)   # 2014-15 .. 2023-24
VALID_SEASON = 2024                         # 2024-25: tune hyperparameters
TEST_SEASON = 2025                          # 2025-26: touched once, at the end


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"
