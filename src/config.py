"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_API = DATA_RAW / "mlb_api"          # cached MLB Stats API responses
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_LOGS = ROOT / "outputs" / "logs"

for _d in (DATA_RAW, DATA_API, DATA_PROCESSED, OUT_TABLES, OUT_FIGURES, OUT_LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# Pitcher-level analysis uses this season only.
ANALYSIS_SEASON = 2026
# Park factors may pool these seasons (only seasons where a venue was the team's home park).
PF_SEASONS = (2024, 2025, 2026)

# Season window stated in the project spec; fetch.py verifies it against the MLB Stats API
# and stops if they disagree.
EXPECTED_WINDOW = {2026: ("2026-03-25", "2026-09-27")}

# Columns kept from the raw Statcast pulls for the analysis.
STATCAST_COLS = [
    "game_pk", "game_date", "game_type", "game_year", "home_team", "away_team",
    "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "outs_when_up", "balls", "strikes", "pitcher", "batter", "events", "description", "des",
    "home_score", "away_score", "post_home_score", "post_away_score", "n_thruorder_pitcher",
]
