"""Required validation (Step 4): computed 2026 season totals vs official lines.

At least 95% of pitchers with 20+ IP must match within 1 out and within 1 H and 1 BB.
Needs the cached Statcast data and pitcher-game table (run src.fetch, src.venues,
src.attribution first); skipped otherwise.
"""
import pytest

from src.attribution import (VALIDATION_REQUIRED_SHARE, compare_to_official,
                             pitcher_games_path, validation_summary)
from src.config import ANALYSIS_SEASON


@pytest.fixture(scope="module")
def comparison():
    if not pitcher_games_path(ANALYSIS_SEASON).exists():
        pytest.skip("pitcher-game table not built yet (run python -m src.attribution)")
    return compare_to_official(ANALYSIS_SEASON)


def test_95pct_of_20ip_pitchers_match_within_tolerance(comparison):
    s = validation_summary(comparison)
    assert s["n_20ip"] > 300, "suspiciously few 20+ IP pitchers"
    assert s["share_within_tol_20ip"] >= VALIDATION_REQUIRED_SHARE, s


def test_every_pitcher_present_in_both_sources(comparison):
    assert (comparison["source"] == "both").all(), comparison.loc[
        comparison["source"] != "both", ["pitcher", "name", "source"]]


def test_league_totals_close(comparison):
    # League-wide, computed minus official should be within a handful of events.
    assert abs(comparison["d_outs"].sum()) <= 5
    assert abs(comparison["d_h"].sum()) <= 10
    assert abs(comparison["d_bb"].sum()) <= 5


def test_starts_and_appearances_match(comparison):
    # Starter/reliever classification relies on games started.
    share = ((comparison["d_g"] == 0) & (comparison["d_gs"] == 0)).mean()
    assert share >= 0.99
