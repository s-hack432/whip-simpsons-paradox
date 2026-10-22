"""Step 2: assign a venue to every game (2024-2026) from the MLB Stats API schedule.

`home_team` in Statcast is not a reliable park identifier (neutral-site series, relocated
games, the Rays' 2025 season at Steinbrenner Field, the Athletics in Sacramento and Las Vegas),
so every game_pk is mapped to the venue the MLB schedule says it was played in.

Definitions
- A team's *home park* in a season is the venue where it played the most designated home games.
- A game is *neutral-site* when it was played anywhere other than the designated home team's
  home park that season (international series, Field of Dreams, Little League Classic,
  relocated games, the Athletics' Las Vegas games). Neutral-site games are excluded from
  park-factor estimation but stay in pitcher totals under their own venue.

Outputs
- data/processed/games.parquet          one row per completed regular-season game
- outputs/tables/venues.csv             venue x season summary (home team, games, neutral games)

Usage: python -m src.venues
"""
from __future__ import annotations

import pandas as pd

from . import mlb_api
from .common import get_logger
from .config import ANALYSIS_SEASON, DATA_PROCESSED, OUT_TABLES, PF_SEASONS
from .fetch import load_statcast

log = get_logger("venues", "step2_venues.log")

GAMES_PATH = DATA_PROCESSED / "games.parquet"


def build_games(seasons=PF_SEASONS) -> pd.DataFrame:
    frames = []
    for season in seasons:
        g = mlb_api.completed_games(season)
        # Home park = venue hosting the most designated home games for that team-season.
        counts = g.groupby(["home_id", "venue_id"]).size().rename("n").reset_index()
        home_park = (counts.sort_values(["home_id", "n"], ascending=[True, False])
                     .drop_duplicates("home_id")[["home_id", "venue_id"]]
                     .rename(columns={"venue_id": "home_park_id"}))
        g = g.merge(home_park, on="home_id", how="left")
        g["is_neutral"] = g["venue_id"] != g["home_park_id"]
        frames.append(g)
    games = pd.concat(frames, ignore_index=True)

    # One display name per venue id (names change, e.g. Minute Maid Park -> Daikin Park):
    # use the name from the most recent season the venue appears in.
    latest = (games.sort_values("official_date").drop_duplicates("venue_id", keep="last")
              .set_index("venue_id")["venue_name"])
    games["venue_name_season"] = games["venue_name"]
    games["venue_name"] = games["venue_id"].map(latest)
    return games


def venue_summary(games: pd.DataFrame) -> pd.DataFrame:
    """Venue x season table: who called it home, how many games, how many neutral-site."""
    rows = []
    for (season, vid), g in games.groupby(["season", "venue_id"]):
        home_teams = sorted(g.loc[~g["is_neutral"], "home_name"].unique())
        neutral_matchups = sorted((g.loc[g["is_neutral"], "away_name"] + " @ "
                                   + g.loc[g["is_neutral"], "home_name"]).unique())
        rows.append({
            "season": season,
            "venue_id": vid,
            "venue_name": g["venue_name"].iloc[0],
            "venue_name_that_season": g["venue_name_season"].iloc[0],
            "home_park_of": "; ".join(home_teams),
            "games": len(g),
            "home_park_games": int((~g["is_neutral"]).sum()),
            "neutral_site_games": int(g["is_neutral"].sum()),
            "neutral_matchups": "; ".join(neutral_matchups),
        })
    return pd.DataFrame(rows).sort_values(["season", "venue_name"]).reset_index(drop=True)


def main():
    games = build_games()
    games.to_parquet(GAMES_PATH, index=False)
    vs = venue_summary(games)
    vs.to_csv(OUT_TABLES / "venues.csv", index=False)

    log.info("=" * 70)
    log.info("Step 2 summary: venues assigned to %d completed regular-season games", len(games))
    for season, g in games.groupby("season"):
        log.info("%s: %d games at %d venues; %d neutral-site games at %d venues", season, len(g),
                 g["venue_id"].nunique(), int(g["is_neutral"].sum()),
                 g.loc[g["is_neutral"], "venue_id"].nunique())
        for _, r in vs[(vs["season"] == season) & (vs["neutral_site_games"] > 0)].iterrows():
            log.info("    neutral: %-28s %d game(s): %s", r["venue_name"], r["neutral_site_games"],
                     r["neutral_matchups"])

    # Home-park changes across seasons (each park is its own venue). Group by team id, since
    # team names change too (Oakland Athletics -> Athletics).
    latest_team = (games.sort_values("official_date").drop_duplicates("home_id", keep="last")
                   .set_index("home_id")["home_name"])
    hp = (games[~games["is_neutral"]].assign(team=lambda d: d["home_id"].map(latest_team))
          .groupby(["team", "season"])["venue_name"].first().unstack())
    changed = hp[hp.nunique(axis=1) > 1]
    if len(changed):
        log.info("Teams whose home park changed during %s-%s:\n%s", min(PF_SEASONS),
                 max(PF_SEASONS), changed.to_string())

    # Every Statcast regular-season game must have a venue.
    for season in PF_SEASONS:
        try:
            sc = load_statcast(season, columns=["game_pk", "game_type", "at_bat_number",
                                                "pitch_number"], logger=log)
        except FileNotFoundError:
            log.warning("%s Statcast data not on disk yet; skipping the join check", season)
            continue
        pks = set(sc["game_pk"].unique())
        mapped = set(games.loc[games["season"] == season, "game_pk"])
        unmapped = sorted(pks - mapped)
        log.info("%s: %d Statcast games, %d with a venue, %d unmapped %s", season, len(pks),
                 len(pks & mapped), len(unmapped), unmapped[:20])
        if unmapped:
            raise SystemExit(f"STOP: {len(unmapped)} Statcast games in {season} have no venue.")

    n26 = games[games["season"] == ANALYSIS_SEASON]
    log.info("%s venues used (game counts):\n%s", ANALYSIS_SEASON,
             n26.groupby(["venue_name", "is_neutral"]).size().rename("games")
             .sort_values(ascending=False).to_string())
    log.info("Saved %s and %s", GAMES_PATH, OUT_TABLES / "venues.csv")


if __name__ == "__main__":
    main()
