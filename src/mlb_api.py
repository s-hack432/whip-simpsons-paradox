"""Thin, cached wrappers around the public MLB Stats API (statsapi.mlb.com)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_API

BASE = "https://statsapi.mlb.com/api/v1"


def _get_json(path: str, params: dict, cache_file: Path, refresh: bool = False) -> dict:
    """GET BASE/path with retries; cache the JSON response on disk."""
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    last_exc = None
    for attempt in range(5):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            break
        except (requests.RequestException, ValueError) as exc:  # network error or bad JSON
            last_exc = exc
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"MLB Stats API request failed: {path} {params}") from last_exc
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(cache_file)
    return data


def season_info(season: int) -> dict:
    """Official season calendar (regular-season start/end dates, etc.)."""
    data = _get_json(f"seasons/{season}", {"sportId": 1}, DATA_API / f"season_{season}.json")
    return data["seasons"][0]


def regular_season_window(season: int) -> tuple[str, str]:
    info = season_info(season)
    return info["regularSeasonStartDate"], info["regularSeasonEndDate"]


def schedule_raw(season: int, refresh: bool = False) -> dict:
    """Full regular-season schedule, hydrated with the final linescore of each game."""
    params = {"sportId": 1, "season": season, "gameType": "R", "hydrate": "linescore"}
    return _get_json("schedule", params, DATA_API / f"schedule_{season}.json", refresh=refresh)


def schedule_entries(season: int) -> pd.DataFrame:
    """One row per schedule *entry*. Postponed/suspended games can appear more than once."""
    data = schedule_raw(season)
    rows = []
    for day in data["dates"]:
        for g in day["games"]:
            ls = g.get("linescore") or {}
            rows.append({
                "game_pk": g["gamePk"],
                "season": season,
                "official_date": g.get("officialDate"),
                "game_datetime_utc": g.get("gameDate"),
                "status": g["status"]["detailedState"],
                "coded_state": g["status"].get("codedGameState"),
                "abstract_state": g["status"].get("abstractGameState"),
                "venue_id": g["venue"]["id"],
                "venue_name": g["venue"]["name"],
                "home_id": g["teams"]["home"]["team"]["id"],
                "home_name": g["teams"]["home"]["team"]["name"],
                "away_id": g["teams"]["away"]["team"]["id"],
                "away_name": g["teams"]["away"]["team"]["name"],
                "home_score": g["teams"]["home"].get("score"),
                "away_score": g["teams"]["away"].get("score"),
                "double_header": g.get("doubleHeader"),
                "game_number": g.get("gameNumber"),
                "scheduled_innings": g.get("scheduledInnings"),
                "series_description": g.get("seriesDescription"),
                "resume_date": g.get("resumeDate"),
                "resumed_from": g.get("resumedFrom"),
                "rescheduled_from": g.get("rescheduledFrom"),
                "ls_inning": ls.get("currentInning"),
                "ls_is_top": ls.get("isTopInning"),
                "ls_outs": ls.get("outs"),
            })
    return pd.DataFrame(rows)


def completed_games(season: int) -> pd.DataFrame:
    """One row per completed regular-season game (coded state 'F': Final or Completed Early).

    If a game_pk has several 'F' entries (e.g. a suspended game listed on both dates),
    keep the latest entry, which carries the final linescore.
    """
    s = schedule_entries(season)
    done = s[s["coded_state"] == "F"].sort_values(["game_pk", "game_datetime_utc"])
    return done.drop_duplicates("game_pk", keep="last").reset_index(drop=True)


def linescore_halves(season: int) -> pd.DataFrame:
    """Official per-half-inning runs and hits for every completed game.

    half = 0 for the top of the inning (away team batting), 1 for the bottom. A half-inning
    that was not played (home team ahead after the top of the 9th) has no 'runs' entry and is
    omitted. `final_outs` is set on the game's last half-inning (from linescore.outs).
    """
    data = schedule_raw(season)
    done = completed_games(season)
    keep = set(zip(done["game_pk"], done["game_datetime_utc"]))
    rows = []
    for day in data["dates"]:
        for g in day["games"]:
            if (g["gamePk"], g.get("gameDate")) not in keep:
                continue
            ls = g.get("linescore") or {}
            for inn in ls.get("innings", []):
                for half, side in ((0, "away"), (1, "home")):
                    rec = inn.get(side) or {}
                    if "runs" not in rec:
                        continue
                    rows.append({"game_pk": g["gamePk"], "inning": inn["num"], "half": half,
                                 "runs": rec.get("runs"), "hits": rec.get("hits")})
    out = pd.DataFrame(rows).sort_values(["game_pk", "inning", "half"]).reset_index(drop=True)
    last = out.groupby("game_pk").tail(1).index
    outs_map = done.set_index("game_pk")["ls_outs"]
    out["final_outs"] = pd.NA
    out.loc[last, "final_outs"] = out.loc[last, "game_pk"].map(outs_map).values
    return out


def season_pitching_stats(season: int) -> pd.DataFrame:
    """Official season pitching lines for every pitcher (one row per player-team split)."""
    params = {
        "stats": "season", "group": "pitching", "season": season, "sportId": 1,
        "gameType": "R", "playerPool": "ALL", "limit": 5000,
    }
    data = _get_json("stats", params, DATA_API / f"pitching_stats_{season}.json")
    rows = []
    for split in data["stats"][0]["splits"]:
        st = split["stat"]
        rows.append({
            "pitcher": split["player"]["id"],
            "name": split["player"].get("fullName"),
            "team_id": (split.get("team") or {}).get("id"),
            "team": (split.get("team") or {}).get("name"),
            "num_teams": split.get("numTeams"),
            "g": st.get("gamesPlayed"),
            "gs": st.get("gamesStarted"),
            "outs": st.get("outs"),
            "ip": st.get("inningsPitched"),
            "h": st.get("hits"),
            "bb": st.get("baseOnBalls"),
            "ibb": st.get("intentionalWalks"),
            "hbp": st.get("hitByPitch"),
            "bf": st.get("battersFaced"),
            "whip": st.get("whip"),
        })
    return pd.DataFrame(rows)


def pitcher_game_log(pitcher: int, season: int) -> pd.DataFrame:
    """Official game-by-game pitching lines for one pitcher (used to explain discrepancies)."""
    params = {"stats": "gameLog", "group": "pitching", "season": season, "gameType": "R"}
    data = _get_json(f"people/{pitcher}/stats", params,
                     DATA_API / f"gamelog_{pitcher}_{season}.json")
    rows = []
    for split in (data.get("stats") or [{}])[0].get("splits", []):
        st = split["stat"]
        rows.append({"game_pk": split["game"]["gamePk"], "date": split.get("date"),
                     "outs": st.get("outs"), "h": st.get("hits"), "bb": st.get("baseOnBalls"),
                     "gs": st.get("gamesStarted")})
    return pd.DataFrame(rows, columns=["game_pk", "date", "outs", "h", "bb", "gs"])


def team_abbrevs(season: int) -> dict[int, str]:
    data = _get_json("teams", {"sportId": 1, "season": season}, DATA_API / f"teams_{season}.json")
    return {t["id"]: t.get("abbreviation") for t in data["teams"]}


def people_names(ids: list[int]) -> dict[int, str]:
    """Full names for MLBAM person ids (batched, cached per batch)."""
    names: dict[int, str] = {}
    ids = sorted({int(i) for i in ids})
    for k in range(0, len(ids), 150):
        batch = ids[k:k + 150]
        cache = DATA_API / f"people_{batch[0]}_{batch[-1]}_{len(batch)}.json"
        data = _get_json("people", {"personIds": ",".join(map(str, batch))}, cache)
        for p in data.get("people", []):
            names[p["id"]] = p.get("fullName")
    return names
