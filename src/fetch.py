"""Step 1: fetch pitch-level Statcast data.

Pulls week-by-week chunks with pybaseball.statcast() and caches each chunk as parquet in
data/raw/. Chunks already on disk are skipped. If pybaseball fails for a chunk, that chunk is
pulled directly from Baseball Savant's CSV search endpoint (one request per day) and the log
says so.

After a season's chunks are on disk, its game coverage is checked against the MLB schedule.
Any completed game missing from Statcast is re-requested individually; if games are still
missing the script stops with a report instead of continuing with partial data.

Usage (from the project root):
    python -m src.fetch                    # 2026 first, then 2024 and 2025
    python -m src.fetch --seasons 2026
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import date, timedelta

import pandas as pd
import pyarrow.parquet as pq
import requests

from . import mlb_api
from .common import get_logger
from .config import ANALYSIS_SEASON, DATA_RAW, EXPECTED_WINDOW, PF_SEASONS, STATCAST_COLS

log = get_logger("fetch", "step1_fetch.log")

SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
# The same query pybaseball sends for each day (all game types; we filter to "R" later).
SAVANT_DAY_QUERY = (
    "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7CPO%7CS%7C=&hfSea="
    "&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA="
    "&game_date_gt={d}&game_date_lt={d}&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn="
    "&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed"
    "&sort_order=desc&min_abs=0&type=details&"
)
SAVANT_GAME_QUERY = "all=true&type=details&game_pk={pk}"


# --------------------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------------------
def week_chunks(start: date, end: date, days: int = 7):
    """Yield (chunk_start, chunk_end) date pairs covering [start, end] inclusive."""
    s = start
    while s <= end:
        e = min(s + timedelta(days=days - 1), end)
        yield s, e
        s = e + timedelta(days=1)


def chunk_path(season: int, s: date, e: date):
    return DATA_RAW / f"statcast_{season}_{s:%Y%m%d}_{e:%Y%m%d}.parquet"


def _savant_csv(query: str) -> pd.DataFrame:
    """One request to the Savant CSV search endpoint, with retries."""
    last_exc = None
    for attempt in range(5):
        try:
            r = requests.get(f"{SAVANT_CSV}?{query}", timeout=300)
            r.raise_for_status()
            text = r.content.decode("utf-8")
            break
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Savant CSV request failed: {query[:120]}") from last_exc
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if "error" in df.columns:
        raise RuntimeError(f"Savant returned an error: {df['error'].iloc[0]}")
    return df


def pull_days_savant(s: date, e: date) -> pd.DataFrame:
    """Fallback: pull [s, e] one day at a time straight from the Savant CSV endpoint."""
    frames = []
    d = s
    while d <= e:
        frames.append(_savant_csv(SAVANT_DAY_QUERY.format(d=d)))
        d += timedelta(days=1)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pull_games_savant(game_pks) -> pd.DataFrame:
    """Pull individual games by game_pk from the Savant CSV endpoint."""
    frames = [_savant_csv(SAVANT_GAME_QUERY.format(pk=int(pk))) for pk in game_pks]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pull_chunk(s: date, e: date) -> tuple[pd.DataFrame, str]:
    """Pull one chunk with pybaseball; fall back to the raw Savant endpoint on failure."""
    import pybaseball  # imported lazily: it is slow to import

    pybaseball.cache.enable()
    try:
        df = pybaseball.statcast(start_dt=str(s), end_dt=str(e), verbose=False)
        return df, "pybaseball"
    except Exception as exc:  # noqa: BLE001 - any scraping failure triggers the fallback
        log.warning("pybaseball failed for %s..%s (%s: %s). Falling back to the Savant CSV "
                    "endpoint (day-by-day requests).", s, e, type(exc).__name__, exc)
        return pull_days_savant(s, e), "savant_csv"


def write_parquet_atomic(df: pd.DataFrame, path) -> None:
    """Write via a temp file so an interrupted run never leaves a half-written chunk."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# Loading (used by later steps)
# --------------------------------------------------------------------------------------
def load_statcast(season: int, columns: list[str] | None = STATCAST_COLS,
                  regular_season_only: bool = True, logger=None) -> pd.DataFrame:
    """Concatenate a season's cached chunks; optionally keep only game_type == 'R'.

    Duplicate pitches (same game_pk, at_bat_number, pitch_number) can occur when a game is
    present in two chunks (e.g. a backfilled game); they are dropped and the count logged.
    """
    logger = logger or log
    files = sorted(DATA_RAW.glob(f"statcast_{season}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Statcast chunks for {season} in {DATA_RAW}. Run src.fetch.")
    frames = []
    for f in files:
        names = pq.read_schema(f).names
        cols = names if columns is None else [c for c in columns if c in names]
        if not cols:
            continue
        part = pd.read_parquet(f, columns=cols)
        if len(part):
            frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"], keep="first")
    if len(df) < n0:
        logger.info("load_statcast(%s): dropped %d duplicate pitch rows", season, n0 - len(df))
    if regular_season_only:
        counts = df["game_type"].value_counts(dropna=False).to_dict()
        non_r = {k: v for k, v in counts.items() if k != "R"}
        if non_r:
            logger.info("load_statcast(%s): excluding %d non-regular-season pitch rows by "
                        "game_type: %s", season, sum(non_r.values()), non_r)
        df = df[df["game_type"] == "R"]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def verify_window(season: int) -> tuple[date, date]:
    start, end = mlb_api.regular_season_window(season)
    log.info("%s regular season per MLB Stats API: %s to %s", season, start, end)
    if season in EXPECTED_WINDOW and (start, end) != EXPECTED_WINDOW[season]:
        raise SystemExit(
            f"STOP: the MLB Stats API gives {season} regular-season dates {start}..{end}, but "
            f"the project spec says {EXPECTED_WINDOW[season]}. Resolve before continuing.")
    return date.fromisoformat(start), date.fromisoformat(end)


def fetch_season(season: int) -> None:
    start, end = verify_window(season)
    chunks = list(week_chunks(start, end))
    log.info("%s: %d weekly chunks", season, len(chunks))
    for i, (s, e) in enumerate(chunks, 1):
        path = chunk_path(season, s, e)
        if path.exists():
            log.info("  [%d/%d] %s..%s already on disk, skipping", i, len(chunks), s, e)
            continue
        t0 = time.time()
        df, method = pull_chunk(s, e)
        if df.empty:
            log.warning("  [%d/%d] %s..%s returned no rows", i, len(chunks), s, e)
            df = pd.DataFrame({c: pd.Series(dtype="object") for c in STATCAST_COLS})
        write_parquet_atomic(df, path)
        n_games = df["game_pk"].nunique() if len(df) else 0
        log.info("  [%d/%d] %s..%s: %d pitches, %d games via %s (%.0fs)",
                 i, len(chunks), s, e, len(df), n_games, method, time.time() - t0)


def check_coverage(season: int, backfill: bool = True) -> dict:
    """Compare Statcast regular-season games with the schedule's completed games."""
    sc = load_statcast(season, columns=["game_pk", "game_type", "game_date",
                                        "at_bat_number", "pitch_number"],
                       regular_season_only=False)
    reg = sc[sc["game_type"] == "R"]
    expected = mlb_api.completed_games(season)
    exp_pks = set(expected["game_pk"])
    got_pks = set(reg["game_pk"].unique())
    missing = sorted(exp_pks - got_pks)
    extra = sorted(got_pks - exp_pks)

    if missing and backfill:
        log.warning("%s: %d completed games missing from Statcast chunks; re-requesting them "
                    "one at a time from the Savant CSV endpoint: %s", season, len(missing), missing)
        bf = pull_games_savant(missing)
        if len(bf):
            path = DATA_RAW / f"statcast_{season}_backfill.parquet"
            if path.exists():
                bf = pd.concat([pd.read_parquet(path), bf], ignore_index=True)
            write_parquet_atomic(bf, path)
        return check_coverage(season, backfill=False)

    gd = pd.to_datetime(reg["game_date"])
    summary = {
        "season": season,
        "pitches_all_types": len(sc),
        "pitches_regular": len(reg),
        "games_statcast": len(got_pks),
        "games_schedule_completed": len(exp_pks),
        "missing": missing,
        "extra": extra,
        "first_date": gd.min().date().isoformat(),
        "last_date": gd.max().date().isoformat(),
        "non_R_game_types": sc.loc[sc["game_type"] != "R", "game_type"].value_counts().to_dict(),
    }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[ANALYSIS_SEASON] + [s for s in PF_SEASONS if s != ANALYSIS_SEASON])
    args = ap.parse_args(argv)

    for season in args.seasons:
        log.info("=" * 70)
        log.info("Fetching %s", season)
        fetch_season(season)
        cov = check_coverage(season)
        sched = mlb_api.schedule_entries(season)
        status_counts = sched.drop_duplicates("game_pk", keep="last")["status"].value_counts().to_dict()
        log.info("%s coverage: %d regular-season pitches in %d games (schedule: %d completed); "
                 "dates %s..%s", season, cov["pitches_regular"], cov["games_statcast"],
                 cov["games_schedule_completed"], cov["first_date"], cov["last_date"])
        log.info("%s schedule: %d unique games; final status counts %s", season,
                 sched["game_pk"].nunique(), status_counts)
        if cov["non_R_game_types"]:
            log.info("%s: non-regular-season rows in the raw chunks (excluded later): %s",
                     season, cov["non_R_game_types"])
        if cov["extra"]:
            log.warning("%s: %d Statcast games are not completed games in the schedule: %s",
                        season, len(cov["extra"]), cov["extra"])
        if cov["missing"]:
            raise SystemExit(
                f"STOP: {len(cov['missing'])} completed {season} games are missing from Statcast "
                f"even after re-requesting them individually: {cov['missing']}")
        log.info("%s: all %d completed games present.", season, cov["games_schedule_completed"])


if __name__ == "__main__":
    main()
