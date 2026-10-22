"""Shared logic: credit outs, hits and walks to pitchers from pitch-level Statcast data.

Outs (Step 4)
-------------
Official innings pitched include outs that do not end a plate appearance (caught stealing,
pickoffs, runners thrown out on the bases), so outs are NOT counted from `events`:

* Pitches are sorted by game_pk, inning, half (Top before Bot), at_bat_number, pitch_number.
* `outs_when_up` is the number of outs before each pitch. Within a half-inning the outs
  recorded on or after pitch i (and before pitch i+1) are outs_when_up[i+1] - outs_when_up[i].
* The last pitch of a half-inning is credited (outs at the end of the half-inning) -
  outs_when_up. That end count is 3, except in the final half-inning of a game, which can end
  early: walk-offs (including extra-inning walk-offs with the automatic runner on second) and
  games called for weather. For that half-inning the end count comes from the official final
  linescore (MLB Stats API `linescore.outs`).
* Outs recorded before the first pitch of a half-inning (e.g. the automatic runner picked
  off before any pitch) appear as outs_when_up > 0 on the first pitch and are credited to the
  pitcher who threw it.
* Outs are credited to the `pitcher` on the pitch.

Hits and walks
--------------
* Hits: events in {single, double, triple, home_run}; charged to the pitcher on that pitch.
* Walks: events in {walk, intent_walk} (intentional walks count; HBP does not). Official
  Baseball Rule 9.16(h): when a reliever enters mid-plate-appearance with the count 2-0, 2-1,
  3-0, 3-1 or 3-2 and the batter walks, the walk is charged to the previous pitcher.

Output: one row per pitcher per game (data/processed/pitcher_games_<season>.parquet).

Usage: python -m src.attribution        # builds 2024-2026 and reconciles games vs linescores
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import mlb_api
from .common import get_logger
from .config import DATA_PROCESSED, OUT_TABLES, PF_SEASONS
from .fetch import load_statcast

log = get_logger("attribution", "step4_attribution.log")

HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
WALK_EVENTS = frozenset({"walk", "intent_walk"})
# Counts at which a mid-PA pitching change leaves a subsequent walk on the previous pitcher.
WALK_CHARGEBACK_COUNTS = frozenset({(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)})

HALF_KEY = ["game_pk", "inning", "half"]
PA_KEY = ["game_pk", "at_bat_number"]


def order_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Add `half` (0 = Top, 1 = Bot) and sort pitches into game order."""
    out = df.copy()
    out["half"] = out["inning_topbot"].map({"Top": 0, "Bot": 1})
    if out["half"].isna().any():
        raise ValueError(f"Unexpected inning_topbot values: {out['inning_topbot'].unique()}")
    for c in ["game_pk", "inning", "at_bat_number", "pitch_number", "outs_when_up", "pitcher"]:
        out[c] = out[c].astype("int64")
    out = out.sort_values(["game_pk", "inning", "half", "at_bat_number", "pitch_number"],
                          kind="mergesort").reset_index(drop=True)
    return out


def final_half_state(season: int) -> pd.DataFrame:
    """Official final half-inning of each game and the outs when the game ended.

    Index game_pk; columns inning, half (0 = Top, 1 = Bot), outs.
    """
    done = mlb_api.completed_games(season)
    fs = pd.DataFrame({
        "game_pk": done["game_pk"],
        "inning": done["ls_inning"],
        "half": np.where(done["ls_is_top"].astype(bool), 0, 1),
        "outs": done["ls_outs"],
    }).dropna()
    return fs.astype("int64").set_index("game_pk")


def credit_outs(df: pd.DataFrame, final_state: pd.DataFrame | None = None) -> pd.Series:
    """Outs credited to the pitcher on each pitch. `df` must be sorted by `order_pitches`.

    final_state: per game_pk, the official final half-inning (inning, half) and the outs
    when the game ended (see `final_half_state`). The end-of-game out count is applied to
    the last half-inning in the pitch data only when that half-inning *is* the official final
    one; otherwise (e.g. a game called before any pitch of its last half-inning) the last
    half-inning with pitches ended normally with 3 outs. Games without a final_state are
    assumed to end with 3 outs.
    """
    g = df.groupby(HALF_KEY, sort=False)
    owu = df["outs_when_up"].astype(int)
    nxt = g["outs_when_up"].shift(-1)
    is_first = g.cumcount() == 0
    is_last = g.cumcount(ascending=False) == 0

    gg = df.groupby("game_pk", sort=False)
    is_final_half = ((df["inning"] == gg["inning"].transform("last"))
                     & (df["half"] == gg["half"].transform("last")))
    end_outs = pd.Series(3, index=df.index, dtype="int64")
    if final_state is not None and len(final_state):
        fs = final_state.reindex(df["game_pk"].to_numpy())
        same_half = ((fs["inning"].to_numpy() == df["inning"].to_numpy())
                     & (fs["half"].to_numpy() == df["half"].to_numpy()))
        use = is_final_half.to_numpy() & same_half & fs["outs"].notna().to_numpy()
        end_outs[use] = fs["outs"].to_numpy()[use].astype("int64")

    outs = np.where(is_last, end_outs - owu, nxt.fillna(0).astype(int) - owu)
    outs = outs + np.where(is_first, owu, 0)
    return pd.Series(outs, index=df.index, name="outs", dtype="int64")


def walk_pitcher(df: pd.DataFrame) -> pd.Series:
    """Pitcher charged with each pitch's walk (rule 9.16(h)); equals `pitcher` otherwise.

    `df` must be sorted by `order_pitches`.
    """
    pa = df.groupby(PA_KEY, sort=False)
    prev = pa["pitcher"].shift(1)
    change = prev.notna() & (prev != df["pitcher"])
    # Count and outgoing pitcher at the most recent pitching change within the PA.
    info = pd.DataFrame({
        "chg_balls": df["balls"].where(change),
        "chg_strikes": df["strikes"].where(change),
        "chg_prev": prev.where(change),
    })
    info = info.groupby([df["game_pk"], df["at_bat_number"]], sort=False).ffill()
    counts = list(zip(info["chg_balls"].fillna(-1).astype(int),
                      info["chg_strikes"].fillna(-1).astype(int)))
    in_set = np.fromiter((c in WALK_CHARGEBACK_COUNTS for c in counts), bool, len(counts))
    charge_back = df["events"].isin(WALK_EVENTS).to_numpy() & info["chg_prev"].notna().to_numpy() & in_set
    out = df["pitcher"].astype("int64").copy()
    out[charge_back] = info.loc[charge_back, "chg_prev"].astype("int64")
    return out.rename("bb_pitcher")


def attribute(sc: pd.DataFrame, final_state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pitch-level table with credited outs, hit/walk flags and the walk-charged pitcher."""
    df = order_pitches(sc)
    df["outs"] = credit_outs(df, final_state)
    df["is_hit"] = df["events"].isin(HIT_EVENTS)
    df["is_walk"] = df["events"].isin(WALK_EVENTS)
    df["is_ibb"] = df["events"].eq("intent_walk").fillna(False)
    # Completed plate appearances (truncated_pa = PA cut short by an inning-ending baserunning out).
    df["is_pa_end"] = df["events"].notna() & df["events"].ne("truncated_pa")
    df["bb_pitcher"] = walk_pitcher(df)
    df["seq"] = np.arange(len(df))
    return df


def pitcher_game_table(pitches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate attributed pitches to one row per (game, pitcher) with game/venue context."""
    p = pitches
    # The fielding side is home in the top half, away in the bottom half.
    p = p.assign(side=np.where(p["half"] == 0, "home", "away"))
    base = (p.groupby(["game_pk", "pitcher"], sort=False)
            .agg(side=("side", "first"), n_sides=("side", "nunique"), outs=("outs", "sum"),
                 h=("is_hit", "sum"), pitches=("seq", "size"), bf=("is_pa_end", "sum"),
                 first_seq=("seq", "min"))
            .reset_index())
    if (base["n_sides"] > 1).any():
        raise ValueError("A pitcher appears for both teams in the same game")
    walks = p[p["is_walk"]]
    bb = walks.groupby(["game_pk", "bb_pitcher"]).size().rename("bb")
    ibb = walks[walks["is_ibb"]].groupby(["game_pk", "bb_pitcher"]).size().rename("ibb")
    bbs = pd.concat([bb, ibb], axis=1).fillna(0).astype(int)
    bbs.index.names = ["game_pk", "pitcher"]
    base = base.merge(bbs.reset_index(), on=["game_pk", "pitcher"], how="left")
    orphan = set(map(tuple, bbs.reset_index()[["game_pk", "pitcher"]].values)) - set(
        map(tuple, base[["game_pk", "pitcher"]].values))
    if orphan:
        raise ValueError(f"Walks charged to pitchers with no pitches in the game: {orphan}")
    base[["bb", "ibb"]] = base[["bb", "ibb"]].fillna(0).astype(int)

    # Starter = first pitcher used by each team in each game.
    first = base.groupby(["game_pk", "side"])["first_seq"].transform("min")
    base["started"] = base["first_seq"] == first

    ctx = games[["game_pk", "season", "official_date", "venue_id", "venue_name", "is_neutral",
                 "home_id", "home_name", "away_id", "away_name"]]
    base = base.merge(ctx, on="game_pk", how="left", validate="many_to_one")
    if base["venue_id"].isna().any():
        raise ValueError("Pitcher-games without a venue: run src.venues first")
    base["is_home"] = base["side"] == "home"
    base["team_id"] = np.where(base["is_home"], base["home_id"], base["away_id"])
    base["team"] = np.where(base["is_home"], base["home_name"], base["away_name"])
    base["opp_id"] = np.where(base["is_home"], base["away_id"], base["home_id"])
    cols = ["season", "game_pk", "official_date", "pitcher", "team_id", "team", "opp_id",
            "is_home", "venue_id", "venue_name", "is_neutral", "started", "outs", "h", "bb",
            "ibb", "bf", "pitches"]
    return base[cols].sort_values(["game_pk", "team_id", "started"],
                                  ascending=[True, True, False]).reset_index(drop=True)


def reconcile_games(pitches: pd.DataFrame, season: int) -> pd.DataFrame:
    """Compare Statcast half-innings, hits and outs per game with the official linescore."""
    ls = mlb_api.linescore_halves(season)
    sc = (pitches.groupby(HALF_KEY)
          .agg(sc_hits=("is_hit", "sum"), sc_outs=("outs", "sum")).reset_index())
    m = ls.merge(sc, on=HALF_KEY, how="outer", indicator=True)
    m["exp_outs"] = np.where(m["final_outs"].notna(), m["final_outs"].fillna(3), 3).astype(float)
    # An official half-inning with no pitches, no outs and no hits (game called before the
    # first pitch of it) is not a data gap.
    benign = (m["_merge"] == "left_only") & (m["exp_outs"] == 0) & (m["hits"] == 0)
    m.loc[benign, "_merge"] = "both"
    per_game = m.groupby("game_pk").agg(
        halves_official=("runs", lambda s: s.notna().sum()),
        halves_statcast=("sc_outs", lambda s: s.notna().sum()),
        hits_official=("hits", "sum"), hits_statcast=("sc_hits", "sum"),
        outs_expected=("exp_outs", "sum"), outs_statcast=("sc_outs", "sum"),
        unmatched=("_merge", lambda s: (s != "both").sum()),
    ).reset_index()
    per_game["ok"] = ((per_game["unmatched"] == 0)
                      & (per_game["hits_official"] == per_game["hits_statcast"])
                      & (per_game["outs_expected"] == per_game["outs_statcast"]))
    # A few official linescores come back with an empty inning list (an API gap, not a
    # Statcast gap); those games cannot be reconciled half-inning by half-inning.
    per_game["official_detail"] = per_game["halves_official"] > 0
    return per_game


def pitcher_games_path(season: int):
    return DATA_PROCESSED / f"pitcher_games_{season}.parquet"


def build(season: int, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build and cache the pitcher-game table for one season; returns it."""
    if games is None:
        games = pd.read_parquet(DATA_PROCESSED / "games.parquet")
    games = games[games["season"] == season]
    sc = load_statcast(season, logger=log)
    final_state = final_half_state(season)
    pitches = attribute(sc, final_state)

    # Games whose official final half-inning has no pitches in Statcast (called before a
    # pitch was thrown in it): the last half-inning with pitches ended normally.
    last_sc = pitches.groupby("game_pk")[["inning", "half"]].last()
    fs = final_state.reindex(last_sc.index)
    differs = last_sc[(last_sc["inning"] != fs["inning"]) | (last_sc["half"] != fs["half"])]
    if len(differs):
        log.info("%s: %d game(s) whose official final half-inning has no Statcast pitches: %s",
                 season, len(differs), list(differs.index))

    neg = pitches[pitches["outs"] < 0]
    if len(neg):
        log.warning("%s: %d pitches with negative credited outs (data glitch):\n%s", season,
                    len(neg), neg[["game_pk", "inning", "half", "at_bat_number", "pitch_number",
                                   "outs_when_up", "events"]].head(20).to_string())
    n_cb = int((pitches["bb_pitcher"] != pitches["pitcher"]).sum())
    log.info("%s: %d pitches, %d half-innings, %d outs credited, %d hits, %d walks "
             "(%d intentional); %d walk(s) charged back to the previous pitcher (rule 9.16(h))",
             season, len(pitches), pitches.groupby(HALF_KEY).ngroups, int(pitches["outs"].sum()),
             int(pitches["is_hit"].sum()), int(pitches["is_walk"].sum()),
             int(pitches["is_ibb"].sum()), n_cb)

    rec = reconcile_games(pitches, season)
    rec.to_csv(OUT_TABLES / f"game_reconciliation_{season}.csv", index=False)
    no_detail = rec[~rec["official_detail"]]
    rec_d = rec[rec["official_detail"]]
    bad = rec_d[~rec_d["ok"]]
    log.info("%s: game-level reconciliation vs official linescores: %d/%d games match exactly on "
             "half-innings, hits and outs", season, int(rec_d["ok"].sum()), len(rec_d))
    if len(no_detail):
        log.info("%s: %d game(s) have no inning-by-inning official linescore in the API and "
                 "were not reconciled: %s", season, len(no_detail), list(no_detail["game_pk"]))
    if len(bad):
        log.info("%s mismatching games (hit differences are official scoring changes made after "
                 "the game, which Statcast's events do not reflect):\n%s", season,
                 bad.head(25).to_string(index=False))

    pg = pitcher_game_table(pitches, games)
    pg.to_parquet(pitcher_games_path(season), index=False)
    log.info("%s: %d pitcher-game rows for %d pitchers saved to %s", season, len(pg),
             pg["pitcher"].nunique(), pitcher_games_path(season).name)
    return pg


def load_pitcher_games(season: int) -> pd.DataFrame:
    path = pitcher_games_path(season)
    if not path.exists():
        return build(season)
    return pd.read_parquet(path)


# --------------------------------------------------------------------------------------
# Validation against official season lines (Step 4)
# --------------------------------------------------------------------------------------
VALIDATION_MIN_OUTS = 60          # pitchers with 20+ IP
VALIDATION_TOL = 1                # within 1 out and 1 H / 1 BB
VALIDATION_REQUIRED_SHARE = 0.95


def season_totals(pg: pd.DataFrame) -> pd.DataFrame:
    return (pg.groupby("pitcher")
            .agg(g=("game_pk", "nunique"), gs=("started", "sum"), outs=("outs", "sum"),
                 h=("h", "sum"), bb=("bb", "sum"), ibb=("ibb", "sum"))
            .reset_index())


def compare_to_official(season: int, pg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Computed vs official (MLB Stats API) season totals, one row per pitcher."""
    pg = load_pitcher_games(season) if pg is None else pg
    ours = season_totals(pg)
    off = mlb_api.season_pitching_stats(season)[["pitcher", "name", "g", "gs", "outs", "h",
                                                  "bb", "ibb"]]
    m = ours.merge(off, on="pitcher", how="outer", suffixes=("_calc", "_off"), indicator=True)
    m["source"] = m.pop("_merge").map({"both": "both", "left_only": "statcast_only",
                                       "right_only": "official_only"}).astype(str)
    for c in ["g", "gs", "outs", "h", "bb", "ibb"]:
        for side in ("calc", "off"):
            m[f"{c}_{side}"] = m[f"{c}_{side}"].fillna(0).astype(int)
        m[f"d_{c}"] = m[f"{c}_calc"] - m[f"{c}_off"]
    m["exact"] = (m["d_outs"] == 0) & (m["d_h"] == 0) & (m["d_bb"] == 0)
    m["within_tol"] = ((m["d_outs"].abs() <= VALIDATION_TOL) & (m["d_h"].abs() <= VALIDATION_TOL)
                       & (m["d_bb"].abs() <= VALIDATION_TOL))
    m["abs_err"] = m["d_outs"].abs() + m["d_h"].abs() + m["d_bb"].abs()
    m["qualifies_20ip"] = m["outs_off"] >= VALIDATION_MIN_OUTS
    return m.sort_values("abs_err", ascending=False).reset_index(drop=True)


def explain_discrepancies(season: int, comp: pd.DataFrame, pg: pd.DataFrame,
                          max_pitchers: int = 25) -> pd.DataFrame:
    """Game-level diff (official game logs vs computed) for the pitchers that disagree."""
    bad = comp[comp["abs_err"] > 0].head(max_pitchers)
    rows = []
    for _, r in bad.iterrows():
        gl = mlb_api.pitcher_game_log(int(r["pitcher"]), season)
        mine = pg[pg["pitcher"] == r["pitcher"]][["game_pk", "official_date", "outs", "h", "bb"]]
        d = mine.merge(gl, on="game_pk", how="outer", suffixes=("_calc", "_off"))
        for c in ["outs", "h", "bb"]:
            d[f"d_{c}"] = d[f"{c}_calc"].fillna(0) - d[f"{c}_off"].fillna(0)
        d = d[(d[["d_outs", "d_h", "d_bb"]] != 0).any(axis=1)]
        for _, x in d.iterrows():
            rows.append({"pitcher": r["pitcher"], "name": r["name"], "game_pk": x["game_pk"],
                         "date": x["official_date"] if pd.notna(x["official_date"]) else x["date"],
                         "d_outs": int(x["d_outs"]), "d_h": int(x["d_h"]), "d_bb": int(x["d_bb"])})
    return pd.DataFrame(rows)


def validation_summary(comp: pd.DataFrame) -> dict:
    q = comp[comp["qualifies_20ip"]]
    return {
        "pitchers_official": int((comp["source"] != "statcast_only").sum()),
        "pitchers_computed": int((comp["source"] != "official_only").sum()),
        "only_in_one_source": int((comp["source"] != "both").sum()),
        "share_exact_all": float(comp["exact"].mean()),
        "share_within_tol_all": float(comp["within_tol"].mean()),
        "n_20ip": int(len(q)),
        "share_exact_20ip": float(q["exact"].mean()),
        "share_within_tol_20ip": float(q["within_tol"].mean()),
        "share_g_gs_match": float(((comp["d_g"] == 0) & (comp["d_gs"] == 0)).mean()),
        "league_d_outs": int(comp["d_outs"].sum()),
        "league_d_h": int(comp["d_h"].sum()),
        "league_d_bb": int(comp["d_bb"].sum()),
    }


def validate(season: int) -> dict:
    pg = load_pitcher_games(season)
    comp = compare_to_official(season, pg)
    comp.to_csv(OUT_TABLES / f"validation_{season}.csv", index=False)
    s = validation_summary(comp)
    log.info("%s validation vs official MLB Stats API season lines:", season)
    log.info("  pitchers: %d official, %d computed, %d in only one source", s["pitchers_official"],
             s["pitchers_computed"], s["only_in_one_source"])
    log.info("  exact match on outs, H and BB: %.1f%% of all pitchers; %.1f%% of the %d pitchers "
             "with 20+ IP", 100 * s["share_exact_all"], 100 * s["share_exact_20ip"], s["n_20ip"])
    log.info("  within 1 out / 1 H / 1 BB: %.1f%% of all; %.1f%% of 20+ IP pitchers "
             "(required: %.0f%%)", 100 * s["share_within_tol_all"],
             100 * s["share_within_tol_20ip"], 100 * VALIDATION_REQUIRED_SHARE)
    log.info("  G and GS identical: %.1f%% of pitchers", 100 * s["share_g_gs_match"])
    log.info("  league totals, computed minus official: outs %+d, H %+d, BB %+d",
             s["league_d_outs"], s["league_d_h"], s["league_d_bb"])
    worst = comp[comp["abs_err"] > 0]
    if len(worst):
        log.info("  pitchers with any discrepancy (%d):\n%s", len(worst), worst[
            ["pitcher", "name", "outs_off", "d_outs", "h_off", "d_h", "bb_off", "d_bb", "d_g",
             "d_gs", "source"]].head(30).to_string(index=False))
        expl = explain_discrepancies(season, comp, pg)
        if len(expl):
            expl.to_csv(OUT_TABLES / f"validation_{season}_discrepant_games.csv", index=False)
            log.info("  game-level location of those discrepancies:\n%s",
                     expl.to_string(index=False))
    gs_off = comp[(comp["d_gs"] != 0) | (comp["d_g"] != 0)]
    if len(gs_off):
        log.info("  G/GS differences:\n%s", gs_off[["pitcher", "name", "g_calc", "g_off", "gs_calc",
                                                   "gs_off"]].head(20).to_string(index=False))
    return s


def main():
    games = pd.read_parquet(DATA_PROCESSED / "games.parquet")
    for season in PF_SEASONS:
        log.info("=" * 70)
        build(season, games)
        validate(season)


if __name__ == "__main__":
    main()
