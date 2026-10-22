"""Step 3: WHIP park factors ((H + BB) per out) and park tiers.

Park factor
-----------
For each venue V that was team T's home park in seasons S_V (within 2024-2026):

    rate_at_venue = (H + BB) / outs in all games at V in S_V        (both teams batting)
    rate_road     = (H + BB) / outs in T's road games in S_V        (both teams batting)
    PF_raw        = rate_at_venue / rate_road

T's hitters and pitchers appear in both samples, so their quality largely cancels.
Neutral-site games are excluded from both samples.

Regression toward 1.0
---------------------
    PF_reg = (outs_V * PF_raw + K * 1.0) / (outs_V + K)

K is the number of outs in one typical season of home games at a venue (the median
venue-season, ~81 games x ~53 outs, both teams). A park with three seasons of data keeps ~3/4
of its raw deviation from 1.0; a park with one season keeps ~1/2. This is a conventional
amount of regression for single-season park factors, whose year-to-year noise is roughly as
large as the true spread between parks.

Uncertainty
-----------
A game-level bootstrap (resampling home games and road games) gives a standard error for
each PF and the share of resamples in which each venue lands in its assigned tier.

Tiers (2026)
------------
Venues used in 2026 are sorted by PF_reg and cut into three tiers holding as close as
possible to a third of 2026 league innings each (innings at the venue, both teams).
Venues that were never anyone's home park in 2024-2026 (neutral-site-only: Las Vegas,
Mexico City, Field of Dreams, Williamsport) have no PF estimate and go to the Neutral tier.
The 2-tier version is Hitter-friendly vs. everything else ("Other").

Outputs: outputs/tables/park_factors.csv, data/processed/venue_tiers_2026.csv
Usage: python -m src.park_factors
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .attribution import load_pitcher_games
from .common import get_logger
from .config import ANALYSIS_SEASON, DATA_PROCESSED, OUT_TABLES, PF_SEASONS

log = get_logger("park_factors", "step3_park_factors.log")

TIERS_3 = ["Pitcher-friendly", "Neutral", "Hitter-friendly"]
TIERS_2 = ["Other", "Hitter-friendly"]
N_BOOT = 1000
SEED = 20260927


def game_totals(seasons=PF_SEASONS) -> pd.DataFrame:
    """One row per game: H, BB, outs for both teams combined, plus venue/team context."""
    frames = []
    for season in seasons:
        pg = load_pitcher_games(season)
        g = (pg.groupby(["season", "game_pk", "venue_id", "venue_name", "is_neutral"])
             .agg(h=("h", "sum"), bb=("bb", "sum"), outs=("outs", "sum")).reset_index())
        teams = pg[pg["is_home"]].groupby("game_pk")["team_id"].first().rename("home_id")
        away = pg[~pg["is_home"]].groupby("game_pk")["team_id"].first().rename("away_id")
        g = g.join(teams, on="game_pk").join(away, on="game_pk")
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    g["hbb"] = g["h"] + g["bb"]
    return g


def home_parks(gt: pd.DataFrame) -> pd.DataFrame:
    """(venue, home team, season) combinations where the venue was that team's home park."""
    hp = (gt[~gt["is_neutral"]].groupby(["venue_id", "home_id", "season"]).size()
          .rename("home_games").reset_index())
    multi = hp.groupby("venue_id")["home_id"].nunique()
    if (multi > 1).any():
        raise ValueError(f"Venues serving as home park for more than one team: {list(multi[multi > 1].index)}")
    return hp


def compute_pf(gt: pd.DataFrame, hp: pd.DataFrame, k_outs: float) -> pd.DataFrame:
    rows = []
    for vid, sub in hp.groupby("venue_id"):
        team = sub["home_id"].iloc[0]
        seasons = sorted(sub["season"])
        at = gt[(gt["venue_id"] == vid) & (~gt["is_neutral"]) & gt["season"].isin(seasons)]
        road = gt[(gt["away_id"] == team) & (~gt["is_neutral"]) & gt["season"].isin(seasons)]
        rate_v = at["hbb"].sum() / at["outs"].sum()
        rate_r = road["hbb"].sum() / road["outs"].sum()
        pf_raw = rate_v / rate_r
        outs_v = at["outs"].sum()
        rows.append({
            "venue_id": vid, "venue": at["venue_name"].iloc[0], "home_team_id": team,
            "seasons_used": ", ".join(map(str, seasons)), "n_seasons": len(seasons),
            "home_games": len(at), "outs_at_venue": int(outs_v),
            "hbb_at_venue": int(at["hbb"].sum()), "rate_at_venue": rate_v,
            "road_games": len(road), "outs_road": int(road["outs"].sum()), "rate_road": rate_r,
            "pf_raw": pf_raw,
            "pf_regressed": (outs_v * pf_raw + k_outs * 1.0) / (outs_v + k_outs),
        })
    return pd.DataFrame(rows)


def bootstrap_pf(gt: pd.DataFrame, hp: pd.DataFrame, k_outs: float, n_boot=N_BOOT,
                 seed=SEED) -> pd.DataFrame:
    """Game-level bootstrap of PF_regressed. Returns an (n_boot x venues) frame."""
    rng = np.random.default_rng(seed)
    out = {}
    for vid, sub in hp.groupby("venue_id"):
        team = sub["home_id"].iloc[0]
        seasons = sub["season"].tolist()
        at = gt[(gt["venue_id"] == vid) & (~gt["is_neutral"]) & gt["season"].isin(seasons)]
        road = gt[(gt["away_id"] == team) & (~gt["is_neutral"]) & gt["season"].isin(seasons)]

        def resample(df):
            idx = rng.integers(0, len(df), size=(n_boot, len(df)))
            return df["hbb"].to_numpy()[idx].sum(1), df["outs"].to_numpy()[idx].sum(1)

        hv, ov = resample(at)
        hr, orr = resample(road)
        pf_raw = (hv / ov) / (hr / orr)
        out[vid] = (ov * pf_raw + k_outs) / (ov + k_outs)
    return pd.DataFrame(out)


def tier_cuts(pf_sorted: pd.Series, weights: pd.Series) -> tuple[int, int]:
    """Positions (in ascending-PF order) of the last venue in tiers 1 and 2."""
    cum = weights.cumsum().to_numpy() / weights.sum()
    c1 = int(np.argmin(np.abs(cum - 1 / 3)))
    c2 = int(np.argmin(np.abs(cum - 2 / 3)))
    return c1, c2


def assign_tiers(pf_2026: pd.DataFrame) -> pd.DataFrame:
    """pf_2026: venues used in 2026 with pf_regressed (NaN if not estimable) and ip_2026."""
    df = pf_2026.copy()
    est = df[df["pf_regressed"].notna()].sort_values("pf_regressed").reset_index(drop=True)
    c1, c2 = tier_cuts(est["pf_regressed"], est["ip_2026"])
    est["tier_3"] = np.where(est.index <= c1, TIERS_3[0],
                             np.where(est.index <= c2, TIERS_3[1], TIERS_3[2]))
    est["tier_source"] = "park factor"
    noest = df[df["pf_regressed"].isna()].copy()
    noest["tier_3"] = "Neutral"
    noest["tier_source"] = "no park-factor estimate (neutral-site-only venue) -> Neutral"
    out = pd.concat([est, noest], ignore_index=True)
    out["tier_2"] = np.where(out["tier_3"] == "Hitter-friendly", "Hitter-friendly", "Other")
    return out


def main():
    gt = game_totals()
    hp = home_parks(gt)
    venue_season_outs = (gt[~gt["is_neutral"]].groupby(["venue_id", "season"])["outs"].sum())
    k_outs = float(venue_season_outs.median())
    log.info("=" * 70)
    log.info("Step 3: park factors from %d games (%d neutral-site games excluded)", len(gt),
             int(gt["is_neutral"].sum()))
    log.info("Regression prior K = %.0f outs (median outs in one venue-season of home games, "
             "both teams; ~%.0f games)", k_outs,
             k_outs / gt.loc[~gt["is_neutral"], "outs"].mean())

    pf = compute_pf(gt, hp, k_outs)
    boot = bootstrap_pf(gt, hp, k_outs)
    pf["pf_regressed_se"] = pf["venue_id"].map(boot.std())

    names = (load_pitcher_games(ANALYSIS_SEASON)[["team_id", "team"]].drop_duplicates()
             .set_index("team_id")["team"])
    team_names = {}
    for season in PF_SEASONS:
        pgs = load_pitcher_games(season)[["team_id", "team"]].drop_duplicates()
        team_names.update(dict(zip(pgs["team_id"], pgs["team"])))
    team_names.update(names.to_dict())
    pf["team"] = pf["home_team_id"].map(team_names)

    # 2026 usage of every venue (innings at the venue, both teams, all 2026 games there).
    g26 = gt[gt["season"] == ANALYSIS_SEASON]
    use = (g26.groupby(["venue_id", "venue_name"])
           .agg(games_2026=("game_pk", "size"), outs_2026=("outs", "sum"),
                neutral_games_2026=("is_neutral", "sum")).reset_index())
    use["ip_2026"] = use["outs_2026"] / 3
    use["ip_share_2026"] = use["ip_2026"] / use["ip_2026"].sum()
    v26 = use.merge(pf.drop(columns=["venue"]), on="venue_id", how="left")
    tiers = assign_tiers(v26)

    # Tier stability under the bootstrap: recompute cuts in each resample.
    est = tiers[tiers["tier_source"] == "park factor"]
    same = np.zeros(len(est))
    ids = est["venue_id"].to_numpy()
    w = est.set_index("venue_id")["ip_2026"]
    assigned = est.set_index("venue_id")["tier_3"]
    for b in range(len(boot)):
        pfb = boot.loc[b, ids].sort_values()
        c1, c2 = tier_cuts(pfb, w.loc[pfb.index])
        tb = pd.Series(np.where(np.arange(len(pfb)) <= c1, TIERS_3[0],
                                np.where(np.arange(len(pfb)) <= c2, TIERS_3[1], TIERS_3[2])),
                       index=pfb.index)
        same += (tb.loc[ids].to_numpy() == assigned.loc[ids].to_numpy())
    tiers["tier_bootstrap_agreement"] = np.nan
    tiers.loc[tiers["tier_source"] == "park factor", "tier_bootstrap_agreement"] = same / len(boot)

    tiers = tiers.sort_values("pf_regressed", ascending=False, na_position="last")
    tiers.to_csv(DATA_PROCESSED / f"venue_tiers_{ANALYSIS_SEASON}.csv", index=False)

    # Park-factor table: all estimable venues (2024-2026) + 2026 tiers where applicable.
    table = pf.merge(tiers[["venue_id", "games_2026", "ip_2026", "ip_share_2026", "tier_3",
                            "tier_2", "tier_bootstrap_agreement"]], on="venue_id", how="left")
    noest = tiers[tiers["pf_regressed"].isna()][["venue_id", "venue_name", "games_2026",
                                                  "ip_2026", "ip_share_2026", "tier_3", "tier_2"]]
    noest = noest.rename(columns={"venue_name": "venue"})
    noest["seasons_used"] = "none (neutral-site only)"
    noest["n_seasons"] = 0
    table = pd.concat([table, noest], ignore_index=True)
    table["tier_3"] = table["tier_3"].fillna("not used in 2026")
    table["tier_2"] = table["tier_2"].fillna("not used in 2026")
    table = table.sort_values("pf_regressed", ascending=False, na_position="last")
    table["rank"] = np.arange(1, len(table) + 1)
    table.loc[table["pf_regressed"].isna(), "rank"] = np.nan
    cols = ["rank", "venue", "team", "seasons_used", "n_seasons", "home_games", "outs_at_venue",
            "rate_at_venue", "road_games", "outs_road", "rate_road", "pf_raw", "pf_regressed",
            "pf_regressed_se", "tier_3", "tier_2", "tier_bootstrap_agreement", "games_2026",
            "ip_2026", "ip_share_2026", "venue_id", "home_team_id"]
    for c in ["rank", "n_seasons", "home_games", "outs_at_venue", "road_games", "outs_road",
              "games_2026", "venue_id", "home_team_id"]:
        table[c] = table[c].round().astype("Int64")       # integer columns (NaN-safe)
    table[cols].to_csv(OUT_TABLES / "park_factors.csv", index=False, float_format="%.4f")

    # ---- Summary -------------------------------------------------------------------
    show = table[cols].copy()
    log.info("Park factors (H+BB per out, venue vs. home team's road games):\n%s",
             show[["rank", "venue", "team", "seasons_used", "pf_raw", "pf_regressed",
                   "pf_regressed_se", "tier_3", "tier_bootstrap_agreement", "games_2026"]]
             .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    top = table.dropna(subset=["pf_regressed"]).head(3)
    gap = top["pf_regressed"].iloc[0] - top["pf_regressed"].iloc[1]
    log.info("Most hitter-friendly: %s (PF %.3f), next %s (PF %.3f); margin %.3f",
             top["venue"].iloc[0], top["pf_regressed"].iloc[0], top["venue"].iloc[1],
             top["pf_regressed"].iloc[1], gap)
    if top["venue"].iloc[0] != "Coors Field":
        log.warning("SANITY CHECK FAILED: Coors Field is not the most hitter-friendly park. "
                    "Investigate before continuing.")

    share = tiers.groupby("tier_3")["ip_share_2026"].sum()
    lo = tiers.groupby("tier_3")["pf_regressed"].agg(["min", "max", "count"])
    for t in TIERS_3[::-1]:
        parks = tiers[tiers["tier_3"] == t]
        log.info("%s tier: %.1f%% of 2026 innings, %d venues, PF %.3f-%.3f:\n    %s", t,
                 100 * share.get(t, 0), len(parks), lo.loc[t, "min"], lo.loc[t, "max"],
                 "; ".join(f"{r.venue_name} ({r.pf_regressed:.3f})" if pd.notna(r.pf_regressed)
                           else f"{r.venue_name} (no PF; neutral site)" for r in parks.itertuples()))
    log.info("2-tier version: Hitter-friendly %.1f%% of 2026 innings, Other %.1f%%",
             100 * share.get("Hitter-friendly", 0), 100 * (1 - share.get("Hitter-friendly", 0)))


if __name__ == "__main__":
    main()
