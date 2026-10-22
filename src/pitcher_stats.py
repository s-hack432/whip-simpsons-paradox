"""Steps 4-5: 2026 pitcher WHIP overall, by park tier, home/road within tier; shrinkage;
park-standardized WHIP.

Definitions
-----------
* WHIP = (H + BB) / IP, IP = outs / 3 (BB includes intentional walks; HBP excluded).
* Role: starter (SP) if >= 50% of appearances were starts, else reliever (RP).
* Qualifying: starters 80+ IP (3-tier version), relievers 40+ IP (2-tier version).
* League tier share S_t and league tier WHIP L_t use every 2026 inning; r_t = L_t / L is the
  league's tier effect (L = league WHIP).

Shrinkage of tier WHIP
----------------------
    shrunk_t = (H_t + BB_t + k * prior_t) / (IP_t + k)

Headline version (chosen with the user; see README "Shrinkage"):
    prior_t = theta_i * r_t,  theta_i = overall_WHIP_i / sum_t(s_it * r_t),  k = 300 IP
theta_i is the pitcher's WHIP re-expressed on a league-average park mix using only his
exposure s_it; prior_t is therefore "this pitcher, in an average tier-t park". Centring the
prior on the pitcher's own level (instead of on the league-average tier WHIP) keeps shrinkage
from dragging good pitchers up and bad pitchers down. k = 300 IP is the empirical-Bayes weight:
the pitcher-by-park-tier interaction variance is small next to sampling noise (diagnostics are
recomputed and printed on every run, outputs/tables/shrinkage_diagnostics.csv).

Sensitivity versions kept in the table:
    *_spec   the project spec as written: prior_t = L_t, k = 20 IP
    *_own20  own-level prior, k = 20 IP
    exposure_adj_whip = theta_i (the k -> infinity limit: exposure only)
    std_whip_raw      = sum_t S_t * raw_t (no shrinkage; needs innings in every tier)

Standardized WHIP and gap:  std_WHIP = sum_t S_t * shrunk_t ;  whip_gap = overall - std_WHIP
(positive = underrated: the overall WHIP makes him look worse than on a league-average park mix).
For the spec version the gap is split exactly into a shrinkage part and a park-mix part:
    whip_gap = (overall - own_mix) + (own_mix - std_WHIP),   own_mix = sum_t s_it * shrunk_t

Outputs
-------
outputs/tables/pitcher_whip.csv           one row per 2026 pitcher (qualifying flag included)
outputs/tables/pitcher_tier_splits.csv    long: pitcher x scheme x tier x {All, Home, Road}
outputs/tables/league_tiers.csv           league IP share and WHIP by scheme x tier x side
outputs/tables/shrinkage_diagnostics.csv  empirical-Bayes prior-weight diagnostics

Usage: python -m src.pitcher_stats
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import mlb_api
from .attribution import load_pitcher_games
from .common import get_logger, ip_display
from .config import ANALYSIS_SEASON, DATA_PROCESSED, OUT_TABLES
from .names import player_names
from .park_factors import TIERS_2, TIERS_3

log = get_logger("pitcher_stats", "step5_pitcher_stats.log")

PRIOR_IP = 300.0            # headline prior weight (empirical-Bayes tuned)
SPEC_PRIOR_IP = 20.0        # the project spec's suggested weight, kept as a sensitivity check
VARIANTS = {                # name: (prior centre, prior weight in IP)
    "main": ("pitcher", PRIOR_IP),
    "spec": ("league", SPEC_PRIOR_IP),
    "own20": ("pitcher", SPEC_PRIOR_IP),
}
STARTER_SHARE = 0.5
MIN_IP = {"SP": 80.0, "RP": 40.0}
SCHEMES = {"3tier": ("tier_3", TIERS_3), "2tier": ("tier_2", TIERS_2)}
ROLE_SCHEME = {"SP": "3tier", "RP": "2tier"}
ABBR = {"Pitcher-friendly": "pf", "Neutral": "neu", "Hitter-friendly": "hf", "Other": "oth"}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_games_with_tiers(season: int = ANALYSIS_SEASON) -> pd.DataFrame:
    """2026 pitcher-game rows with the venue's park tiers attached."""
    pg = load_pitcher_games(season)
    tiers = pd.read_csv(DATA_PROCESSED / f"venue_tiers_{season}.csv")[["venue_id", "tier_3",
                                                                       "tier_2"]]
    pg = pg.merge(tiers, on="venue_id", how="left", validate="many_to_one")
    if pg["tier_3"].isna().any():
        raise ValueError("Some 2026 games are at venues without a tier; rerun src.park_factors")
    pg["hbb"] = pg["h"] + pg["bb"]
    pg["side"] = np.where(pg["is_home"], "Home", "Road")
    return pg


def league_tiers(pg: pd.DataFrame) -> pd.DataFrame:
    """League IP share and WHIP by scheme x tier x side (All / Home / Road)."""
    rows = []
    total_outs = pg["outs"].sum()
    league_whip = 3 * pg["hbb"].sum() / total_outs
    for scheme, (col, tiers) in SCHEMES.items():
        for tier in tiers:
            sub = pg[pg[col] == tier]
            for side, s in [("All", sub), ("Home", sub[sub["is_home"]]),
                            ("Road", sub[~sub["is_home"]])]:
                w = 3 * s["hbb"].sum() / s["outs"].sum()
                rows.append({"scheme": scheme, "tier": tier, "side": side,
                             "outs": int(s["outs"].sum()), "ip": s["outs"].sum() / 3,
                             "h": int(s["h"].sum()), "bb": int(s["bb"].sum()),
                             "ip_share": s["outs"].sum() / total_outs, "whip": w,
                             "rel_to_league": w / league_whip})
    return pd.DataFrame(rows)


def tier_matrices(pg: pd.DataFrame, col: str, tiers: list[str]):
    """Pitcher x tier matrices of H+BB and outs (zeros where a pitcher has no innings)."""
    hbb = pg.pivot_table(index="pitcher", columns=col, values="hbb", aggfunc="sum",
                         fill_value=0).reindex(columns=tiers, fill_value=0)
    outs = pg.pivot_table(index="pitcher", columns=col, values="outs", aggfunc="sum",
                          fill_value=0).reindex(columns=tiers, fill_value=0)
    return hbb.astype(float), outs.astype(float)


# --------------------------------------------------------------------------------------
# Standardization
# --------------------------------------------------------------------------------------
def standardize(hbb: pd.DataFrame, outs: pd.DataFrame, share: pd.Series, lwhip: pd.Series,
                league_whip: float, prior: str, prior_ip: float) -> pd.DataFrame:
    """Per-pitcher standardized WHIP for one tier scheme.

    hbb, outs: pitcher x tier; share: league IP share S_t; lwhip: league tier WHIP L_t.
    prior: "league" (centre L_t) or "pitcher" (centre theta_i * r_t).
    """
    tiers = list(hbb.columns)
    S = share.reindex(tiers).to_numpy()
    L = lwhip.reindex(tiers).to_numpy()
    r = L / league_whip
    ip = outs.to_numpy() / 3.0
    H = hbb.to_numpy()
    ip_tot = ip.sum(1)
    overall = H.sum(1) / ip_tot
    s = ip / ip_tot[:, None]                              # pitcher's own IP shares
    exposure = (s * r).sum(1)                             # 1.0 = league-average park mix
    theta = overall / exposure                            # exposure-only adjustment
    if prior == "league":
        prior_t = np.broadcast_to(L, ip.shape)
    elif prior == "pitcher":
        prior_t = theta[:, None] * r[None, :]
    else:
        raise ValueError(prior)
    shrunk = (H + prior_ip * prior_t) / (ip + prior_ip)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(ip > 0, H / np.where(ip > 0, ip, 1), np.nan)
    std = (shrunk * S).sum(1)
    own_mix = (shrunk * s).sum(1)
    std_raw = np.where((ip > 0).all(1), (np.nan_to_num(raw) * S).sum(1), np.nan)
    out = pd.DataFrame({
        "exposure_index": exposure, "exposure_adj_whip": theta,
        "exposure_gap": overall - theta, "std_whip_raw": std_raw, "raw_gap": overall - std_raw,
        "std_whip": std, "whip_gap": overall - std, "own_mix_whip": own_mix,
        "gap_shrinkage_part": overall - own_mix, "gap_park_part": own_mix - std,
    }, index=hbb.index)
    for j, t in enumerate(tiers):
        a = ABBR[t]
        out[f"ip_{a}"] = ip[:, j]
        out[f"share_{a}"] = s[:, j]
        out[f"whip_raw_{a}"] = raw[:, j]
        out[f"whip_shrunk_{a}"] = shrunk[:, j]
    return out


PRIOR_INDEPENDENT = ["exposure_index", "exposure_adj_whip", "exposure_gap", "std_whip_raw",
                     "raw_gap"]


def eb_diagnostics(pg: pd.DataFrame, league: pd.DataFrame, min_ip_list=(40, 80)) -> pd.DataFrame:
    """Method-of-moments empirical-Bayes prior weights (in IP) for tier WHIP.

    Sampling noise: Var(WHIP over IP innings) = sigma2 / IP with sigma2 = 3 * phi, where phi is
    the per-out variance of baserunners around each pitcher's season rate (games as units,
    degrees-of-freedom corrected).

    * Pitcher-centred prior: fit theta_i * r_t by weighted least squares in each pitcher's tiers;
      the residual sum of squares in excess of (T - 1) * sigma2 estimates the pitcher-by-tier
      interaction variance tau2_int; k = sigma2 / tau2_int.
    * League-centred prior: squared deviations from L_t in excess of noise estimate the talent
      variance tau2_talent; k = sigma2 / tau2_talent.
    """
    tot = pg.groupby("pitcher")[["hbb", "outs"]].sum()
    rate = tot["hbb"] / tot["outs"].replace(0, np.nan)
    g = pg.join(rate.rename("rate"), on="pitcher").dropna(subset=["rate"])
    phi = ((g["hbb"] - g["rate"] * g["outs"]) ** 2).sum() / g["outs"].sum()
    phi *= len(g) / (len(g) - g["pitcher"].nunique())
    sigma2 = 3 * phi
    league_whip = 3 * pg["hbb"].sum() / pg["outs"].sum()
    rows = []
    for scheme, (col, tiers) in SCHEMES.items():
        hbb, outs = tier_matrices(pg, col, tiers)
        lw = (league[(league["scheme"] == scheme) & (league["side"] == "All")]
              .set_index("tier")["whip"].reindex(tiers).to_numpy())
        rho = lw / league_whip
        for min_ip in min_ip_list:
            n_all = outs.to_numpy() / 3
            keep = n_all.sum(1) >= min_ip
            n, H = n_all[keep], hbb.to_numpy()[keep]
            raw = np.where(n > 0, H / np.where(n > 0, n, 1), 0.0)
            N = n.sum(1)
            t_eff = (n > 0).sum(1)
            theta = (n * rho * raw).sum(1) / (n * rho ** 2).sum(1)
            rss = (n * (raw - theta[:, None] * rho) ** 2).sum(1)
            tau2_int = (rss - (t_eff - 1) * sigma2).sum() / (N - (n ** 2).sum(1) / N).sum()
            rss_l = (n * (raw - rho * league_whip) ** 2).sum(1)
            tau2_tal = (rss_l - t_eff * sigma2).sum() / N.sum()
            rows.append({
                "scheme": scheme, "min_ip": min_ip, "n_pitchers": int(keep.sum()),
                "noise_var_whip_per_ip": sigma2,
                "tau2_interaction": tau2_int,
                "k_eb_pitcher_centred_ip": sigma2 / tau2_int if tau2_int > 0 else np.inf,
                "tau2_talent": tau2_tal,
                "k_eb_league_centred_ip": sigma2 / tau2_tal if tau2_tal > 0 else np.inf,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def build_table(pg: pd.DataFrame, league: pd.DataFrame, variants=VARIANTS) -> pd.DataFrame:
    """Wide per-pitcher table: totals, role, and every scheme x variant column."""
    total = (pg.groupby("pitcher")
             .agg(g=("game_pk", "nunique"), gs=("started", "sum"), outs=("outs", "sum"),
                  h=("h", "sum"), bb=("bb", "sum"), ibb=("ibb", "sum"))
             .reset_index())
    total["ip"] = total["outs"] / 3
    total["ip_display"] = total["outs"].map(ip_display)
    total["whip"] = 3 * (total["h"] + total["bb"]) / total["outs"].replace(0, np.nan)
    total["gs_share"] = total["gs"] / total["g"]
    total["role"] = np.where(total["gs_share"] >= STARTER_SHARE, "SP", "RP")
    total["qualifies"] = total["ip"] >= total["role"].map(MIN_IP)

    # Team label: team with the most outs; `teams` lists all teams for traded pitchers.
    abbr = mlb_api.team_abbrevs(ANALYSIS_SEASON)
    team_outs = (pg.groupby(["pitcher", "team_id"])["outs"].sum().reset_index()
                 .sort_values("outs", ascending=False))
    main_team = team_outs.drop_duplicates("pitcher").set_index("pitcher")["team_id"]
    total["team"] = total["pitcher"].map(main_team).map(abbr)
    total["teams"] = total["pitcher"].map(team_outs.groupby("pitcher")["team_id"].apply(
        lambda s: "/".join(abbr.get(t, str(t)) for t in s)))

    league_whip = 3 * pg["hbb"].sum() / pg["outs"].sum()
    out = total.set_index("pitcher")
    for scheme, (col, tiers) in SCHEMES.items():
        hbb, outs = tier_matrices(pg, col, tiers)
        lg = league[(league["scheme"] == scheme) & (league["side"] == "All")].set_index("tier")
        for name, (prior, k) in variants.items():
            st = standardize(hbb, outs, lg["ip_share"], lg["whip"], league_whip, prior, k)
            if name != "main":
                st = st.drop(columns=PRIOR_INDEPENDENT + [c for c in st.columns if c.startswith(
                    ("ip_", "share_", "whip_raw_"))])
            st.columns = [f"{c}__{scheme}_{name}" for c in st.columns]
            out = out.join(st)
        # Share of each tier's innings pitched at home (home/road context).
        side = pg.pivot_table(index="pitcher", columns=[col, "side"], values="outs",
                              aggfunc="sum", fill_value=0)
        for t in tiers:
            home = side[(t, "Home")] if (t, "Home") in side.columns else 0
            road = side[(t, "Road")] if (t, "Road") in side.columns else 0
            out[f"home_share_{ABBR[t]}__{scheme}"] = home / (home + road).replace(0, np.nan)
    return out.reset_index()


def headline_columns(tab: pd.DataFrame) -> pd.DataFrame:
    """Role-appropriate headline columns (3-tier for SP, 2-tier for RP) and ranks."""
    t = tab.copy()
    t["scheme"] = t["role"].map(ROLE_SCHEME)
    is3 = t["scheme"] == "3tier"

    for c in ["std_whip", "whip_gap"] + PRIOR_INDEPENDENT:
        t[c] = np.where(is3, t[f"{c}__3tier_main"], t[f"{c}__2tier_main"])
    for v in ("spec", "own20"):
        for c in ["std_whip", "whip_gap"]:
            t[f"{c}_{v}"] = np.where(is3, t[f"{c}__3tier_{v}"], t[f"{c}__2tier_{v}"])
    t["gap_shrinkage_part_spec"] = np.where(is3, t["gap_shrinkage_part__3tier_spec"],
                                            t["gap_shrinkage_part__2tier_spec"])
    t["gap_park_part_spec"] = np.where(is3, t["gap_park_part__3tier_spec"],
                                       t["gap_park_part__2tier_spec"])
    # Hitter-friendly exposure is identical in both schemes.
    t["ip_hf"] = t["ip_hf__3tier_main"]
    t["share_hf"] = t["share_hf__3tier_main"]
    t["home_share_hf"] = t["home_share_hf__3tier"]
    t["verdict"] = np.where(t["whip_gap"] > 0, "underrated", "overrated")

    # Ranks among qualifying pitchers of the same role (1 = lowest WHIP).
    for c in ["rank_overall", "rank_std", "rank_change"]:
        t[c] = np.nan
    for role in ("SP", "RP"):
        m = t["qualifies"] & (t["role"] == role)
        t.loc[m, "rank_overall"] = t.loc[m, "whip"].rank(method="min")
        t.loc[m, "rank_std"] = t.loc[m, "std_whip"].rank(method="min")
    t["rank_change"] = t["rank_overall"] - t["rank_std"]     # > 0: moves up when standardized
    return t


def split_table(pg: pd.DataFrame, league: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Long table: pitcher x scheme x tier x side (All/Home/Road) with raw and shrunk WHIP."""
    rows = []
    for scheme, (col, tiers) in SCHEMES.items():
        g = (pg.groupby(["pitcher", col, "side"])[["h", "bb", "outs"]].sum().reset_index()
             .rename(columns={col: "tier"}))
        a = g.groupby(["pitcher", "tier"])[["h", "bb", "outs"]].sum().reset_index()
        a["side"] = "All"
        g = pd.concat([a, g], ignore_index=True)
        g["scheme"] = scheme
        rows.append(g)
    sp = pd.concat(rows, ignore_index=True)
    sp["ip"] = sp["outs"] / 3
    sp["whip"] = 3 * (sp["h"] + sp["bb"]) / sp["outs"].replace(0, np.nan)
    sp.insert(1, "name", sp["pitcher"].map(names))
    lg = league.set_index(["scheme", "tier", "side"])["whip"]
    sp["league_whip"] = [lg.get((s, t, d), np.nan) for s, t, d in
                         zip(sp["scheme"], sp["tier"], sp["side"])]
    return sp[["pitcher", "name", "scheme", "tier", "side", "h", "bb", "outs", "ip", "whip",
               "league_whip"]].sort_values(["pitcher", "scheme", "tier", "side"])


def main():
    pg = load_games_with_tiers()
    league = league_tiers(pg)
    league.to_csv(OUT_TABLES / "league_tiers.csv", index=False, float_format="%.4f")
    league_whip = 3 * pg["hbb"].sum() / pg["outs"].sum()

    diag = eb_diagnostics(pg, league)
    diag.to_csv(OUT_TABLES / "shrinkage_diagnostics.csv", index=False, float_format="%.5f")

    tab = headline_columns(build_table(pg, league))
    names = player_names(tab["pitcher"], logger=log)
    tab.insert(1, "name", tab["pitcher"].map(names))
    split_table(pg, league, names).to_csv(OUT_TABLES / "pitcher_tier_splits.csv", index=False,
                                          float_format="%.4f")

    first = ["pitcher", "name", "team", "teams", "role", "qualifies", "g", "gs", "gs_share",
             "ip", "ip_display", "h", "bb", "ibb", "whip", "scheme", "ip_hf", "share_hf",
             "home_share_hf", "exposure_index", "std_whip", "whip_gap", "verdict",
             "rank_overall", "rank_std", "rank_change", "exposure_adj_whip", "exposure_gap",
             "std_whip_own20", "whip_gap_own20", "std_whip_spec", "whip_gap_spec",
             "gap_shrinkage_part_spec", "gap_park_part_spec", "std_whip_raw", "raw_gap"]
    rest = sorted(c for c in tab.columns if c not in first)
    tab = tab[first + rest].sort_values(["qualifies", "role", "whip_gap"],
                                         ascending=[False, False, False])
    tab.to_csv(OUT_TABLES / "pitcher_whip.csv", index=False, float_format="%.4f")

    # ---- Summary -------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Steps 4-5 summary (%d)", ANALYSIS_SEASON)
    log.info("League WHIP %.3f. Tier WHIP and IP share (all / home pitchers / road pitchers):",
             league_whip)
    for scheme in SCHEMES:
        for t in SCHEMES[scheme][1][::-1]:
            x = league[(league["scheme"] == scheme) & (league["tier"] == t)].set_index("side")
            log.info("  %s %-16s share %.3f  WHIP %.3f (home %.3f, road %.3f)", scheme, t,
                     x.loc["All", "ip_share"], x.loc["All", "whip"], x.loc["Home", "whip"],
                     x.loc["Road", "whip"])
    log.info("Empirical-Bayes prior weights (noise variance of WHIP = %.2f / IP):",
             diag["noise_var_whip_per_ip"].iloc[0])
    for d in diag.itertuples():
        log.info("  %s, pitchers with %d+ IP (n=%d): pitcher-centred k = %s IP "
                 "(interaction SD %.3f); league-centred k = %.0f IP (talent SD %.3f)", d.scheme,
                 d.min_ip, d.n_pitchers, "inf" if np.isinf(d.k_eb_pitcher_centred_ip)
                 else f"{d.k_eb_pitcher_centred_ip:.0f}", np.sqrt(max(d.tau2_interaction, 0)),
                 d.k_eb_league_centred_ip, np.sqrt(max(d.tau2_talent, 0)))
    log.info("Headline prior: pitcher-centred, k = %.0f IP", PRIOR_IP)

    q = tab[tab["qualifies"]]
    log.info("Pitchers: %d total; %d SP (>=%.0f%% of appearances as a starter), %d RP", len(tab),
             int((tab["role"] == "SP").sum()), 100 * STARTER_SHARE,
             int((tab["role"] == "RP").sum()))
    log.info("Qualifying: %d starters (80+ IP, 3-tier), %d relievers (40+ IP, 2-tier)",
             int((q["role"] == "SP").sum()), int((q["role"] == "RP").sum()))
    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    cols = ["name", "team", "ip_display", "whip", "std_whip", "whip_gap", "share_hf",
            "rank_overall", "rank_std"]
    for role in ("SP", "RP"):
        qr = q[q["role"] == role]
        log.info("%s (n=%d): corr(overall WHIP, gap) headline %.2f | own-level 20 IP %.2f | "
                 "spec %.2f; corr(headline gap, exposure-only gap) %.2f; headline gap SD %.3f, "
                 "range %+.3f..%+.3f", role, len(qr),
                 np.corrcoef(qr["whip"], qr["whip_gap"])[0, 1],
                 np.corrcoef(qr["whip"], qr["whip_gap_own20"])[0, 1],
                 np.corrcoef(qr["whip"], qr["whip_gap_spec"])[0, 1],
                 np.corrcoef(qr["whip_gap"], qr["exposure_gap"])[0, 1],
                 qr["whip_gap"].std(), qr["whip_gap"].min(), qr["whip_gap"].max())
        log.info("%s most underrated:\n%s", role,
                 qr.nlargest(10, "whip_gap")[cols].to_string(index=False, float_format=fmt))
        log.info("%s most overrated:\n%s", role,
                 qr.nsmallest(10, "whip_gap")[cols].to_string(index=False, float_format=fmt))
    log.info("Saved pitcher_whip.csv, pitcher_tier_splits.csv, league_tiers.csv, "
             "shrinkage_diagnostics.csv")


if __name__ == "__main__":
    main()
