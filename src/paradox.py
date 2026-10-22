"""Step 6: Simpson's-paradox pairs.

Among qualifying 2026 pitchers of the same role (starters with starters, relievers with
relievers), find ordered pairs (A, B) such that

    raw WHIP_A,t < raw WHIP_B,t  in every park tier t, and
    overall WHIP_A > overall WHIP_B,

with at least MIN_TIER_IP innings in every tier for both pitchers. Raw (observed) tier WHIPs
are used: Simpson's paradox is a statement about observed rates, and with raw rates the overall
WHIP is exactly the innings-weighted average of the tier WHIPs.

Robustness: each pitcher's games are resampled with replacement (N_BOOT times, independently
for each pitcher). A pair's robustness is the share of resamples in which the full paradox
(every tier inequality and the overall reversal) still holds; a resample in which either
pitcher has no innings in some tier counts as "does not hold". Pairs below 50% are flagged as
likely noise.

Both the 3-tier (primary) and 2-tier versions are run for both roles.

Output: outputs/tables/paradox_pairs.csv
Usage: python -m src.paradox
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import get_logger, ip_display
from .config import OUT_TABLES
from .park_factors import TIERS_2, TIERS_3
from .pitcher_stats import load_games_with_tiers

log = get_logger("paradox", "step6_paradox.log")

MIN_TIER_IP = 10.0
N_BOOT = 2000
N_PERM = 200
ROBUST_SHARE = 0.5
SEED = 2026
SCHEMES = {"3tier": ("tier_3", TIERS_3), "2tier": ("tier_2", TIERS_2)}
ABBR = {"Pitcher-friendly": "pf", "Neutral": "neu", "Hitter-friendly": "hf", "Other": "oth"}


def tier_arrays(pg: pd.DataFrame, pitchers, col: str, tiers: list[str]):
    """Observed per-tier (hbb, outs) arrays, shape (n_pitchers, n_tiers)."""
    sub = pg[pg["pitcher"].isin(pitchers)]
    hbb = (sub.pivot_table(index="pitcher", columns=col, values="hbb", aggfunc="sum", fill_value=0)
           .reindex(index=pitchers, columns=tiers, fill_value=0).to_numpy(float))
    outs = (sub.pivot_table(index="pitcher", columns=col, values="outs", aggfunc="sum",
                            fill_value=0)
            .reindex(index=pitchers, columns=tiers, fill_value=0).to_numpy(float))
    return hbb, outs


def find_pairs(hbb: np.ndarray, outs: np.ndarray, min_tier_ip: float = MIN_TIER_IP):
    """Indices (a, b) of all ordered paradox pairs among the rows of hbb/outs."""
    ip = outs / 3
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(outs > 0, 3 * hbb / outs, np.nan)
        overall = 3 * hbb.sum(1) / outs.sum(1)
    eligible = (ip >= min_tier_ip).all(1)
    idx = np.flatnonzero(eligible)
    pairs = []
    for a in idx:
        better_everywhere = (w[a][None, :] < w[idx]).all(1)
        worse_overall = overall[a] > overall[idx]
        for b in idx[better_everywhere & worse_overall]:
            pairs.append((a, b))
    return pairs, w, overall, eligible


def bootstrap_tiers(pg: pd.DataFrame, pitchers, col: str, tiers: list[str], n_boot: int,
                    rng: np.random.Generator):
    """Resample each pitcher's games; returns (hbb, outs) arrays of shape (P, n_boot, T)."""
    tier_index = {t: i for i, t in enumerate(tiers)}
    P, T = len(pitchers), len(tiers)
    hbb_b = np.zeros((P, n_boot, T))
    outs_b = np.zeros((P, n_boot, T))
    groups = pg[pg["pitcher"].isin(pitchers)].groupby("pitcher")
    for i, pid in enumerate(pitchers):
        g = groups.get_group(pid)
        n = len(g)
        onehot = np.zeros((n, T))
        onehot[np.arange(n), g[col].map(tier_index).to_numpy()] = 1
        counts = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)        # (n_boot, n)
        hbb_b[i] = counts @ (onehot * g["hbb"].to_numpy()[:, None])
        outs_b[i] = counts @ (onehot * g["outs"].to_numpy()[:, None])
    return hbb_b, outs_b


def paradox_share(hbb_b, outs_b, a: int, b: int) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        wa = np.where(outs_b[a] > 0, hbb_b[a] / outs_b[a], np.nan)
        wb = np.where(outs_b[b] > 0, hbb_b[b] / outs_b[b], np.nan)
        oa = hbb_b[a].sum(1) / outs_b[a].sum(1)
        ob = hbb_b[b].sum(1) / outs_b[b].sum(1)
    holds = (wa < wb).all(1) & (oa > ob)          # NaN comparisons are False
    return float(holds.mean())


def null_pair_counts(pg: pd.DataFrame, pitchers, col: str, tiers: list[str], park_mix,
                     rng, n_perm: int = N_PERM) -> tuple[np.ndarray, np.ndarray]:
    """Paradox-pair counts when park tiers carry no information.

    Within each pitcher, the tier labels of his games are shuffled: his number of games in
    each tier (his exposure) and his overall WHIP are unchanged, but any link between park
    tier and results is broken. Pairs found under this null are Simpson reversals produced
    by game-to-game noise alone. Returns the pair counts and, per permutation, the share of
    pairs in which A has the tougher park mix.
    """
    sub = pg[pg["pitcher"].isin(pitchers)][["pitcher", col, "hbb", "outs"]]
    tier_index = {t: i for i, t in enumerate(tiers)}
    groups = [(g["hbb"].to_numpy(float), g["outs"].to_numpy(float),
               g[col].map(tier_index).to_numpy()) for _, g in sub.groupby("pitcher", sort=True)]
    park_mix = np.asarray(park_mix, float)
    counts = np.zeros(n_perm, dtype=int)
    mix_share = np.full(n_perm, np.nan)
    T = len(tiers)
    for k in range(n_perm):
        hbb = np.zeros((len(groups), T))
        outs = np.zeros((len(groups), T))
        for i, (h, o, t) in enumerate(groups):
            tp = rng.permutation(t)
            np.add.at(hbb[i], tp, h)
            np.add.at(outs[i], tp, o)
        pairs = find_pairs(hbb, outs)[0]
        counts[k] = len(pairs)
        if pairs:
            mix_share[k] = np.mean([park_mix[a] > park_mix[b] for a, b in pairs])
    return counts, mix_share


def run(pg: pd.DataFrame, pw: pd.DataFrame, scheme: str, role: str, rng_boot,
        rng_null) -> pd.DataFrame:
    col, tiers = SCHEMES[scheme]
    q = pw[(pw["qualifies"]) & (pw["role"] == role)].sort_values("pitcher")
    pitchers = q["pitcher"].tolist()
    hbb, outs = tier_arrays(pg, pitchers, col, tiers)
    pairs, w, overall, eligible = find_pairs(hbb, outs)
    log.info("%s %s: %d qualifying, %d with >= %.0f IP in every tier, %d paradox pairs", scheme,
             role, len(pitchers), int(eligible.sum()), MIN_TIER_IP, len(pairs))
    park_mix = q[f"exposure_index__{scheme}_main"].to_numpy()
    null, null_mix = null_pair_counts(pg, pitchers, col, tiers, park_mix, rng_null)
    obs_mix = np.mean([park_mix[a] > park_mix[b] for a, b in pairs]) if pairs else np.nan
    NULL_SUMMARY.append({
        "scheme": scheme, "role": role, "observed_pairs": len(pairs),
        "null_mean": null.mean(), "null_p2_5": np.percentile(null, 2.5),
        "null_p97_5": np.percentile(null, 97.5),
        "share_null_ge_observed": float((null >= len(pairs)).mean()),
        "observed_park_mix_share": obs_mix, "null_park_mix_share_mean": np.nanmean(null_mix),
        "null_park_mix_share_p97_5": np.nanpercentile(null_mix, 97.5),
        "share_null_mix_ge_observed": float(np.nanmean(null_mix >= obs_mix)),
    })
    log.info("  noise-only benchmark (tier labels shuffled within pitcher, %d permutations): "
             "mean %.1f pairs, 95%% range %.0f-%.0f; observed %d. Share of pairs where A has "
             "the tougher park mix: observed %.0f%%, noise-only mean %.0f%% (97.5th pct %.0f%%)",
             N_PERM, null.mean(), np.percentile(null, 2.5), np.percentile(null, 97.5),
             len(pairs), 100 * obs_mix, 100 * np.nanmean(null_mix),
             100 * np.nanpercentile(null_mix, 97.5))
    if not pairs:
        return pd.DataFrame()
    involved = sorted({i for p in pairs for i in p})
    sub_ids = [pitchers[i] for i in involved]
    hbb_b, outs_b = bootstrap_tiers(pg, sub_ids, col, tiers, N_BOOT, rng_boot)
    pos = {i: k for k, i in enumerate(involved)}
    info = q.set_index("pitcher")
    rows = []
    for a, b in pairs:
        pa, pb = pitchers[a], pitchers[b]
        rec = {"scheme": scheme, "role": role,
               "pitcher_a": pa, "name_a": info.loc[pa, "name"], "team_a": info.loc[pa, "team"],
               "pitcher_b": pb, "name_b": info.loc[pb, "name"], "team_b": info.loc[pb, "team"],
               "overall_whip_a": overall[a], "overall_whip_b": overall[b],
               "ip_a": outs[a].sum() / 3, "ip_b": outs[b].sum() / 3,
               "ip_display_a": ip_display(outs[a].sum()), "ip_display_b": ip_display(outs[b].sum())}
        for j, t in enumerate(tiers):
            k = ABBR[t]
            rec[f"whip_{k}_a"] = w[a, j]
            rec[f"whip_{k}_b"] = w[b, j]
            rec[f"ip_{k}_a"] = outs[a, j] / 3
            rec[f"ip_{k}_b"] = outs[b, j] / 3
            rec[f"share_{k}_a"] = outs[a, j] / outs[a].sum()
            rec[f"share_{k}_b"] = outs[b, j] / outs[b].sum()
        rec["overall_gap"] = overall[a] - overall[b]
        rec["min_tier_margin"] = float(np.min(w[b] - w[a]))
        rec["hf_share_diff"] = rec["share_hf_a"] - rec["share_hf_b"]
        # Park mix: how much each pitcher's tier exposure inflates WHIP relative to the league
        # mix (league tier effects only). If A's mix is tougher, the reversal is the park-mix
        # paradox the project is about; otherwise it comes from the pitchers' own splits.
        rec["park_mix_a"] = info.loc[pa, f"exposure_index__{scheme}_main"]
        rec["park_mix_b"] = info.loc[pb, f"exposure_index__{scheme}_main"]
        rec["mechanism"] = "park mix" if rec["park_mix_a"] > rec["park_mix_b"] else "own splits"
        rec["bootstrap_share"] = paradox_share(hbb_b, outs_b, pos[a], pos[b])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["robust"] = out["bootstrap_share"] >= ROBUST_SHARE
    return out.sort_values("bootstrap_share", ascending=False).reset_index(drop=True)


NULL_SUMMARY: list[dict] = []


def main():
    # Separate random streams so the bootstrap does not depend on the benchmark.
    rng_boot = np.random.default_rng(SEED)
    rng_null = np.random.default_rng(SEED + 1)
    pg = load_games_with_tiers()
    pw = pd.read_csv(OUT_TABLES / "pitcher_whip.csv")
    results = []
    log.info("=" * 70)
    log.info("Step 6: paradox pairs (raw tier WHIP; >= %.0f IP per tier; %d bootstrap "
             "resamples of each pitcher's games)", MIN_TIER_IP, N_BOOT)
    for scheme in ("3tier", "2tier"):
        for role in ("SP", "RP"):
            res = run(pg, pw, scheme, role, rng_boot, rng_null)
            if len(res):
                log.info("  %s %s: %d pairs (%d park-mix, %d own-splits); %d robust (bootstrap "
                         "share >= %.0f%%); max share %.1f%%; median %.1f%%", scheme, role,
                         len(res), int((res["mechanism"] == "park mix").sum()),
                         int((res["mechanism"] == "own splits").sum()), int(res["robust"].sum()),
                         100 * ROBUST_SHARE, 100 * res["bootstrap_share"].max(),
                         100 * res["bootstrap_share"].median())
                results.append(res)
    allp = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    allp.to_csv(OUT_TABLES / "paradox_pairs.csv", index=False, float_format="%.4f")
    pd.DataFrame(NULL_SUMMARY).to_csv(OUT_TABLES / "paradox_null_benchmark.csv", index=False,
                                      float_format="%.3f")

    for scheme in ("3tier", "2tier"):
        sub = allp[allp["scheme"] == scheme] if len(allp) else allp
        if not len(sub):
            continue
        tiers = SCHEMES[scheme][1]
        cols = (["role", "name_a", "team_a", "name_b", "team_b", "overall_whip_a", "overall_whip_b"]
                + [f"whip_{ABBR[t]}_{s}" for t in tiers[::-1] for s in ("a", "b")]
                + ["park_mix_a", "park_mix_b", "mechanism", "bootstrap_share"])
        log.info("Top %s pairs:\n%s", scheme, sub.head(15)[cols].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
    log.info("Saved %s (%d pairs)", OUT_TABLES / "paradox_pairs.csv", len(allp))


if __name__ == "__main__":
    main()
