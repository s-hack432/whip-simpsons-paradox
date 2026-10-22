"""Step 7: figures (PNG) for the write-up.

fig1_paradox_3tier.png      paradox pairs, 3 park tiers (centerpiece)
fig1b_paradox_2tier.png     paradox pairs, 2 park tiers (robustness version)
fig2_exposure.png           featured pitchers' share of innings by park tier
fig3_overall_vs_std.png     overall vs park-standardized WHIP, all qualifying pitchers
fig4_park_factors.png       venues ranked by regressed park factor, colored by tier
fig5_rank_change.png        largest WHIP-rank moves from overall to standardized

Colors come from a validated colorblind-safe palette (see README): pitcher A / starters =
blue, pitcher B / relievers = red; park tiers use a diverging blue-gray-red scheme.

Text is set in Source Serif 4, an open-source serif bundled in assets/fonts/ (SIL Open Font
License) so the figures render the same anywhere.

Featured pairs = the highest-bootstrap-share pairs whose reversal is the park-mix kind
(the pitcher who looks worse overall pitched in the tougher park mix), each pitcher at most
once. All pairs, of both kinds, are in outputs/tables/paradox_pairs.csv.

Usage: python -m src.plots
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import mlb_api  # noqa: E402
from .common import get_logger  # noqa: E402
from .config import ANALYSIS_SEASON, OUT_FIGURES, OUT_TABLES, ROOT  # noqa: E402

log = get_logger("plots", "step7_plots.log")

# ---- Palette (light mode, validated with the dataviz palette validator) ---------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOT1 = "#2a78d6"      # blue: pitcher A / starters
SLOT2 = "#e34948"      # red: pitcher B / relievers
TIER_COLOR = {"Pitcher-friendly": "#2a78d6", "Neutral": "#898781", "Hitter-friendly": "#e34948"}
RANK_BEFORE = "#86b6ef"   # blue ramp step 250
RANK_AFTER = "#1c5cab"    # blue ramp step 550
DPI = 200
TIERS3 = ["Pitcher-friendly", "Neutral", "Hitter-friendly"]
ABBR = {"Pitcher-friendly": "pf", "Neutral": "neu", "Hitter-friendly": "hf", "Other": "oth"}
SCHEME_TIERS = {"3tier": (TIERS3, ["Pitcher-friendly", "Neutral", "Hitter-friendly"]),
                "2tier": (["Other", "Hitter-friendly"], ["Other parks", "Hitter-friendly"])}


FONT_DIR = ROOT / "assets" / "fonts"
FONT = "Source Serif 4"


def register_fonts():
    """Register the bundled font files so figures render the same on any machine."""
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))


def setup_style():
    register_fonts()
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": [FONT, "DejaVu Serif"], "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK,
    })


def _lum(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def text_on(fill: str) -> str:
    """White or ink text, whichever has more contrast on the fill."""
    lf = _lum(fill)
    return "#ffffff" if 1.05 / (lf + 0.05) >= (lf + 0.05) / (_lum(INK) + 0.05) else INK


def fig_header(fig, title: str, subtitle: str, top_in: float = 0.18):
    """Figure title + subtitle anchored a fixed distance (inches) below the top edge."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - top_in / h, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - (top_in + 0.36) / h, subtitle, fontsize=9.3, color=INK2, va="top",
             linespacing=1.4)


def panel_header(ax, title: str, subtitle: str | None = None, legend_row: bool = False):
    """Stack title, subtitle and (optionally) room for a legend row above the axes.

    Offsets are in points so they do not depend on the axes height. A legend placed with
    loc="lower left", bbox_to_anchor=(0, 1) fills the legend row.
    """
    sub_off = 20 if legend_row else 5
    pad = sub_off + (15 if subtitle else 0) + 3
    ax.set_title(title, loc="left", pad=pad)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, sub_off),
                    textcoords="offset points", fontsize=8.3, color=INK2, va="bottom", ha="left")


def fmt_whip(v: float, other: float) -> str:
    """2 decimals unless the two values would look tied, then 3."""
    return f"{v:.3f}" if round(v, 2) == round(other, 2) else f"{v:.2f}"


def last_name(name: str) -> str:
    parts = name.split()
    return " ".join(parts[1:]) if len(parts) > 1 else name


# --------------------------------------------------------------------------------------
# Featured pairs
# --------------------------------------------------------------------------------------
def featured_pairs(pairs: pd.DataFrame, scheme: str, role: str = "SP", k: int = 4) -> pd.DataFrame:
    """Top park-mix pairs by bootstrap share, each pitcher used at most once."""
    sub = pairs[(pairs["scheme"] == scheme) & (pairs["role"] == role)
                & (pairs["mechanism"] == "park mix")].sort_values("bootstrap_share",
                                                                  ascending=False)
    chosen, used = [], set()
    for _, r in sub.iterrows():
        if r["pitcher_a"] in used or r["pitcher_b"] in used:
            continue
        chosen.append(r)
        used |= {r["pitcher_a"], r["pitcher_b"]}
        if len(chosen) == k:
            break
    return pd.DataFrame(chosen).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Figure 1: paradox pairs
# --------------------------------------------------------------------------------------
def paradox_panel(ax, r: pd.Series, scheme: str):
    tiers, labels = SCHEME_TIERS[scheme]
    keys = [ABBR[t] for t in tiers]
    va = [r[f"whip_{k}_a"] for k in keys] + [r["overall_whip_a"]]
    vb = [r[f"whip_{k}_b"] for k in keys] + [r["overall_whip_b"]]
    x = np.arange(len(va), dtype=float)
    x[-1] += 0.35                                      # separate the Overall group
    w, gap = 0.34, 0.03
    ymax = max(max(va), max(vb))
    for side, xs, vals, color in (("a", x - w / 2 - gap / 2, va, SLOT1),
                                  ("b", x + w / 2 + gap / 2, vb, SLOT2)):
        other_vals = vb if side == "a" else va
        ax.bar(xs, vals, w, color=color, linewidth=0)
        for i, (xi, v) in enumerate(zip(xs, vals)):
            ax.text(xi, v + ymax * 0.015, fmt_whip(v, other_vals[i]), ha="center", va="bottom",
                    fontsize=8, color=INK, fontweight="bold" if i == len(vals) - 1 else "normal")
            if i < len(keys):
                share = r[f"share_{keys[i]}_{side}"]
                ip = r[f"ip_{keys[i]}_{side}"]
                ax.text(xi, ymax * 0.025, f"{share:.0%}\n{ip:.0f} IP", ha="center", va="bottom",
                        fontsize=6.8, color=text_on(color), linespacing=1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels + ["Overall"])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(0, ymax * 1.1)
    ax.set_ylabel("WHIP")
    ax.grid(axis="x", visible=False)
    ax.axvline(x[-2] + (x[-1] - x[-2]) / 2, color=GRID, linewidth=0.8)
    panel_header(ax, f"{r['name_a']} ({r['team_a']}) vs. {r['name_b']} ({r['team_b']})",
                 f"Paradox holds in {r['bootstrap_share']:.0%} of bootstrap resamples", True)
    handles = [Patch(color=SLOT1, label=f"{last_name(r['name_a'])}: lower WHIP in every tier"),
               Patch(color=SLOT2, label=f"{last_name(r['name_b'])}: lower WHIP overall")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              handlelength=1.0, borderaxespad=0.15, columnspacing=1.2)


def fig_paradox(pairs: pd.DataFrame, scheme: str, path, title: str, subtitle: str):
    feat = featured_pairs(pairs, scheme)
    if feat.empty:
        log.warning("No %s pairs to plot", scheme)
        return feat
    n = len(feat)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    height = 4.5 * nrows + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, height))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r) in zip(axes, feat.iterrows()):
        paradox_panel(ax, r, scheme)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig_header(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.15 / height), h_pad=3.0, w_pad=3)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return feat


# --------------------------------------------------------------------------------------
# Figure 2: exposure
# --------------------------------------------------------------------------------------
def fig_exposure(feat_sets, pw: pd.DataFrame, league: pd.DataFrame, path):
    info = pw.set_index("pitcher")
    lg = league[(league["scheme"] == "3tier") & (league["side"] == "All")].set_index("tier")
    rows = [("row", "League (all 2026 innings)", [lg.loc[t, "ip_share"] for t in TIERS3],
             "average park mix (x1.000)")]
    for scheme_label, scheme, feat in feat_sets:
        for _, r in feat.iterrows():
            rows.append(("header", f"{r['name_a']} vs. {r['name_b']} ({scheme_label}): paradox "
                         f"holds in {r['bootstrap_share']:.0%} of resamples", None, None))
            for side, role_txt in (("a", "lower WHIP in every tier"),
                                   ("b", "lower WHIP overall")):
                pid = r[f"pitcher_{side}"]
                shares = [info.loc[pid, f"share_{ABBR[t]}__3tier_main"] for t in TIERS3]
                rows.append(("row", f"{r[f'name_{side}']} ({r[f'team_{side}']})", shares,
                             f"park mix x{r[f'park_mix_{side}']:.3f} | {role_txt}"))
    n_rows = sum(1 for r in rows if r[0] == "row")
    n_head = sum(1 for r in rows if r[0] == "header")
    height = 0.36 * n_rows + 0.42 * n_head + 2.2
    fig, ax = plt.subplots(figsize=(11.5, height))
    y = 0.0
    yticks, ylabels = [], []
    for kind, name, shares, note in rows:
        if kind == "header":
            y += 0.35
            ax.text(0, y, name, fontsize=8.2, color=INK2, va="center", ha="left",
                    fontweight="bold")
            y += 0.75
            continue
        left = 0.0
        for t, s in zip(TIERS3, shares):
            c = TIER_COLOR[t]
            ax.barh(y, s, left=left, height=0.62, color=c, edgecolor=SURFACE, linewidth=2)
            if s >= 0.075:
                ax.text(left + s / 2, y, f"{s:.0%}", ha="center", va="center", fontsize=7.8,
                        color=text_on(c))
            left += s
        ax.text(1.015, y, note, va="center", fontsize=8, color=INK2)
        yticks.append(y)
        ylabels.append(name)
        y += 0.9
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(y - 0.2, -0.6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Share of the pitcher's 2026 innings, by park tier")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=TIER_COLOR[t], label=t) for t in TIERS3], ncol=3,
              loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.0, borderaxespad=0.3)
    fig_header(fig, "Why the paradox happens: the pitcher who looks worse overall worked in a "
                    "tougher park mix",
               "Park mix = how much a pitcher's blend of park tiers inflates WHIP relative to the "
               "league-average blend (x1.000), using league-wide tier WHIPs.\nThe pitcher with the "
               "higher park mix carries more innings in the high-WHIP tiers, which drags his overall "
               "WHIP up even though he was better in every tier.")
    fig.tight_layout(rect=(0, 0, 0.74, 1 - 1.25 / height))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 3: overall vs standardized
# --------------------------------------------------------------------------------------
def number_points(ax, marks, base_pt: float = 7.5, step_pt: float = 6.5, min_pt: float = 9.5):
    """Put a small number beside each marked point, pushed outward along the perpendicular
    to y = x (down-right for direction +1, up-left for -1) until it clears earlier numbers.

    marks: list of (number, x, y, direction). Call after the final layout is set.
    """
    fig = ax.figure
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72
    placed = []
    for num, x, y, direction in sorted(marks, key=lambda m: m[1]):
        xd, yd = ax.transData.transform((x, y))
        ux, uy = direction * 0.7071, -direction * 0.7071
        dist = base_pt
        while any(np.hypot(xd + ux * dist * px_per_pt - qx, yd + uy * dist * px_per_pt - qy)
                  < min_pt * px_per_pt for qx, qy in placed):
            dist += step_pt
        placed.append((xd + ux * dist * px_per_pt, yd + uy * dist * px_per_pt))
        ax.annotate(str(num), xy=(x, y), xytext=(ux * dist, uy * dist),
                    textcoords="offset points", ha="center", va="center", fontsize=7,
                    fontweight="bold", color=INK,
                    arrowprops=None if dist <= base_pt else dict(arrowstyle="-", color=AXIS,
                                                                  lw=0.6, shrinkA=4, shrinkB=3))


def number_key(ax, title: str, pts: pd.DataFrame, start: int, x: float, y: float, dy: float):
    """A numbered list of labeled pitchers, drawn in data coordinates."""
    ax.text(x, y, title, fontsize=8.2, color=INK2, fontweight="bold", va="top", ha="left")
    for i, (_, p) in enumerate(pts.iterrows()):
        ax.text(x, y - (i + 1.25) * dy, f"{start + i:>2}   {p['name']} ({p['team']})   "
                f"{p['whip_gap']:+.3f}", fontsize=7.8, color=INK, va="top", ha="left")


def fig_scatter(pw: pd.DataFrame, league: pd.DataFrame, path):
    q = pw[pw["qualifies"]].copy()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 7.2),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    lo = min(q["whip"].min(), q["std_whip"].min()) - 0.05
    hi = max(q["whip"].max(), q["std_whip"].max()) + 0.05
    span = hi - lo
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=1)
    for role, color in (("SP", SLOT1), ("RP", SLOT2)):
        s = q[q["role"] == role]
        for a, xcol, ycol in ((ax, "whip", "std_whip"), (bx, "share_hf", "whip_gap")):
            a.scatter(s[xcol], s[ycol], s=16, color=color, edgecolor=SURFACE, linewidth=0.8,
                      zorder=3)
    top = q.nlargest(10, "whip_gap").reset_index(drop=True)      # underrated
    bot = q.nsmallest(10, "whip_gap").reset_index(drop=True)     # overrated
    hl = pd.concat([top, bot])
    ax.scatter(hl["whip"], hl["std_whip"], s=36, facecolor="none", edgecolor=INK, lw=0.8,
               zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Overall WHIP (as pitched)")
    ax.set_ylabel("Park-standardized WHIP (league-average park mix)")
    ax.text(hi - 0.01 * span, hi - 0.015 * span, "y = x", ha="right", va="top", fontsize=8,
            color=MUTED)
    dy = 0.026 * span
    number_key(ax, "Overrated by overall WHIP (above the line)", bot, 11,
               lo + 0.02 * span, hi - 0.03 * span, dy)
    number_key(ax, "Underrated by overall WHIP (below the line)", top, 1,
               lo + 0.50 * span, lo + 0.36 * span, dy)
    panel_header(ax, "Standardization moves WHIP by at most about 0.05",
                 "Circled and numbered: the 10 largest gaps in each direction "
                 "(gap = overall - standardized).")

    league_hf = league[(league["scheme"] == "3tier") & (league["tier"] == "Hitter-friendly")
                       & (league["side"] == "All")]["ip_share"].iloc[0]
    bx.axhline(0, color=MUTED, lw=1, zorder=1)
    bx.axvline(league_hf, color=AXIS, lw=1, zorder=1)
    bx.text(league_hf + 0.01, q["whip_gap"].min(), f"league-wide share {league_hf:.0%}",
            fontsize=7.8, color=MUTED, va="bottom")
    bx.set_xlabel("Share of the pitcher's innings in hitter-friendly parks")
    bx.set_ylabel("WHIP gap (overall - standardized)")
    bx.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    panel_header(bx, "The gap is set by where they pitched",
                 "Positive gap = underrated. Two clusters: home park hitter-friendly or not.")
    handles = [Line2D([], [], marker="o", ls="", color=SLOT1, markeredgecolor=SURFACE, ms=6,
                      label=f"Starters ({(q['role'] == 'SP').sum()}; 80+ IP; 3 park tiers)"),
               Line2D([], [], marker="o", ls="", color=SLOT2, markeredgecolor=SURFACE, ms=6,
                      label=f"Relievers ({(q['role'] == 'RP').sum()}; 40+ IP; 2 park tiers)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.95 / 7.2),
               ncol=2, handletextpad=0.3, columnspacing=1.5)
    fig_header(fig, "Overall vs. park-standardized WHIP, 2026 qualifying pitchers",
               "Standardized WHIP re-weights each pitcher's (shrunken) tier WHIPs to the "
               "league-wide innings mix across park tiers.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.3 / 7.2), w_pad=3)
    # Numbers go on after the layout is final (their collision check uses display units).
    marks = ([(i + 1, p["whip"], p["std_whip"], +1) for i, p in top.iterrows()]
             + [(i + 11, p["whip"], p["std_whip"], -1) for i, p in bot.iterrows()])
    number_points(ax, marks)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 4: park factors
# --------------------------------------------------------------------------------------
def fig_park_factors(pf: pd.DataFrame, path):
    abbr = mlb_api.team_abbrevs(ANALYSIS_SEASON)
    d = pf[pf["tier_3"].isin(TIERS3) & pf["pf_regressed"].notna()].sort_values("pf_regressed")
    height = 9.8
    fig, ax = plt.subplots(figsize=(10.5, height))
    y = np.arange(len(d))
    colors = [TIER_COLOR[t] for t in d["tier_3"]]
    ax.barh(y, d["pf_regressed"] - 1, left=1, height=0.62, color=colors, linewidth=0)
    ax.errorbar(d["pf_regressed"], y, xerr=d["pf_regressed_se"], fmt="none", ecolor=INK2,
                elinewidth=0.8, capsize=0, zorder=4)
    for yi, (_, r) in zip(y, d.iterrows()):
        v = r["pf_regressed"]
        off = r["pf_regressed_se"] + 0.004
        ax.text(v + off if v >= 1 else v - off, yi, f"{v:.3f}", va="center",
                ha="left" if v >= 1 else "right", fontsize=7.6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.venue} ({abbr.get(int(r.home_team_id), '')})" for r in d.itertuples()])
    ax.axvline(1, color=AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    tiers = d["tier_3"].tolist()
    for i in range(1, len(tiers)):
        if tiers[i] != tiers[i - 1]:
            ax.axhline(i - 0.5, color=GRID, lw=1)
    xmin = d["pf_regressed"].min() - d["pf_regressed_se"].max() - 0.03
    xmax = d["pf_regressed"].max() + d["pf_regressed_se"].max() + 0.03
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Regressed park factor for (H + BB) per out   "
                  "(1.00 = average; 1.10 = 10% more baserunners per out)")
    share = d.groupby("tier_3")["ip_share_2026"].sum()
    ax.legend(handles=[Patch(color=TIER_COLOR[t], label=f"{t} ({share.get(t, 0):.0%} of 2026 "
                                                         "innings)") for t in TIERS3[::-1]],
              loc="lower right", handlelength=1.0)
    fig_header(fig, "Coors Field and Sutter Health Park are the most hitter-friendly parks for WHIP",
               "Rate of (H+BB) per out at the venue vs. the same team's road games, 2024-26 "
               "(home-park seasons only), regressed toward 1.0 by one season of outs.\nWhiskers: "
               "+/-1 bootstrap standard error. Bars colored by 2026 tier; each tier holds about a "
               "third of 2026 innings.")
    note = ("Not shown: neutral-site venues without an estimate, assigned to Neutral (Las Vegas "
            "Ballpark, Estadio Alfredo Harp Helu, Field of Dreams,\nJourney Bank Ballpark), and "
            "parks not used in 2026 (George M. Steinbrenner Field 2025: 1.023; Oakland Coliseum "
            "2024: 0.994).")
    fig.text(0.012, 0.012, note, fontsize=7.6, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.045, 1, 1 - 1.15 / height))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 5: rank change
# --------------------------------------------------------------------------------------
def fig_rank_change(pw: pd.DataFrame, path, k: int = 8):
    q = pw[pw["qualifies"]].copy()
    height = 7.4
    fig, axes = plt.subplots(1, 2, figsize=(13, height))
    for ax, role, label in ((axes[0], "SP", "Starters"), (axes[1], "RP", "Relievers")):
        s = q[q["role"] == role]
        n = len(s)
        d = pd.concat([s.nlargest(k, "rank_change"), s.nsmallest(k, "rank_change")])
        d = d.sort_values("rank_change", ascending=False)
        y = np.arange(len(d))
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.plot([r["rank_overall"], r["rank_std"]], [yi, yi], color=AXIS, lw=2,
                    solid_capstyle="round", zorder=2)
        ax.scatter(d["rank_overall"], y, s=46, color=RANK_BEFORE, edgecolor=SURFACE, lw=1.5,
                   zorder=3)
        ax.scatter(d["rank_std"], y, s=46, color=RANK_AFTER, edgecolor=SURFACE, lw=1.5, zorder=4)
        for yi, (_, r) in zip(y, d.iterrows()):
            ch = int(r["rank_change"])
            right = max(r["rank_overall"], r["rank_std"])
            ax.text(right + n * 0.02, yi, f"{'up' if ch > 0 else 'down'} {abs(ch)}  "
                    f"(gap {r['whip_gap']:+.3f})", va="center", fontsize=7.8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.name} ({r.team})" for r in d.itertuples()])
        ax.invert_yaxis()
        ax.set_xlim(0, n * 1.25)
        ax.set_xlabel(f"Rank among {n} qualifying {label.lower()} (1 = lowest WHIP)")
        ax.grid(axis="y", visible=False)
        ax.axhline(k - 0.5, color=GRID, lw=1)
        panel_header(ax, f"{label}: biggest movers",
                     f"Top {k} risers (upper block) and fallers (lower block).")
    handles = [Line2D([], [], marker="o", ls="", color=RANK_BEFORE, markeredgecolor=SURFACE, ms=7,
                      label="Rank by overall WHIP"),
               Line2D([], [], marker="o", ls="", color=RANK_AFTER, markeredgecolor=SURFACE, ms=7,
                      label="Rank by park-standardized WHIP")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 1 - 0.95 / height),
               ncol=2, handletextpad=0.3, columnspacing=1.5)
    mx_sp = int(q.loc[q["role"] == "SP", "rank_change"].abs().max())
    mx_rp = int(q.loc[q["role"] == "RP", "rank_change"].abs().max())
    fig_header(fig, f"Giving everyone the same park mix moves pitchers up to {mx_sp} places "
                    f"among starters and {mx_rp} among relievers",
               "Ranks within role among qualifying pitchers. Moves are largest in the crowded "
               "middle of the WHIP distribution, where a 0.03 shift passes many pitchers.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.3 / height), w_pad=4)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    setup_style()
    pairs = pd.read_csv(OUT_TABLES / "paradox_pairs.csv")
    pw = pd.read_csv(OUT_TABLES / "pitcher_whip.csv")
    pf = pd.read_csv(OUT_TABLES / "park_factors.csv")
    league = pd.read_csv(OUT_TABLES / "league_tiers.csv")

    best = pairs.groupby("scheme")["bootstrap_share"].max()
    f3 = fig_paradox(
        pairs, "3tier", OUT_FIGURES / "fig1_paradox_3tier.png",
        "Better in every park tier, worse overall: Simpson's paradox in 2026 WHIP",
        "Top starter pairs (3 park tiers; 10+ IP per tier for both) where the pitcher who looks "
        "worse overall had the tougher park mix. Numbers at the bar base: share of his innings "
        "in that tier, and innings.\n"
        f"None is robust: the best holds in {best['3tier']:.0%} of 2,000 bootstrap resamples "
        "(robust = 50%+), and noise alone produces about as many such pairs.")
    f2 = fig_paradox(
        pairs, "2tier", OUT_FIGURES / "fig1b_paradox_2tier.png",
        "Two park tiers (hitter-friendly vs. everything else): the same paradox, still fragile",
        "Top starter pairs (10+ IP per tier for both) where the pitcher who looks worse overall "
        "had the tougher park mix. Numbers at the bar base: share of his innings in that tier, "
        f"and innings.\nBest bootstrap share: {best['2tier']:.0%} (robust = 50%+).")
    fig_exposure([("3 tiers", "3tier", f3), ("2 tiers", "2tier", f2)], pw, league,
                 OUT_FIGURES / "fig2_exposure.png")
    fig_scatter(pw, league, OUT_FIGURES / "fig3_overall_vs_std.png")
    fig_park_factors(pf, OUT_FIGURES / "fig4_park_factors.png")
    fig_rank_change(pw, OUT_FIGURES / "fig5_rank_change.png")
    feats = pd.concat([f3.assign(figure="fig1"), f2.assign(figure="fig1b")], ignore_index=True)
    feats.to_csv(OUT_TABLES / "featured_pairs.csv", index=False, float_format="%.4f")
    log.info("Step 7: wrote %d figures to %s; featured pairs saved to featured_pairs.csv",
             len(list(OUT_FIGURES.glob("*.png"))), OUT_FIGURES)


if __name__ == "__main__":
    main()
