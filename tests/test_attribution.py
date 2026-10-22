"""Unit tests for out / hit / walk attribution and the WHIP helpers (synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from src.attribution import attribute, pitcher_game_table
from src.common import ip_display, shrunk_whip, whip

P1, P2 = 101, 202  # pitcher ids


def pitches(rows, game_pk=1):
    """rows: (inning, topbot, at_bat_number, pitch_number, outs_when_up, pitcher, events[, balls, strikes])."""
    recs = []
    for r in rows:
        inning, tb, ab, pn, owu, pitcher, ev = r[:7]
        balls, strikes = (r[7], r[8]) if len(r) > 7 else (0, 0)
        recs.append({"game_pk": game_pk, "inning": inning, "inning_topbot": tb,
                     "at_bat_number": ab, "pitch_number": pn, "outs_when_up": owu,
                     "pitcher": pitcher, "events": ev, "balls": balls, "strikes": strikes})
    return pd.DataFrame(recs)


def final_state(game_pk, inning, half, outs):
    return pd.DataFrame({"inning": [inning], "half": [half], "outs": [outs]},
                        index=pd.Index([game_pk], name="game_pk"))


def outs_by_pitcher(df, fs=None):
    a = attribute(df, fs)
    return a.groupby("pitcher")["outs"].sum().to_dict(), a


def test_three_up_three_down():
    df = pitches([
        (1, "Top", 1, 1, 0, P1, "field_out"),
        (1, "Top", 2, 1, 1, P1, "strikeout"),
        (1, "Top", 3, 1, 2, P1, "field_out"),
    ])
    got, a = outs_by_pitcher(df)
    assert got == {P1: 3}
    assert a["outs"].tolist() == [1, 1, 1]


def test_baserunning_out_between_pitches_is_counted():
    # Runner caught stealing during the 2nd PA (outs go 0 -> 1 mid-PA); not a PA event.
    df = pitches([
        (1, "Top", 1, 1, 0, P1, "single"),
        (1, "Top", 2, 1, 0, P1, None),        # caught stealing on this pitch
        (1, "Top", 2, 2, 1, P1, "field_out"),
        (1, "Top", 3, 1, 2, P1, "field_out"),
    ])
    got, a = outs_by_pitcher(df)
    assert got == {P1: 3}
    assert a["outs"].tolist() == [0, 1, 1, 1]


def test_inning_ending_caught_stealing_truncated_pa():
    # Third out is a caught stealing during an unfinished PA.
    df = pitches([
        (1, "Top", 1, 1, 0, P1, "field_out"),
        (1, "Top", 2, 1, 1, P1, "field_out"),
        (1, "Top", 3, 1, 2, P1, "single"),
        (1, "Top", 4, 1, 2, P1, "truncated_pa"),
    ])
    got, _ = outs_by_pitcher(df)
    assert got == {P1: 3}


def test_pitching_change_splits_outs():
    df = pitches([
        (1, "Top", 1, 1, 0, P1, "field_out"),
        (1, "Top", 2, 1, 1, P1, "double"),
        (1, "Top", 3, 1, 1, P2, "grounded_into_double_play"),
    ])
    got, _ = outs_by_pitcher(df)
    assert got == {P1: 1, P2: 2}


def test_walk_off_single_credits_no_out():
    # Bottom 9th: one out recorded, then a walk-off single. Game ends with 1 out.
    df = pitches([
        (9, "Top", 1, 1, 0, P1, "field_out"),
        (9, "Top", 2, 1, 1, P1, "field_out"),
        (9, "Top", 3, 1, 2, P1, "field_out"),
        (9, "Bot", 4, 1, 0, P2, "field_out"),
        (9, "Bot", 5, 1, 1, P2, "double"),
        (9, "Bot", 6, 1, 1, P2, "single"),
    ])
    got, a = outs_by_pitcher(df, final_state(1, 9, 1, 1))
    assert got == {P1: 3, P2: 1}
    # Without the official end state the walk-off pitcher would be wrongly credited 2 extra outs.
    got_naive, _ = outs_by_pitcher(df)
    assert got_naive[P2] == 3


def test_walk_off_sac_fly_credits_one_out():
    df = pitches([
        (10, "Bot", 1, 1, 0, P2, "walk"),
        (10, "Bot", 2, 1, 0, P2, "field_out"),
        (10, "Bot", 3, 1, 1, P2, "sac_fly"),   # 2nd out, winning run scores
    ])
    got, a = outs_by_pitcher(df, final_state(1, 10, 1, 2))
    assert a["outs"].tolist() == [0, 1, 1]
    assert got == {P2: 2}


def test_out_before_first_pitch_credited_to_first_pitcher():
    # Extra inning: the automatic runner is picked off before the first pitch.
    df = pitches([
        (10, "Top", 1, 1, 1, P1, "field_out"),
        (10, "Top", 2, 1, 2, P1, "strikeout"),
    ])
    got, _ = outs_by_pitcher(df)
    assert got == {P1: 3}


def test_game_called_before_last_half_inning_had_a_pitch():
    # Official final half-inning is the bottom of the 6th with 0 outs, but no pitch was thrown
    # in it; the top of the 6th ended normally with 3 outs.
    df = pitches([
        (6, "Top", 1, 1, 0, P1, "field_out"),
        (6, "Top", 2, 1, 1, P1, "field_out"),
        (6, "Top", 3, 1, 2, P1, "force_out"),
    ])
    got, _ = outs_by_pitcher(df, final_state(1, 6, 1, 0))
    assert got == {P1: 3}


def test_walk_charged_to_previous_pitcher_on_ball_heavy_count():
    # Reliever enters at 3-1 and walks the batter: walk charged to the previous pitcher.
    df = pitches([
        (1, "Top", 1, 1, 0, P1, None, 0, 0),
        (1, "Top", 1, 2, 0, P1, None, 1, 0),
        (1, "Top", 1, 3, 0, P1, None, 2, 0),
        (1, "Top", 1, 4, 0, P1, None, 3, 0),
        (1, "Top", 1, 5, 0, P2, None, 3, 1),   # change happened at 3-1
        (1, "Top", 1, 6, 0, P2, "walk", 3, 2),
    ])
    a = attribute(df)
    assert a.loc[a["events"] == "walk", "bb_pitcher"].item() == P1


def test_walk_charged_to_reliever_otherwise():
    df = pitches([
        (1, "Top", 1, 1, 0, P1, None, 0, 0),
        (1, "Top", 1, 2, 0, P2, None, 1, 0),   # change at 1-0
        (1, "Top", 1, 3, 0, P2, None, 2, 0),
        (1, "Top", 1, 4, 0, P2, None, 3, 0),
        (1, "Top", 1, 5, 0, P2, "walk", 3, 0),
    ])
    a = attribute(df)
    assert a.loc[a["events"] == "walk", "bb_pitcher"].item() == P2


def test_hits_and_walks_flags():
    df = pitches([
        (1, "Top", 1, 1, 0, P1, "single"),
        (1, "Top", 2, 1, 0, P1, "home_run"),
        (1, "Top", 3, 1, 0, P1, "intent_walk"),
        (1, "Top", 4, 1, 0, P1, "hit_by_pitch"),     # not a walk
        (1, "Top", 5, 1, 0, P1, "field_error"),      # not a hit
        (1, "Top", 6, 1, 0, P1, "grounded_into_double_play"),
        (1, "Top", 7, 1, 2, P1, "strikeout"),
    ])
    a = attribute(df)
    assert int(a["is_hit"].sum()) == 2
    assert int(a["is_walk"].sum()) == 1
    assert int(a["outs"].sum()) == 3


def test_pitcher_game_table_starters_sides_and_walk_chargeback():
    df = pitches([
        (1, "Top", 1, 1, 0, P1, "field_out"),
        (1, "Top", 2, 1, 1, P1, None, 0, 0),
        (1, "Top", 2, 2, 1, P1, None, 1, 0),
        (1, "Top", 2, 3, 1, P1, None, 2, 0),
        (1, "Top", 2, 4, 1, P2, "walk", 3, 0),     # change at 3-0 -> walk charged to P1
        (1, "Top", 3, 1, 1, P2, "grounded_into_double_play"),
        (1, "Bot", 4, 1, 0, 303, "strikeout"),
        (1, "Bot", 5, 1, 1, 303, "single"),
        (1, "Bot", 6, 1, 1, 303, "double_play"),
    ])
    games = pd.DataFrame([{"game_pk": 1, "season": 2026, "official_date": "2026-04-01",
                           "venue_id": 19, "venue_name": "Coors Field", "is_neutral": False,
                           "home_id": 115, "home_name": "Rockies", "away_id": 119,
                           "away_name": "Dodgers"}])
    pg = pitcher_game_table(attribute(df), games).set_index("pitcher")
    assert pg.loc[P1, "started"] and not pg.loc[P2, "started"] and pg.loc[303, "started"]
    assert pg.loc[P1, "team_id"] == 115 and pg.loc[P1, "is_home"]
    assert pg.loc[303, "team_id"] == 119 and not pg.loc[303, "is_home"]
    assert (pg.loc[P1, "outs"], pg.loc[P1, "bb"]) == (1, 1)
    assert (pg.loc[P2, "outs"], pg.loc[P2, "bb"]) == (2, 0)
    assert (pg.loc[303, "outs"], pg.loc[303, "h"]) == (3, 1)


def test_whip_and_ip_helpers():
    assert whip(10, 5, 45) == pytest.approx(1.0)            # 15 baserunners in 15 IP
    assert np.isnan(whip(1, 0, 0))
    assert ip_display(100) == "33.1"
    # 20 IP prior at 1.30 pulls a 2.00 WHIP over 10 IP two-thirds of the way to 1.30
    assert shrunk_whip(15, 5, 30, 1.30, 20) == pytest.approx((20 + 26) / 30)
