"""Player-name lookup: pybaseball.playerid_reverse_lookup, falling back to the MLB Stats API."""
from __future__ import annotations

import pandas as pd

from . import mlb_api
from .config import DATA_PROCESSED

NAMES_PATH = DATA_PROCESSED / "player_names.csv"


def player_names(ids, logger=None) -> dict[int, str]:
    ids = sorted({int(i) for i in ids})
    cached = pd.read_csv(NAMES_PATH) if NAMES_PATH.exists() else pd.DataFrame(
        columns=["mlbam", "name", "source"])
    have = set(cached["mlbam"].astype(int))
    need = [i for i in ids if i not in have]
    rows = []
    if need:
        found = {}
        try:
            from pybaseball import playerid_reverse_lookup

            lk = playerid_reverse_lookup(need, key_type="mlbam")
            for r in lk.itertuples():
                first = str(r.name_first).strip().title() if pd.notna(r.name_first) else ""
                last = str(r.name_last).strip().title() if pd.notna(r.name_last) else ""
                if first or last:
                    found[int(r.key_mlbam)] = (f"{first} {last}".strip(), "chadwick")
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning("playerid_reverse_lookup failed (%s); using MLB Stats API names", exc)
        # Chadwick title-cases names ("Mcclanahan"); prefer the Stats API's exact spelling,
        # and use it for anyone Chadwick does not know yet (2026 debuts).
        api = mlb_api.people_names(need)
        for i in need:
            if i in api and api[i]:
                rows.append({"mlbam": i, "name": api[i],
                             "source": "mlb_api" if i not in found else "mlb_api (chadwick match)"})
            elif i in found:
                rows.append({"mlbam": i, "name": found[i][0], "source": found[i][1]})
            else:
                rows.append({"mlbam": i, "name": f"MLBAM {i}", "source": "missing"})
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(NAMES_PATH, index=False)
    m = dict(zip(cached["mlbam"].astype(int), cached["name"]))
    return {i: m.get(i, f"MLBAM {i}") for i in ids}
