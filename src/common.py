"""Small shared helpers: logging, WHIP arithmetic, shrinkage."""
from __future__ import annotations

import logging
import sys

import numpy as np

from .config import OUT_LOGS


def get_logger(name: str, logfile: str | None = None) -> logging.Logger:
    """Logger that prints to stdout and, optionally, appends to outputs/logs/<logfile>."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        fh = logging.FileHandler(OUT_LOGS / logfile, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def ip_from_outs(outs):
    """Innings pitched as a true decimal (outs / 3)."""
    return np.asarray(outs, dtype=float) / 3.0


def ip_display(outs) -> str:
    """Baseball-style IP string, e.g. 100 outs -> '33.1'."""
    outs = int(outs)
    return f"{outs // 3}.{outs % 3}"


def whip(h, bb, outs):
    """WHIP = (H + BB) / IP, with IP = outs / 3. Returns NaN when outs == 0."""
    h = np.asarray(h, dtype=float)
    bb = np.asarray(bb, dtype=float)
    outs = np.asarray(outs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(outs > 0, 3.0 * (h + bb) / outs, np.nan)


def shrunk_whip(h, bb, outs, prior_whip, prior_ip):
    """Empirical-Bayes style shrinkage of WHIP toward `prior_whip`.

    Equivalent to adding `prior_ip` innings pitched at exactly `prior_whip`:
        (H + BB + prior_ip * prior_whip) / (IP + prior_ip)
    """
    h = np.asarray(h, dtype=float)
    bb = np.asarray(bb, dtype=float)
    ip = np.asarray(outs, dtype=float) / 3.0
    return (h + bb + prior_ip * np.asarray(prior_whip, dtype=float)) / (ip + prior_ip)
