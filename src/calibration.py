"""Calibration and prior shift — DISPLAY ONLY.

The decision uses the RAW score, exactly as in the validated MATLAB twin.
Nothing here touches a decision.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def apply_isotonic(p_raw: np.ndarray, cal: pd.DataFrame | None) -> np.ndarray:
    """Map raw scores through the frozen Stage 20 isotonic calibration.

    Linear interpolation between knots, clamped at the ends — the same
    operation as apply_calibration.m. If the mapping is absent the raw score
    is returned UNCHANGED and the caller must label it uncalibrated rather
    than silently presenting it as a probability.
    """
    p = np.asarray(p_raw, dtype=float)
    if cal is None or not len(cal):
        return p
    x = cal["x_raw"].to_numpy()
    y = cal["y_calibrated"].to_numpy()
    order = np.argsort(x)
    x, y = x[order], y[order]
    out = np.interp(p, x, y, left=y[0], right=y[-1])
    return np.clip(out, 0.0, 1.0)


def prior_shift(p_cal: np.ndarray, prior_cal: float, prior_op: float) -> np.ndarray:
    """Re-express a calibrated probability under a different prevalence.

    A calibrated probability is only meaningful under the class prior of the
    data it was calibrated on. The training set is 80.3% damaged by design;
    a fleet inspection population is not. Without this correction the gauge
    reads 0.726 on a healthy wing beside a HEALTHY verdict — both internally
    correct, and contradictory on screen.

        odds_op = odds_cal * (pi_op/(1-pi_op)) * ((1-pi_cal)/pi_cal)

    prior_op is an ASSUMPTION and must be displayed wherever a value derived
    from it is displayed.
    """
    p = np.clip(np.asarray(p_cal, dtype=float), 1e-12, 1 - 1e-12)
    odds = p / (1 - p) * (prior_op / (1 - prior_op)) * ((1 - prior_cal) / prior_cal)
    return odds / (1 + odds)


def display_probability(p_raw, cal, prior_cal, prior_op):
    """Full display chain. Returns (value, is_calibrated)."""
    if cal is None or not len(cal):
        return np.asarray(p_raw, dtype=float), False
    return prior_shift(apply_isotonic(p_raw, cal), prior_cal, prior_op), True
