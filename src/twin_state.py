"""Twin state at a given mission tick.

Reproduces the MATLAB twin's reporting rules. It does NOT recompute anything:
detection, component and status all come from the logged MATLAB decision. The
only things computed here are display transforms and the envelope check.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .config import (PRIOR_DATASET, PRIOR_OPERATIONAL, ENVELOPE_G,
                     MIN_COMPONENT_PROB, COMPONENT_LABELS)
from .calibration import display_probability


@dataclass
class TwinState:
    t: float
    load_g: float
    p_raw: float
    p_display: float
    is_calibrated: bool
    threshold: float
    detected: bool
    confidence: float
    data_quality: float
    modal_age: float
    n_gauge_valid: int
    n_accel_valid: int
    status: str
    alert: str
    banner: str
    banner_kind: str
    in_envelope: bool
    envelope_text: str
    component_idx: int          # 0 = not published, 1..3
    component_label: str
    posterior3: np.ndarray | None
    mechanism: int
    phase_idx: int
    phase_label: str
    dead_gauges: list = field(default_factory=list)
    dead_accels: list = field(default_factory=list)


def _phase_of(t: float, phases: pd.DataFrame) -> tuple[int, str]:
    for _, r in phases.iterrows():
        if r["t_start"] <= t < r["t_end"]:
            return int(r["idx"]), str(r["label"])
    last = phases.iloc[-1]
    return int(last["idx"]), str(last["label"])


def state_at(data: dict, k: int) -> TwinState:
    m = data["mission"].iloc[k]
    phases = data["phases"]
    meta = data["meta"]

    p_raw = float(m["p_raw"])
    p_disp, is_cal = display_probability(
        np.array([p_raw]), data["calibration"], PRIOR_DATASET, PRIOR_OPERATIONAL)
    p_disp = float(p_disp[0])

    load = float(m["load_g"])
    lo, hi = ENVELOPE_G
    in_env = lo <= load <= hi
    env_text = ("LOAD WITHIN VALIDATED ENVELOPE" if in_env
                else f"OUT OF MEASUREMENT ENVELOPE ({lo}-{hi} g)")

    status = str(m["status"])
    alert = str(m["alert"])

    # Banner. Envelope violation takes precedence: outside the validated range
    # the twin has no basis for a verdict and must not invent one.
    if not in_env:
        banner, kind = "OUT OF MEASUREMENT ENVELOPE", "noverdict"
    elif "NO VERDICT" in alert or status == "INSUFFICIENT DATA":
        banner, kind = "NO VERDICT  -  INSUFFICIENT DATA", "noverdict"
    elif "DAMAGE" in alert:
        banner = "DAMAGE DETECTED" + (" (DEGRADED)" if status == "DEGRADED" else "")
        kind = "damage"
    elif status == "DEGRADED":
        banner, kind = "HEALTHY  (DEGRADED INSTRUMENTATION)", "degraded"
    else:
        banner, kind = "HEALTHY", "healthy"

    # ---- component: merge front+rear spar, honour the confidence floor ----
    post3, comp_idx = None, 0
    if data["posteriors"] is not None:
        pr = data["posteriors"].iloc[k]
        p4 = np.array([pr["p_upper"], pr["p_lower"], pr["p_front"], pr["p_rear"]],
                      dtype=float)
        if np.all(np.isfinite(p4)):
            post3 = np.array([p4[0], p4[1], p4[2] + p4[3]])
            if int(m["component"]) > 0 and post3.max() >= MIN_COMPONENT_PROB:
                comp_idx = int(np.argmax(post3)) + 1
    if comp_idx == 0 and int(m["component"]) > 0 and post3 is None:
        # posteriors unavailable: fall back to the MATLAB published class,
        # mapping 4 -> 3 by merging spar. Never fabricate a distribution.
        c = int(m["component"])
        comp_idx = 3 if c >= 3 else c

    comp_label = COMPONENT_LABELS[comp_idx - 1] if comp_idx > 0 else "NOT PUBLISHED"

    pidx, plabel = _phase_of(float(m["t_s"]), phases)
    dg = data["fault_map"]["faults_gauges"].get(pidx, [])
    da = data["fault_map"]["faults_accels"].get(pidx, [])

    return TwinState(
        t=float(m["t_s"]), load_g=load, p_raw=p_raw, p_display=p_disp,
        is_calibrated=is_cal, threshold=float(m["threshold"]),
        detected=bool(m["detected"]), confidence=float(m["confidence"]),
        data_quality=float(m["data_quality"]), modal_age=float(m["modal_age_s"]),
        n_gauge_valid=int(m["n_gauge_valid"]), n_accel_valid=int(m["n_accel_valid"]),
        status=status, alert=alert, banner=banner, banner_kind=kind,
        in_envelope=in_env, envelope_text=env_text,
        component_idx=comp_idx, component_label=comp_label, posterior3=post3,
        mechanism=int(m["mechanism"]), phase_idx=pidx, phase_label=plabel,
        dead_gauges=dg, dead_accels=da,
    )
