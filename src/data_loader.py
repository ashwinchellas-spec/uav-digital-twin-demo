"""Load the MATLAB export. Read-only, cached, no recomputation.

This module is the ONLY place that touches the exported files. Everything
downstream works on already-parsed arrays, so playback never re-reads disk.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from .config import find_export, NAN_SENTINEL


def _decode_sentinels(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert the MATLAB exporter's -1 NaN sentinel back to NaN.

    export_for_streamlit.m writes NaN as -1 because CSV has no NaN literal
    that MATLAB and pandas agree on. Probability, confidence and data quality
    are all non-negative, so the sentinel is unambiguous. Leaving it undecoded
    would display Phase 6's withdrawn verdict as p = -1.
    """
    for c in cols:
        if c in df.columns:
            df.loc[df[c] == NAN_SENTINEL, c] = np.nan
    return df


@st.cache_data(show_spinner=False)
def load_all(export_dir: str | None = None) -> dict:
    d = Path(export_dir) if export_dir else find_export()

    mission = pd.read_csv(d / "mission.csv")
    mission = _decode_sentinels(mission, ["p_raw", "confidence", "data_quality"])

    out = {
        "dir": str(d),
        "mission": mission,
        "phases": pd.read_csv(d / "phases.csv"),
        "meta": json.loads((d / "meta.json").read_text()),
    }

    opt = {
        "posteriors":     "posteriors.csv",
        "faults_gauges":  "faults_gauges.csv",
        "faults_accels":  "faults_accels.csv",
        "gauges":         "sensors_gauges.csv",
        "accels":         "sensors_accels.csv",
        "planform":       "wing_planform.csv",
        "calibration":    "calibration.csv",
        "scenario":       "scenario.csv",
        "bands":          "scenario_bands.csv",
        "baseline":       "baseline.csv",
    }
    for key, fname in opt.items():
        p = d / fname
        out[key] = pd.read_csv(p) if p.exists() else None

    if out["posteriors"] is not None:
        out["posteriors"] = _decode_sentinels(
            out["posteriors"],
            [c for c in out["posteriors"].columns if c.startswith("p_")])

    # Fault mask: phase index -> list of failed sensor indices. REAL, exported
    # from the mission definition. Never synthesised.
    out["fault_map"] = {}
    for key, col in (("faults_gauges", "gauge_idx"), ("faults_accels", "accel_idx")):
        m = {}
        if out[key] is not None and len(out[key]):
            for ph, grp in out[key].groupby("phase_idx"):
                m[int(ph)] = grp[col].astype(int).tolist()
        out["fault_map"][key] = m

    return out


def integrity(data: dict) -> list[tuple[str, bool, str]]:
    """Checks run at startup and shown in the provenance panel."""
    meta = data["meta"]
    checks = []
    checks.append(("frozen baseline hash verified",
                   bool(meta.get("baseline_hash_verified", False)),
                   str(meta.get("baseline_hash", ""))[:16] + "..."))
    checks.append(("MATLAB artifacts unmodified",
                   not meta.get("matlab_artifacts_modified", True), ""))
    checks.append(("no quantitative RUL published",
                   not meta.get("rul_published", True), ""))
    checks.append(("fracture-life prediction disabled",
                   not meta.get("fracture_life_enabled", True),
                   "GIc / paris_C remain NaN"))
    checks.append(("component posteriors available",
                   data["posteriors"] is not None, ""))
    checks.append(("real sensor fault mask available",
                   bool(data["fault_map"]["faults_gauges"]), ""))
    checks.append(("calibration mapping available",
                   data["calibration"] is not None,
                   f"ECE {meta.get('ece_before', float('nan')):.4f} "
                   f"-> {meta.get('ece_after', float('nan')):.4f}"))
    checks.append(("Stage 16b status",
                   meta.get("stage16b_status") == "PARTIAL",
                   "PARTIAL - block logic verified, Simulink model not"))
    return checks
