"""Configuration and declared assumptions for the web Digital Twin.

Every ASSUMED value lives here, in one place, labelled. Nothing in this file
is measured; it is all either read from the MATLAB export or declared as an
assumption that the UI must display alongside any number derived from it.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
def find_export(start: Path | None = None) -> Path:
    """Locate streamlit_export/ produced by export_for_streamlit.m."""
    here = Path(start or __file__).resolve()
    for base in [here.parent.parent, *here.parents]:
        for cand in (base / "streamlit_export",
                     base / "UAVWingDigitalTwin" / "streamlit_export"):
            if (cand / "mission.csv").exists():
                return cand
    raise FileNotFoundError(
        "streamlit_export/ not found. Run export_for_streamlit.m in MATLAB "
        "and place the folder beside this application, or set DT_EXPORT_DIR."
    )

# ------------------------------------------------------- ASSUMED values
# The training set is 80.3% damaged BY DESIGN, for class balance. A fleet
# inspection population is not. A calibrated probability is only meaningful
# under the prior of the data it was calibrated on, so the displayed value is
# re-expressed at the operational prior below and the prior is shown with it.
PRIOR_DATASET = 0.803          # measured: composition of the VL test split
PRIOR_OPERATIONAL = 0.05       # [ASSUMED] declared, not measured

# Validated measurement envelope. Outside this the twin does not extrapolate.
ENVELOPE_G = (0.5, 3.0)

# Component confidence floor, matching the MATLAB twin.
MIN_COMPONENT_PROB = 0.40

# NaN sentinel used by the MATLAB exporter (nz()): probabilities, confidence
# and data quality are all non-negative, so -1 is unambiguous.
NAN_SENTINEL = -1.0

# ------------------------------------------------------------- palette
COL = {
    "bg":        "#0d0f12",
    "panel":     "#16181d",
    "panel_hi":  "#1d2027",
    "text":      "#e8eaee",
    "muted":     "#8b93a1",
    "grid":      "#2a2e36",
    "measured":  "#3a9bdc",   # BLUE   sensor-derived
    "ai":        "#3fbf7f",   # GREEN  AI estimate
    "physics":   "#e8a33d",   # AMBER  physics under ASSUMED severity
    "healthy":   "#2f7d4f",
    "damage":    "#a33636",
    "noverdict": "#7a6320",
    "degraded":  "#b07a20",
    "dead":      "#d14545",
}

COMPONENT_LABELS = ["upper skin", "lower skin", "spar"]
COMPONENT_COLORS = ["#3a9bdc", "#3fbf7f", "#e8a33d"]
PART_COLORS_4 = ["#3a9bdc", "#3fbf7f", "#e8a33d", "#d16a6a"]
