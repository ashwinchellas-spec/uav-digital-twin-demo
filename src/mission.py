"""Mission playback state machine.

Playback is index-based over the logged ticks. It never interpolates a
DECISION: every displayed verdict is the one MATLAB recorded at that tick.
"""
from __future__ import annotations
import streamlit as st


def init(n: int):
    ss = st.session_state
    ss.setdefault("k", 0)
    ss.setdefault("playing", False)
    ss.setdefault("speed", 8)
    ss.setdefault("n", n)


def clamp(k: int, n: int) -> int:
    return max(0, min(int(k), n - 1))


def step(n: int) -> bool:
    """Advance one frame. Returns True when the end is reached."""
    ss = st.session_state
    ss.k = clamp(ss.k + ss.speed, n)
    if ss.k >= n - 1:
        ss.playing = False
        return True
    return False


def narrative(phase_idx: int) -> str:
    return {
        1: "Baseline condition. The detection rate here is the FALSE ALARM "
           "behaviour of the chosen operating point, not a fault.",
        2: "Load raised, structure STILL HEALTHY. Under the fixed-load model "
           "this produced p = 0.968 — a larger response than real damage.",
        3: "Lower-skin fibre breakage. Detection is immediate: the strain path "
           "updates every tick while the modal path takes up to 64 s.",
        4: "Damage grown. Severity is NOT claimed as an AI number — the physics "
           "panel shows an ASSUMED band instead.",
        5: "Two gauges failed. DEGRADED: the twin answers but declares its "
           "instrumentation incomplete.",
        6: "Twelve gauges failed including ALL front-spar gauges. The twin "
           "WITHDRAWS its verdict rather than guessing.",
        7: "Sensors restored. The verdict returns and agrees with the "
           "pre-fault state.",
    }.get(int(phase_idx), "")
