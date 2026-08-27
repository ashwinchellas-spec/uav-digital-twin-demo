"""UAV-WING-001 Physics-Informed Digital Twin — web interface.

Presents the VALIDATED MATLAB results. It does not recompute physics, does not
run the neural network, and does not re-derive any decision. Every verdict
shown is the one the MATLAB twin recorded.

SOURCE: SIMULATED throughout.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.config import (COL, PRIOR_OPERATIONAL, PRIOR_DATASET, ENVELOPE_G,
                        COMPONENT_LABELS)
from src.data_loader import load_all, integrity
from src.twin_state import state_at
from src import mission as mplay
from src import provenance as pv
from src import visualization as viz

st.set_page_config(page_title="UAV-WING-001 Digital Twin",
                   page_icon="✈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""<style>
 .stApp {{background:{COL['bg']}}}
 [data-testid="stSidebar"] {{background:{COL['panel']}}}
 .block-container {{padding-top:1.1rem;padding-bottom:1rem;max-width:1500px}}
 h1,h2,h3 {{color:{COL['text']}}}
 .panel {{background:{COL['panel']};border:1px solid {COL['grid']};
          border-radius:6px;padding:11px 13px;margin-bottom:9px}}
 .ttl {{color:{COL['muted']};font-size:.7rem;text-transform:uppercase;
        letter-spacing:.09em;margin-bottom:7px;font-weight:600}}
 .sim {{background:#3a2a08;color:#ffc94d;border:1px solid #6b5216;
        border-radius:4px;padding:5px 13px;font-weight:700;font-size:.78rem;
        letter-spacing:.09em;display:inline-block}}
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------- load
try:
    D = load_all()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

META = D["meta"]
MISSION, PHASES = D["mission"], D["phases"]
N = len(MISSION)
mplay.init(N)

# Refuse to start on an unverified baseline, exactly as load_frozen.m does.
if not META.get("baseline_hash_verified", False):
    st.error("FROZEN BASELINE HASH MISMATCH — the twin will not start. "
             "A wrong baseline biases every feature and nothing else would "
             "raise an error.")
    st.stop()

# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Mission control")
    c1, c2 = st.columns(2)
    if c1.button("▶ Play" if not st.session_state.playing else "⏸ Pause",
                 use_container_width=True):
        st.session_state.playing = not st.session_state.playing
        st.rerun()
    if c2.button("↺ Reset", use_container_width=True):
        st.session_state.k = 0; st.session_state.playing = False; st.rerun()

    st.session_state.speed = st.select_slider(
        "Playback speed", options=[1, 2, 4, 8, 16, 32],
        value=st.session_state.speed, format_func=lambda v: f"{v}×")

    k = st.slider("Mission time", 0, N - 1, st.session_state.k,
                  format="", label_visibility="collapsed")
    if k != st.session_state.k:
        st.session_state.k = k; st.session_state.playing = False

    st.markdown("**Jump to phase**")
    for _, r in PHASES.iterrows():
        lbl = str(r["label"])
        lbl = lbl[2:] if lbl[:2].strip().isdigit() else lbl
        if st.button(f"{int(r['idx'])}. {lbl}", use_container_width=True,
                     key=f"ph{r['idx']}"):
            idx = int(np.searchsorted(MISSION["t_s"].to_numpy(), r["t_start"]))
            st.session_state.k = mplay.clamp(idx, N)
            st.session_state.playing = False
            st.rerun()

    st.divider()
    st.markdown("### Integrity")
    for name, ok, extra in integrity(D):
        st.markdown(
            f"<div style='font-size:.72rem;margin-bottom:3px'>"
            f"<span style='color:{'#3fbf7f' if ok else '#d14545'}'>"
            f"{'✓' if ok else '✕'}</span> "
            f"<span style='color:{COL['text']}'>{name}</span>"
            + (f"<br><span style='color:{COL['muted']};margin-left:14px'>{extra}</span>"
               if extra else "") + "</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        f"<div style='font-size:.68rem;color:{COL['muted']};line-height:1.5'>"
        f"<b style='color:{COL['text']}'>Declared assumption</b><br>"
        f"Operational prevalence <b>{PRIOR_OPERATIONAL:.0%}</b> [ASSUMED].<br>"
        f"Calibration prior {PRIOR_DATASET:.1%} (training composition).<br>"
        f"Envelope {ENVELOPE_G[0]}–{ENVELOPE_G[1]} g.</div>",
        unsafe_allow_html=True)

S = state_at(D, st.session_state.k)

# ---------------------------------------------------------------- header
h1, h2, h3 = st.columns([3, 1.5, 3])
with h1:
    st.markdown(f"<h2 style='margin:0'>UAV-WING-001 &nbsp;"
                f"<span style='color:{COL['muted']};font-weight:400;"
                f"font-size:1rem'>Digital Twin</span></h2>",
                unsafe_allow_html=True)
with h2:
    st.markdown("<div class='sim'>SOURCE: SIMULATED</div>", unsafe_allow_html=True)
with h3:
    st.markdown(
        f"<div style='text-align:right;font-size:.68rem;color:{COL['muted']};"
        f"line-height:1.5'>model {META.get('model_version','—')} &nbsp;|&nbsp; "
        f"baseline {META.get('baseline_version','—')}<br>"
        f"hash {str(META.get('baseline_hash',''))[:16]}… &nbsp;|&nbsp; "
        f"Stage 16b <b style='color:{COL['physics']}'>"
        f"{META.get('stage16b_status','—')}</b></div>", unsafe_allow_html=True)

st.markdown(pv.banner(S.banner, S.banner_kind), unsafe_allow_html=True)

st.markdown(
    f"<div style='color:{COL['muted']};font-size:.75rem;margin:-6px 0 10px 0'>"
    f"<b style='color:{COL['text']}'>t = {S.t:.0f} s</b> &nbsp;|&nbsp; "
    f"phase {S.phase_idx} &nbsp;|&nbsp; load {S.load_g:.1f} g &nbsp;|&nbsp; "
    f"modal age {S.modal_age:.0f} s &nbsp;|&nbsp; status {S.status} "
    f"&nbsp;|&nbsp; <span style='color:"
    f"{COL['ai'] if S.in_envelope else COL['physics']}'>{S.envelope_text}"
    f"</span></div>", unsafe_allow_html=True)

nar = mplay.narrative(S.phase_idx)
if nar:
    st.markdown(f"<div class='panel' style='border-left:3px solid {COL['muted']}'>"
                f"<span style='color:{COL['muted']};font-size:.8rem'>{nar}</span>"
                f"</div>", unsafe_allow_html=True)

# ------------------------------------------------------------ main row
c1, c2, c3 = st.columns([1.25, 1, 1])

with c1:
    st.markdown("<div class='ttl'>Wing state &nbsp;"
                + pv.chip("MEASURED", "measured") + "</div>", unsafe_allow_html=True)
    row = PHASES[PHASES["idx"] == S.phase_idx]
    y0 = float(row["damage_y0"].iloc[0]) if len(row) else np.nan
    st.plotly_chart(viz.wing_view(D["planform"], D["gauges"], D["accels"], S, y0),
                    use_container_width=True, config={"displayModeBar": False})

with c2:
    st.markdown("<div class='ttl'>Detection &nbsp;"
                + pv.chip("AI ESTIMATE", "ai") + "</div>", unsafe_allow_html=True)
    if np.isnan(S.p_raw):
        st.markdown(pv.metric("Probability", "—", "ai",
                    "verdict withdrawn: insufficient sensor data"),
                    unsafe_allow_html=True)
    else:
        st.plotly_chart(viz.gauge_dial(S.p_display, S.threshold, S.is_calibrated),
                        use_container_width=True, config={"displayModeBar": False})
        lab = (f"P(damage) = {S.p_display:.3f}" if S.is_calibrated
               else f"raw score = {S.p_raw:.3f} (UNCALIBRATED)")
        sub = (f"calibrated, prior {PRIOR_OPERATIONAL:.0%} [ASSUMED] &nbsp;|&nbsp; "
               f"raw {S.p_raw:.3f} &nbsp;|&nbsp; decision at raw {S.threshold:.3f}"
               if S.is_calibrated else "no calibration mapping available")
        st.markdown(pv.metric(lab, "", "ai", sub), unsafe_allow_html=True)
        st.markdown(pv.metric("Confidence / data quality",
                    f"{S.confidence:.2f} &nbsp;/&nbsp; {S.data_quality:.2f}",
                    "measured"), unsafe_allow_html=True)

with c3:
    st.markdown("<div class='ttl'>Component &nbsp;"
                + pv.chip("AI ESTIMATE", "ai") + "</div>", unsafe_allow_html=True)
    st.plotly_chart(viz.component_bars(S.posterior3, S.component_idx),
                    use_container_width=True, config={"displayModeBar": False})
    if S.component_idx > 0:
        st.markdown(pv.metric("Published component",
                    S.component_label.upper(), "ai",
                    f"mechanism: {['matrix crack','fibre break','delamination'][max(S.mechanism-1,0)]}"),
                    unsafe_allow_html=True)
    else:
        st.markdown(pv.metric("Component", "NOT PUBLISHED", "ai",
                    "below the confidence floor, modal block stale, "
                    "or that component's gauges are lost"), unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{COL['muted']};font-size:.68rem'>Front and rear "
        f"spar are <b>merged</b>: their discrimination measured "
        f"<b>{META.get('front_rear_balanced',0)*100:.1f}%</b> against a 50% "
        f"chance baseline.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------ second row
d1, d2 = st.columns([2.1, 1])
with d1:
    st.markdown("<div class='ttl'>Trends &nbsp;"
                + pv.chip("MEASURED + AI", "measured") + "</div>",
                unsafe_allow_html=True)
    st.plotly_chart(viz.trend(MISSION, PHASES, st.session_state.k),
                    use_container_width=True, config={"displayModeBar": False})
with d2:
    st.markdown("<div class='ttl'>Sensors &nbsp;"
                + pv.chip("MEASURED", "measured") + "</div>", unsafe_allow_html=True)
    st.plotly_chart(viz.sensor_grid(S.n_gauge_valid, 24, S.dead_gauges, "gauge"),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"<div style='color:{COL['muted']};font-size:.7rem;"
                f"margin:-6px 0 6px 0'>strain gauges "
                f"{S.n_gauge_valid}/24 valid</div>", unsafe_allow_html=True)
    st.plotly_chart(viz.sensor_grid(S.n_accel_valid, 16, S.dead_accels, "accel"),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"<div style='color:{COL['muted']};font-size:.7rem'>"
                f"accelerometers {S.n_accel_valid}/16 valid</div>",
                unsafe_allow_html=True)

# ------------------------------------------------------------ scenario
st.markdown("<div class='ttl'>Scenario physics &nbsp;"
            + pv.chip("ASSUMED SEVERITY", "physics") + "</div>",
            unsafe_allow_html=True)
e1, e2 = st.columns([2.1, 1])
with e1:
    parts4 = ["upper_skin", "lower_skin", "front_spar", "rear_spar"]
    comp_name = parts4[1] if S.component_idx == 0 else \
        (parts4[S.component_idx - 1] if S.component_idx <= 2 else "front_spar")
    mech_name = ["matrix_crack", "fibre_break", "delamination"][max(S.mechanism - 1, 0)]
    st.plotly_chart(viz.scenario_plot(D["scenario"], D["bands"], comp_name, mech_name),
                    use_container_width=True, config={"displayModeBar": False})
with e2:
    st.markdown(
        f"<div class='panel' style='border-left:3px solid {COL['physics']}'>"
        f"<div style='color:{COL['physics']};font-weight:600;font-size:.8rem;"
        f"margin-bottom:6px'>NO REMAINING-LIFE PREDICTION</div>"
        f"<div style='color:{COL['muted']};font-size:.72rem;line-height:1.55'>"
        f"{META.get('rul_statement','')}</div></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{COL['muted']};font-size:.68rem;line-height:1.5'>"
        f"GIc, GIIc, paris_C and paris_m remain <b>NaN</b> in the material "
        f"card. They were not fabricated, and fracture-life prediction is "
        f"disabled.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------ footer
st.divider()
st.markdown(pv.legend_html(), unsafe_allow_html=True)
with st.expander("Validated performance and limitations"):
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**Validated performance** *(synthetic, `[C]`)*")
        st.markdown(
            f"- Component, 3-class: **{META.get('component_balanced',0)*100:.1f}%** "
            f"balanced vs {META.get('component_chance',1/3)*100:.1f}% chance\n"
            f"- 4-class was {META.get('component_balanced_4class',0)*100:.1f}% "
            f"vs 25.0% chance\n"
            f"- Calibration ECE **{META.get('ece_before',float('nan')):.4f} → "
            f"{META.get('ece_after',float('nan')):.4f}**\n"
            f"- Decision space: **{META.get('decision_space','raw score')}**\n"
            f"- Envelope **{ENVELOPE_G[0]}–{ENVELOPE_G[1]} g**")
    with f2:
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {x}" for x in META.get("limitations", [])))
    st.caption("Physics core is `[V]` against NASA RP-1351, the MacNeal–Harder "
               "patch test and closed-form FSDT plate solutions. **No AI result "
               "is `[V]`.** Synthetic performance is not physical validation.")

# ------------------------------------------------------------- playback
if st.session_state.playing:
    time.sleep(0.08)
    mplay.step(N)
    st.rerun()
