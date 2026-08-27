"""Plotly figures. Visualisation-only: nothing here is authoritative."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import COL, COMPONENT_LABELS, COMPONENT_COLORS, PART_COLORS_4


def _dark(fig, h=300, margin=None):
    fig.update_layout(
        height=h, paper_bgcolor=COL["panel"], plot_bgcolor=COL["bg"],
        font=dict(color=COL["text"], size=11),
        margin=margin or dict(l=48, r=16, t=28, b=38),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=COL["grid"], zerolinecolor=COL["grid"])
    fig.update_yaxes(gridcolor=COL["grid"], zerolinecolor=COL["grid"])
    return fig


def wing_view(planform, gauges, accels, st_, damage_y0=None):
    """Wing planform with REAL sensor positions and the REAL fault mask."""
    fig = go.Figure()
    if planform is not None and len(planform):
        y = planform["y_span"].to_numpy()
        xf = planform["x_front"].to_numpy()
        xr = planform["x_rear"].to_numpy()
        fig.add_trace(go.Scatter(
            x=np.concatenate([y, y[::-1]]), y=np.concatenate([xf, xr[::-1]]),
            fill="toself", fillcolor="rgba(58,70,88,0.42)",
            line=dict(color="#5d6a7d", width=1.4), hoverinfo="skip"))

    if gauges is not None and len(gauges):
        dead = set(st_.dead_gauges)
        for _, g in gauges.iterrows():
            i = int(g["gauge_idx"])
            is_dead = i in dead
            fig.add_trace(go.Scatter(
                x=[g["y_span"]], y=[g["x_chord"]], mode="markers",
                marker=dict(
                    size=13 if is_dead else 9,
                    color=COL["dead"] if is_dead else PART_COLORS_4[int(g["component_idx"]) - 1],
                    symbol="x" if is_dead else "circle",
                    line=dict(color="#0d0f12", width=1)),
                hovertemplate=(f"gauge {i} — {g['component_name']}"
                               f"{' — FAILED' if is_dead else ''}<extra></extra>")))

    if accels is not None and len(accels):
        dead_a = set(st_.dead_accels)
        for _, a in accels.iterrows():
            i = int(a["accel_idx"])
            d = i in dead_a
            fig.add_trace(go.Scatter(
                x=[a["y_span"]], y=[a["x_chord"]], mode="markers",
                marker=dict(size=11 if d else 7,
                            color=COL["dead"] if d else "#9aa4b4",
                            symbol="x" if d else "diamond",
                            line=dict(color="#0d0f12", width=1)),
                hovertemplate=f"accel {i}{' — FAILED' if d else ''}<extra></extra>"))

    if damage_y0 is not None and np.isfinite(damage_y0) and planform is not None:
        xc = float(np.interp(damage_y0, planform["y_span"],
                             (planform["x_front"] + planform["x_rear"]) / 2))
        fig.add_trace(go.Scatter(
            x=[damage_y0], y=[xc], mode="markers+text",
            marker=dict(size=22, color="#e34d4d", symbol="star",
                        line=dict(color="#fff", width=1.5)),
            text=["TRUE DAMAGE"], textposition="top center",
            textfont=dict(color="#ff9b9b", size=9), hoverinfo="skip"))

    fig.update_xaxes(title_text="span y [m]")
    fig.update_yaxes(title_text="chord x [m]", scaleanchor="x", scaleratio=1)
    return _dark(fig, 300)


def trend(mission, phases, k, threshold_col="threshold"):
    fig = go.Figure()
    t = mission["t_s"].to_numpy()
    for _, r in phases.iterrows():
        fig.add_vline(x=float(r["t_start"]), line=dict(color="#4b5261",
                      width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=t, y=mission["load_g"] / 4.0, mode="lines",
                  line=dict(color="#6fbf5a", width=1.4, dash="dot"),
                  name="load g / 4",
                  hovertemplate="load %{customdata:.2f} g<extra></extra>",
                  customdata=mission["load_g"]))
    fig.add_trace(go.Scatter(x=t, y=mission[threshold_col], mode="lines",
                  line=dict(color=COL["physics"], width=1.4, dash="dash"),
                  name="threshold"))
    fig.add_trace(go.Scatter(x=t, y=mission["p_raw"], mode="lines",
                  line=dict(color=COL["measured"], width=1.8),
                  name="raw score",
                  hovertemplate="t %{x:.0f}s  raw %{y:.3f}<extra></extra>"))
    fig.add_vline(x=float(mission["t_s"].iloc[k]),
                  line=dict(color="#ffffff", width=1.8))
    fig.update_xaxes(title_text="mission time [s]")
    fig.update_yaxes(title_text="score", range=[0, 1.05])
    fig.update_layout(showlegend=True,
        legend=dict(orientation="h", y=1.13, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    return _dark(fig, 268, dict(l=48, r=16, t=42, b=40))


def component_bars(post3, comp_idx):
    published = comp_idx > 0
    v = post3 if post3 is not None else np.zeros(3)
    if post3 is None and published:
        v = np.zeros(3); v[comp_idx - 1] = 1.0
    cols = [c if published else c + "44" for c in COMPONENT_COLORS]
    fig = go.Figure(go.Bar(
        x=COMPONENT_LABELS, y=v, marker_color=cols,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.05], title_text="posterior")
    return _dark(fig, 218)


def sensor_grid(n_valid, total, dead, label):
    """Grid of sensor tiles. Dead ones are the REAL failed indices."""
    cols = 8
    rows = int(np.ceil(total / cols))
    xs, ys, cs, tx = [], [], [], []
    dead = set(dead)
    for i in range(1, total + 1):
        xs.append((i - 1) % cols)
        ys.append(rows - 1 - (i - 1) // cols)
        cs.append(COL["dead"] if i in dead else "#3fa06a")
        tx.append(f"{label} {i}" + (" — FAILED" if i in dead else ""))
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers", marker=dict(size=19, symbol="square",
        color=cs, line=dict(color=COL["bg"], width=2)),
        text=tx, hovertemplate="%{text}<extra></extra>"))
    fig.update_xaxes(visible=False, range=[-0.6, cols - 0.4])
    fig.update_yaxes(visible=False, range=[-0.6, rows - 0.4])
    return _dark(fig, 96, dict(l=6, r=6, t=6, b=6))


def scenario_plot(scenario, bands, component_name, mechanism_name):
    fig = go.Figure()
    if scenario is None or not len(scenario):
        return _dark(fig, 260)
    sub = scenario[(scenario["component"] == component_name)]
    for mech, colr in zip(sorted(sub["mechanism"].unique()),
                          [COL["measured"], COL["ai"], COL["physics"]]):
        s = sub[sub["mechanism"] == mech].sort_values("assumed_severity")
        fig.add_trace(go.Scatter(
            x=s["assumed_severity"], y=s["margin"], mode="lines+markers",
            line=dict(color=colr, width=2.2 if mech == mechanism_name else 1.2),
            opacity=1.0 if mech == mechanism_name else 0.45,
            name=mech.replace("_", " "),
            hovertemplate=f"{mech}<br>severity %{{x:.2f}}<br>margin %{{y:.2f}}<extra></extra>"))
    if bands is not None:
        for _, b in bands.iterrows():
            fig.add_vrect(x0=b["lo"], x1=b["hi"], fillcolor="#ffffff",
                          opacity=0.05, line_width=0,
                          annotation_text=b["name"],
                          annotation_font=dict(size=9, color=COL["muted"]))
    fig.update_xaxes(title_text="ASSUMED severity  (scenario input, NOT an AI estimate)")
    fig.update_yaxes(title_text="margin vs allowable")
    fig.update_layout(showlegend=True,
        legend=dict(orientation="h", y=1.15, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    return _dark(fig, 262, dict(l=48, r=16, t=44, b=44))


def gauge_dial(value, threshold_disp, is_cal):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number=dict(font=dict(size=30, color=COL["text"]), valueformat=".3f"),
        gauge=dict(
            axis=dict(range=[0, 1], tickcolor=COL["muted"],
                      tickfont=dict(size=9, color=COL["muted"])),
            bar=dict(color=COL["ai"] if is_cal else COL["muted"], thickness=0.26),
            bgcolor=COL["bg"], borderwidth=0,
            steps=[dict(range=[0, 0.33], color="#1e3d2c"),
                   dict(range=[0.33, 0.66], color="#3d371e"),
                   dict(range=[0.66, 1.0], color="#3d1e1e")])))
    return _dark(fig, 190, dict(l=22, r=22, t=14, b=6))
