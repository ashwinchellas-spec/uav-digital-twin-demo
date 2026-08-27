"""Provenance rendering. Every displayed number carries its source."""
from __future__ import annotations
import streamlit as st
from .config import COL

LEGEND = [
    ("BLUE",  COL["measured"], "measured / sensor-derived"),
    ("GREEN", COL["ai"],       "AI estimate"),
    ("AMBER", COL["physics"],  "physics under an ASSUMED severity band"),
]


def chip(text: str, kind: str) -> str:
    c = {"measured": COL["measured"], "ai": COL["ai"],
         "physics": COL["physics"], "muted": COL["muted"]}.get(kind, COL["muted"])
    return (f"<span style='background:{c}22;color:{c};border:1px solid {c}55;"
            f"border-radius:3px;padding:1px 6px;font-size:0.68rem;"
            f"letter-spacing:.04em'>{text}</span>")


def metric(label: str, value: str, kind: str = "measured", sub: str = "") -> str:
    c = {"measured": COL["measured"], "ai": COL["ai"],
         "physics": COL["physics"]}.get(kind, COL["text"])
    s = (f"<div style='background:{COL['panel_hi']};border-left:3px solid {c};"
         f"border-radius:4px;padding:8px 11px;margin-bottom:7px'>"
         f"<div style='color:{COL['muted']};font-size:0.68rem;"
         f"text-transform:uppercase;letter-spacing:.06em'>{label}</div>"
         f"<div style='color:{COL['text']};font-size:1.22rem;font-weight:600;"
         f"line-height:1.35'>{value}</div>")
    if sub:
        s += f"<div style='color:{COL['muted']};font-size:0.7rem'>{sub}</div>"
    return s + "</div>"


def banner(text: str, kind: str) -> str:
    bg = {"healthy": COL["healthy"], "damage": COL["damage"],
          "noverdict": COL["noverdict"], "degraded": COL["degraded"]}.get(kind, COL["panel"])
    return (f"<div style='background:{bg};color:#fff;border-radius:5px;"
            f"padding:13px;text-align:center;font-size:1.35rem;font-weight:700;"
            f"letter-spacing:.06em;margin:4px 0 12px 0'>{text}</div>")


def legend_html() -> str:
    parts = [f"<span style='color:{c};font-weight:600'>{n}</span> "
             f"<span style='color:{COL['muted']}'>{d}</span>"
             for n, c, d in LEGEND]
    return ("<div style='font-size:0.72rem;line-height:1.6'>PROVENANCE&nbsp;&nbsp;"
            + "&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts) + "</div>")
