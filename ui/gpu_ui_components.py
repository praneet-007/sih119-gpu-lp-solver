"""
gpu_ui_components.py -- GPU Optimization Control Center visual components.

REBUILD (not a cosmetic patch): every function here renders a piece of a
single continuous computational-journey visual, not an isolated "card".
front1.py is the orchestration layer -- it owns all real data (parsed
CSV, solver output) and decides what's real vs DEMO; this module only
knows how to *animate* whatever it's handed. No function here invents a
number -- every value rendered is a parameter passed in by the caller.

Animation strategy (see MASTER PROMPT section 30 -- no expensive Python
rerender loops):
  - Anything whose final value is known at render time (bar fills, file
    travel, particle flow, pulses, staggered reveals, the iteration
    timeline, the console) is pure CSS: a single st.markdown() call with
    a `@keyframes` animation and, where the target is dynamic, a CSS
    custom property (`--target: 74%; animation: grow to var(--target)`).
    The browser animates it smoothly with zero Streamlit reruns.
  - The two places where a *counting* numeric animation genuinely needs
    a clock (system-profile counters, the speedup number) use a small
    vanilla-JS snippet inside st.components.v1.html -- an isolated
    iframe, still zero Streamlit reruns, per the "lightweight
    JavaScript / Streamlit components" guidance.
  - The iteration timeline and matrix fingerprint are real SVG built
    from the caller's actual data points (positions computed in Python
    from real values), not a canned image.
"""

from __future__ import annotations

import html as _html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


ACCENT = "#76b900"
ACCENT_BRIGHT = "#a6ff00"
BG_DARK = "#04060a"
WARN = "#ffb020"
DANGER = "#ff4d4d"


# ============================================================
# BASE CSS -- keyframes + shared control-center classes
# ============================================================

def inject_base_css() -> None:
    """Call once near the top of front1.py. Everything below assumes
    these classes/keyframes exist on the page."""

    st.markdown(
        """
        <style>

        .ctrl-mono { font-family: "JetBrains Mono", "Consolas", "Courier New", monospace; }

        /* ---------------- vertical stage pipeline (data pipe, not a box row) ---------------- */

        .ctrl-pipe { padding: 6px 2px; }

        .ctrl-pipe-node {
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 3px 0;
        }

        .ctrl-pipe-glyph {
            width: 20px;
            height: 20px;
            flex: 0 0 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 900;
        }

        .ctrl-pipe-glyph.pending { border: 1px solid rgba(255,255,255,0.15); color: #4d5a47; }
        .ctrl-pipe-glyph.done { background: rgba(118,185,0,0.18); color: #76b900; border: 1px solid rgba(118,185,0,0.5); }
        .ctrl-pipe-glyph.active {
            background: #76b900; color: #04060a; box-shadow: 0 0 16px rgba(118,185,0,0.75);
            animation: ctrl-node-pulse 1.1s ease-in-out infinite;
        }

        @keyframes ctrl-node-pulse {
            0%, 100% { box-shadow: 0 0 8px rgba(118,185,0,0.45); }
            50% { box-shadow: 0 0 22px rgba(118,185,0,0.9); }
        }

        .ctrl-pipe-label {
            font-size: 11.5px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }

        .ctrl-pipe-label.pending { color: #4d5a47; }
        .ctrl-pipe-label.done { color: #9aa696; }
        .ctrl-pipe-label.active { color: #a6ff00; text-shadow: 0 0 8px rgba(166,255,0,0.5); }

        /* the connector itself is the "data pipe" between two nodes */

        .ctrl-pipe-connector {
            position: relative;
            width: 20px;
            flex: 0 0 20px;
            height: 30px;
            margin-left: 0px;
        }

        .ctrl-pipe-connector::before {
            content: "";
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 2px;
            transform: translateX(-50%);
        }

        .ctrl-pipe-connector.pending::before { background: repeating-linear-gradient(180deg, rgba(255,255,255,0.12) 0 4px, transparent 4px 8px); }
        .ctrl-pipe-connector.done::before { background: rgba(118,185,0,0.55); }
        .ctrl-pipe-connector.active::before { background: linear-gradient(180deg, rgba(118,185,0,0.15), rgba(118,185,0,0.55)); }

        .ctrl-pipe-connector .particle {
            position: absolute;
            left: 50%;
            top: 0%;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #a6ff00;
            box-shadow: 0 0 8px #a6ff00;
            transform: translateX(-50%);
            animation: ctrl-pipe-flow 1.1s linear infinite;
        }

        @keyframes ctrl-pipe-flow {
            0%   { top: 0%; opacity: 0; }
            12%  { opacity: 1; }
            88%  { opacity: 1; }
            100% { top: 92%; opacity: 0; }
        }

        .ctrl-pipe-connector .stream-p { animation-duration: 0.55s; }

        .ctrl-pipe-connector .file-marker {
            position: absolute;
            left: 50%;
            top: 0%;
            transform: translateX(-50%);
            font-size: 13px;
            animation: ctrl-pipe-flow 1.6s ease-in-out infinite;
            filter: drop-shadow(0 0 5px rgba(118,185,0,0.8));
        }

        .ctrl-pipe-connector .packet-marker {
            position: absolute;
            left: 50%;
            top: 0%;
            transform: translateX(-50%);
            font-size: 12px;
            color: #a6ff00;
            animation: ctrl-pipe-flow 1.2s ease-in-out infinite;
        }

        .ctrl-pipe-connector .assemble-p {
            position: absolute;
            top: 0%;
            width: 6px;
            height: 6px;
            background: #a6ff00;
            box-shadow: 0 0 6px #a6ff00;
            animation: ctrl-pipe-assemble 1.3s ease-in infinite;
        }

        @keyframes ctrl-pipe-assemble {
            0%   { top: 0%; opacity: 0; }
            15%  { opacity: 1; }
            100% { top: 90%; left: 50% !important; opacity: 0; }
        }

        .ctrl-pipe-connector .ripple-ring {
            position: absolute;
            left: 50%;
            bottom: -2px;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 2px solid #a6ff00;
            transform: translateX(-50%) scale(0.3);
            animation: ctrl-pipe-ripple 1.1s ease-out infinite;
        }

        @keyframes ctrl-pipe-ripple {
            0%   { transform: translateX(-50%) scale(0.3); opacity: 0.9; }
            100% { transform: translateX(-50%) scale(2.4); opacity: 0; }
        }

        .ctrl-pipe-mode-tag {
            font-size: 8.5px;
            color: #4d5a47;
            letter-spacing: 0.5px;
            margin-left: 4px;
        }
        .ctrl-pipe-mode-tag.active { color: #76b900; font-weight: 800; }

        /* ---------------- section frame (replaces "card") ---------------- */

        .ctrl-frame {
            border-left: 2px solid rgba(118,185,0,0.35);
            padding: 4px 0 4px 16px;
            margin: 10px 0 18px 2px;
        }

        .ctrl-frame-title {
            font-size: 13px;
            font-weight: 900;
            letter-spacing: 2px;
            color: #76b900;
            margin-bottom: 6px;
        }

        .ctrl-tag {
            display: inline-block;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1px;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 8px;
            vertical-align: middle;
        }

        .ctrl-tag-demo { background: rgba(255,176,32,0.15); color: #ffb020; border: 1px solid rgba(255,176,32,0.5); }
        .ctrl-tag-live { background: rgba(118,185,0,0.15); color: #76b900; border: 1px solid rgba(118,185,0,0.5); }
        .ctrl-tag-unavail { background: rgba(255,255,255,0.06); color: #808a7c; border: 1px solid rgba(255,255,255,0.15); }

        /* ---------------- physical file object + travel rail ---------------- */

        .ctrl-file-obj {
            display: inline-block;
            border: 1px solid rgba(118,185,0,0.5);
            border-radius: 10px;
            padding: 12px 18px;
            background: rgba(118,185,0,0.06);
            box-shadow: 0 0 20px rgba(118,185,0,0.15);
        }

        .ctrl-file-obj .fname { color: #d7e3cf; font-weight: 800; font-size: 14px; }
        .ctrl-file-obj .fmeta { color: #9aa696; font-size: 11.5px; margin-top: 4px; }

        .ctrl-travel-rail {
            position: relative;
            height: 30px;
            margin: 10px 0 16px 0;
            background: linear-gradient(90deg, rgba(118,185,0,0.05), rgba(118,185,0,0.25));
            border-radius: 15px;
            overflow: hidden;
        }

        .ctrl-travel-rail .marker {
            position: absolute;
            top: 6px;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: radial-gradient(circle, #d4ff66, #76b900);
            box-shadow: 0 0 14px #a6ff00;
            animation: ctrl-travel 2.2s ease-in-out infinite;
        }

        @keyframes ctrl-travel {
            0%   { left: 1%; }
            50%  { left: 94%; }
            100% { left: 1%; }
        }

        .ctrl-travel-endlabel {
            position: absolute;
            right: 10px;
            top: 6px;
            font-size: 10px;
            font-weight: 800;
            color: #04060a;
            letter-spacing: 1px;
        }

        .ctrl-travel-startlabel {
            position: absolute;
            left: 10px;
            top: 6px;
            font-size: 10px;
            font-weight: 800;
            color: #04060a;
            letter-spacing: 1px;
        }

        /* ---------------- generic labeled fill bars (tank / bottleneck / constraint) ---------------- */

        .ctrl-bar-row { margin: 9px 0; }

        .ctrl-bar-head {
            display: flex;
            justify-content: space-between;
            font-size: 11.5px;
            color: #b8c7ad;
            margin-bottom: 3px;
        }

        .ctrl-bar-track {
            height: 16px;
            border-radius: 5px;
            background: rgba(255,255,255,0.05);
            overflow: hidden;
        }

        .ctrl-bar-fill {
            height: 100%;
            border-radius: 5px;
            width: 0%;
            animation: ctrl-grow 1.3s cubic-bezier(.2,.9,.25,1) forwards;
        }

        @keyframes ctrl-grow { to { width: var(--target); } }

        .ctrl-bar-fill.normal { background: linear-gradient(90deg, #3f5c00, #76b900); }
        .ctrl-bar-fill.warn   { background: linear-gradient(90deg, #7a5b00, #ffb020); }
        .ctrl-bar-fill.crit   {
            background: linear-gradient(90deg, #7a1f1f, #ff4d4d);
            animation: ctrl-grow 1.3s cubic-bezier(.2,.9,.25,1) forwards, ctrl-crit-pulse 1s ease-in-out infinite 1.3s;
        }

        @keyframes ctrl-crit-pulse {
            0%, 100% { box-shadow: 0 0 4px rgba(255,77,77,0.3); }
            50% { box-shadow: 0 0 18px rgba(255,77,77,0.85); }
        }

        /* ---------------- equations ---------------- */

        .ctrl-eq {
            font-family: "JetBrains Mono", monospace;
            font-size: 13.5px;
            color: #d7e3cf;
            padding: 3px 0;
            opacity: 0;
            transform: translateX(-8px);
            animation: ctrl-eq-in 0.45s ease forwards;
            border-bottom: 1px dashed rgba(118,185,0,0.1);
        }

        .ctrl-eq b { color: #76b900; }

        @keyframes ctrl-eq-in { to { opacity: 1; transform: translateX(0); } }

        /* ---------------- transform arrow (equations -> matrix, etc.) ---------------- */

        .ctrl-xform {
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 14px 0;
        }

        .ctrl-xform-node {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1px;
            color: #9aa696;
            white-space: nowrap;
        }

        .ctrl-xform-track {
            flex: 1;
            height: 4px;
            border-radius: 2px;
            background: rgba(118,185,0,0.12);
            position: relative;
            overflow: visible;
        }

        .ctrl-xform-track .ripple {
            position: absolute;
            top: -6px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid #a6ff00;
            animation: ctrl-ripple 1.6s ease-out infinite;
        }

        @keyframes ctrl-ripple {
            0%   { left: 0%; opacity: 1; transform: scale(0.4); }
            80%  { opacity: 0.15; }
            100% { left: 100%; opacity: 0; transform: scale(1.3); }
        }

        /* ---------------- matrix bracket notation ---------------- */

        .ctrl-matrix {
            font-family: "JetBrains Mono", monospace;
            font-size: 12.5px;
            color: #cfe8b0;
            background: rgba(118,185,0,0.04);
            border: 1px solid rgba(118,185,0,0.25);
            border-radius: 10px;
            padding: 14px 18px;
            overflow-x: auto;
            white-space: pre;
            line-height: 1.55;
        }

        /* ---------------- CUDA transfer ---------------- */

        .ctrl-zone {
            border: 1px solid rgba(118,185,0,0.4);
            border-radius: 12px;
            padding: 14px;
            text-align: center;
            min-width: 190px;
        }

        .ctrl-zone-title { font-size: 11px; font-weight: 900; letter-spacing: 1.5px; color: #76b900; margin-bottom: 6px; }
        .ctrl-zone-sub { font-size: 11px; color: #9aa696; }

        .ctrl-cuda-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin: 14px 0;
        }

        .ctrl-cuda-track {
            flex: 0 0 160px;
            height: 6px;
            border-radius: 3px;
            background: rgba(118,185,0,0.1);
            position: relative;
        }

        .ctrl-cuda-track .pkt {
            position: absolute;
            top: -4px;
            width: 10px;
            height: 10px;
            border-radius: 2px;
            background: #a6ff00;
            box-shadow: 0 0 8px #a6ff00;
            animation: ctrl-pkt 1.4s linear infinite;
        }

        @keyframes ctrl-pkt {
            0%   { left: -3%; opacity: 0; }
            10%  { opacity: 1; }
            90%  { opacity: 1; }
            100% { left: 100%; opacity: 0; }
        }

        /* ---------------- GPU / grid blocks ---------------- */

        .ctrl-grid-wrap {
            border: 1px solid rgba(118,185,0,0.35);
            border-radius: 12px;
            background: rgba(0,0,0,0.55);
            padding: 14px;
        }

        .ctrl-grid {
            display: grid;
            grid-template-columns: repeat(20, 1fr);
            gap: 3px;
        }

        .ctrl-block {
            width: 100%;
            padding-top: 100%;
            border-radius: 2px;
            background: rgba(118,185,0,0.12);
        }

        .ctrl-block.on {
            background: #a6ff00;
            box-shadow: 0 0 6px #a6ff00;
            animation: ctrl-flicker 1.1s ease-in-out infinite;
        }

        @keyframes ctrl-flicker { 0%, 100% { opacity: 0.35; } 50% { opacity: 1; } }

        /* ---------------- iteration timeline (SVG host) ---------------- */

        .ctrl-svg-host { width: 100%; overflow-x: auto; }

        /* ---------------- cascade (objective / residual step-down) ---------------- */

        .ctrl-cascade { display: flex; flex-direction: column; align-items: flex-start; }

        .ctrl-cascade-item {
            font-family: "JetBrains Mono", monospace;
            font-size: 15px;
            font-weight: 800;
            color: #d7e3cf;
            opacity: 0;
            animation: ctrl-eq-in 0.4s ease forwards;
        }

        .ctrl-cascade-item.final { color: #76b900; font-size: 19px; }

        .ctrl-cascade-arrow { color: #4d5a47; font-size: 12px; margin: 1px 0 1px 4px; }

        /* ---------------- race ---------------- */

        .ctrl-race-track {
            height: 26px;
            border-radius: 6px;
            background: rgba(255,255,255,0.05);
            overflow: hidden;
            margin: 4px 0 2px 0;
        }

        .ctrl-race-fill {
            height: 100%;
            border-radius: 6px;
            width: 0%;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            font-size: 11px;
            font-weight: 800;
            color: #04060a;
            animation: ctrl-grow 1.6s cubic-bezier(.15,.85,.3,1) forwards;
        }

        .ctrl-race-fill.cpu { background: linear-gradient(90deg, #555, #999); }
        .ctrl-race-fill.gpu { background: linear-gradient(90deg, #4c7a00, #a6ff00); animation-duration: 0.7s; }

        /* ---------------- benchmark console ---------------- */

        .ctrl-console {
            font-family: "JetBrains Mono", monospace;
            font-size: 12.5px;
            background: #010301;
            border: 1px solid rgba(118,185,0,0.4);
            border-radius: 10px;
            padding: 14px 16px;
            color: #b8c7ad;
            line-height: 1.65;
        }

        .ctrl-console .line {
            opacity: 0;
            transform: translateY(4px);
            animation: ctrl-eq-in 0.3s ease forwards;
            white-space: pre-wrap;
        }

        .ctrl-console .tag-info { color: #6fa8ff; }
        .ctrl-console .tag-ok   { color: #76b900; }
        .ctrl-console .tag-warn { color: #ffb020; }
        .ctrl-console .tag-demo { color: #ffb020; font-weight: 900; }

        /* ---------------- timeline bars (CPU vs GPU phases) ---------------- */

        .ctrl-tl-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
        .ctrl-tl-label { width: 46px; font-size: 11px; font-weight: 800; color: #9aa696; }
        .ctrl-tl-track { flex: 1; display: flex; height: 20px; border-radius: 4px; overflow: hidden; background: rgba(255,255,255,0.04); }
        .ctrl-tl-seg { height: 100%; }

        </style>
        """,
        unsafe_allow_html=True,
    )


def _esc(s) -> str:
    return _html.escape(str(s))


# ============================================================
# 1. HERO
# ============================================================

def render_hero(demo_mode: bool, solver_ready: bool) -> None:
    mode_txt = "DEMO MODE" if demo_mode else "LIVE MODE"
    mode_cls = "ctrl-tag-demo" if demo_mode else "ctrl-tag-live"
    status_txt = "● GPU SOLVER READY" if solver_ready else "● GPU SOLVER NOT FOUND"
    status_color = ACCENT if solver_ready else WARN

    st.markdown(
        f"""
        <div style="text-align:center; padding: 6px 0 18px 0;">
            <div style="font-size:44px; font-weight:900; letter-spacing:3px; color:{ACCENT};
                        text-shadow:0 0 14px rgba(118,185,0,0.55), 0 0 36px rgba(118,185,0,0.25);">
                ⚡ GPU OPTIMIZATION CONTROL CENTER
            </div>
            <div style="font-size:14px; letter-spacing:2px; color:#8ea184; margin-top:4px;">
                INDIGENOUS GPU-ACCELERATED PETROLEUM OPTIMIZATION ENGINE
            </div>
            <div style="margin-top:10px;">
                <span class="ctrl-tag {mode_cls}">{mode_txt}</span>
                <span class="ctrl-tag ctrl-mono" style="color:{status_color}; border:1px solid {status_color}44;">{status_txt}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 2. Persistent stage PIPELINE -- vertical, three-state, data-pipe
# connectors. THIS is the component the MASTER PROMPT's "critical
# correction" is about: a static row of labelled boxes joined by plain
# arrow glyphs is explicitly rejected. Instead:
#   - every stage node has exactly one of three states: pending (○),
#     active (● pulsing), done (✓) -- driven by `current_index`.
#   - every connector between two stages is itself the animated element:
#     - a connector whose destination stage is BEFORE current_index is
#       "done" -- a static solid line, no motion (that transfer already
#       happened, nothing to animate anymore).
#     - the ONE connector whose destination stage EQUALS current_index
#       is "active" -- this is the only connector allowed to animate at
#       any given moment (per the correction: "do not animate everything
#       at once, only the current transition"). It plays real moving
#       particles (styled per `mode` -- a travelling file icon, small
#       particle dots, converging "assemble" fragments, a single packet
#       icon, or a fast particle stream) plus a ripple ring at the
#       bottom representing the pulse-of-arrival that activates the next
#       stage.
#     - connectors whose destination is AFTER current_index are
#       "pending" -- a dim dashed line, no motion (nothing has reached
#       there yet).
# ============================================================

# (label, icon, connector_mode_into_this_stage). mode is None for the
# very first stage (nothing flows into INPUT itself).
STAGE_DEFS = [
    ("INPUT", "📄", None),
    ("ANALYSIS", "🔍", "file"),
    ("LP FORMULATION", "📐", "particles"),
    ("EQUATIONS", "🧮", "particles"),
    ("MATRIX", "🔄", "assemble"),
    ("CUDA TRANSFER", "🚀", "packet"),
    ("GPU SOLVER", "⚡", "stream"),
    ("ITERATIONS", "📈", "pulse"),
    ("SOLUTION", "🎯", "pulse"),
    ("CPU VS GPU", "⚔", "packet"),
    ("BENCHMARK", "≡", "packet"),
]


def _connector_particles(mode: str) -> str:
    if mode == "file":
        return '<div class="file-marker">📄</div>'
    if mode == "packet":
        return '<div class="packet-marker">▣</div>'
    if mode == "assemble":
        return "".join(
            f'<div class="assemble-p" style="left:{30 + i * 15}%; animation-delay:{i * 0.18:.2f}s;"></div>'
            for i in range(3)
        )
    if mode == "stream":
        return "".join(
            f'<div class="particle stream-p" style="animation-delay:{i * 0.12:.2f}s;"></div>' for i in range(5)
        )
    if mode == "pulse":
        return ""  # pulse mode relies purely on the ripple ring below
    # default: "particles"
    return "".join(
        f'<div class="particle" style="animation-delay:{i * 0.28:.2f}s;"></div>' for i in range(3)
    )


def render_stage_pipeline(current_index: int) -> None:
    """
    Render the full vertical pipeline in its CURRENT state. Call this
    again (into the same st.empty() placeholder) every time
    `current_index` advances -- each call is cheap (pure HTML/CSS
    string), so re-rendering on real stage transitions is fine; this is
    NOT a rerun-driven animation loop, the motion itself is CSS.
    """
    parts = ['<div class="ctrl-pipe">']

    for i, (label, icon, mode) in enumerate(STAGE_DEFS):
        if i > 0:
            if i < current_index:
                conn_state = "done"
            elif i == current_index:
                conn_state = "active"
            else:
                conn_state = "pending"

            particles = _connector_particles(mode) if conn_state == "active" else ""
            ripple = '<div class="ripple-ring"></div>' if conn_state == "active" else ""
            mode_tag_cls = "ctrl-pipe-mode-tag active" if conn_state == "active" else "ctrl-pipe-mode-tag"
            mode_label = {"file": "FILE MOVING", "particles": "DATA FLOW", "assemble": "ASSEMBLING",
                          "packet": "TRANSFER", "stream": "HIGH-SPEED STREAM", "pulse": "PULSE"}.get(mode, "")

            parts.append(
                f'<div class="ctrl-pipe-connector {conn_state}">{particles}{ripple}</div>'
                f'<div class="{mode_tag_cls}" style="margin-left:29px; margin-top:-26px; margin-bottom:6px;">'
                f'{mode_label if conn_state == "active" else ""}</div>'
            )

        if i < current_index:
            node_state, glyph = "done", "✓"
        elif i == current_index:
            node_state, glyph = "active", "●"
        else:
            node_state, glyph = "pending", "○"

        parts.append(
            f'<div class="ctrl-pipe-node">'
            f'<div class="ctrl-pipe-glyph {node_state}">{glyph}</div>'
            f'<div class="ctrl-pipe-label {node_state}">{icon} {_esc(label)}</div>'
            f'</div>'
        )

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def frame_start(title: str, tag_html: str = "") -> None:
    st.markdown(f'<div class="ctrl-frame"><div class="ctrl-frame-title">{_esc(title)}{tag_html}</div>', unsafe_allow_html=True)


def frame_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def demo_tag() -> str:
    return '<span class="ctrl-tag ctrl-tag-demo">DEMO / SIMULATED</span>'


def unavailable_tag() -> str:
    return '<span class="ctrl-tag ctrl-tag-unavail">UNAVAILABLE IN LIVE MODE</span>'


# ============================================================
# 3. Physical file object + travel animation
# ============================================================

def render_file_object(filename: str, size_kb: float, nnz: int) -> None:
    st.markdown(
        f"""
        <div class="ctrl-file-obj">
            <div class="fname">📄 {_esc(filename)}</div>
            <div class="fmeta">CSV &nbsp;|&nbsp; {size_kb:,.2f} KB &nbsp;|&nbsp; {nnz:,} NON-ZERO ENTRIES</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_travel_rail(start_label: str, end_label: str) -> None:
    """A physically travelling marker moving between two named stages --
    the marker's position keyframe (not just opacity) is what moves."""
    st.markdown(
        f"""
        <div class="ctrl-travel-rail">
            <div class="ctrl-travel-startlabel">{_esc(start_label)}</div>
            <div class="marker"></div>
            <div class="ctrl-travel-endlabel">{_esc(end_label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4. System profile -- animated counters via lightweight JS
# ============================================================

def render_system_profile_counters(metrics: list[tuple[str, float, str]]) -> None:
    """
    metrics: list of (label, target_number, suffix). Renders a row of
    counters that animate 0 -> target once, using a small vanilla-JS
    requestAnimationFrame loop inside an isolated component iframe --
    no Streamlit rerun involved, so it stays smooth regardless of page
    size (MASTER PROMPT section 30).
    """
    cells = []
    for label, value, suffix in metrics:
        cells.append(
            f'<div class="cell"><div class="lbl">{_esc(label)}</div>'
            f'<div class="val" data-target="{value}" data-suffix="{_esc(suffix)}">0{_esc(suffix)}</div></div>'
        )

    html_doc = f"""
    <html><head><style>
        body {{ margin:0; background:transparent; font-family:'JetBrains Mono',monospace; }}
        .row {{ display:flex; flex-wrap:wrap; gap:10px; }}
        .cell {{
            flex:1; min-width:120px; border:1px solid rgba(118,185,0,0.35);
            background:rgba(118,185,0,0.05); border-radius:10px; padding:12px; text-align:center;
        }}
        .lbl {{ font-size:10px; letter-spacing:1px; color:#9aa696; font-weight:700; }}
        .val {{ font-size:22px; font-weight:900; color:#76b900; margin-top:4px; }}
    </style></head>
    <body>
        <div class="row">{"".join(cells)}</div>
        <script>
        document.querySelectorAll('.val').forEach(function(el) {{
            var target = parseFloat(el.getAttribute('data-target'));
            var suffix = el.getAttribute('data-suffix');
            var isInt = Math.abs(target - Math.round(target)) < 1e-9;
            var start = null;
            var duration = 1100;
            function step(ts) {{
                if (!start) start = ts;
                var p = Math.min((ts - start) / duration, 1);
                var current = target * (1 - Math.pow(1 - p, 3));
                el.textContent = (isInt ? Math.round(current).toLocaleString() : current.toFixed(2)) + suffix;
                if (p < 1) requestAnimationFrame(step);
            }}
            requestAnimationFrame(step);
        }});
        </script>
    </body></html>
    """
    components.html(html_doc, height=100 + (len(metrics) // 6) * 90)


# ============================================================
# 5. Analysis sequence (staggered processing steps)
# ============================================================

def render_analysis_sequence(steps: list[str], stagger: float = 0.35) -> None:
    lines = []
    for i, s in enumerate(steps):
        lines.append(
            f'<div class="ctrl-eq" style="animation-delay:{i * stagger:.2f}s;">▸ {_esc(s)}</div>'
        )
    st.markdown("".join(lines), unsafe_allow_html=True)


# ============================================================
# 6. Labeled fill bars (tanks / bottlenecks / constraints -- shared)
# ============================================================

def render_fill_bars(rows: list[tuple[str, float, str]]) -> None:
    """rows: list of (label, pct[0-100], status_text). Severity color
    picked from pct: >=98 crit, >=80 warn, else normal."""
    out = []
    for label, pct, status in rows:
        pct = max(0.0, min(100.0, pct))
        sev = "crit" if pct >= 98 else ("warn" if pct >= 80 else "normal")
        out.append(
            f"""
            <div class="ctrl-bar-row">
                <div class="ctrl-bar-head"><span>{_esc(label)}</span><span>{pct:.0f}% &nbsp; {_esc(status)}</span></div>
                <div class="ctrl-bar-track"><div class="ctrl-bar-fill {sev}" style="--target:{pct:.1f}%;"></div></div>
            </div>
            """
        )
    st.markdown("".join(out), unsafe_allow_html=True)


# ============================================================
# 7 & 9. Transform arrow (equations->matrix, csv->LP, etc.)
# ============================================================

def render_transform_arrow(left_label: str, center_label: str, right_label: str) -> None:
    st.markdown(
        f"""
        <div class="ctrl-xform">
            <div class="ctrl-xform-node">{_esc(left_label)}</div>
            <div class="ctrl-xform-track"><div class="ripple"></div></div>
            <div class="ctrl-xform-node" style="color:#76b900;">{_esc(center_label)}</div>
            <div class="ctrl-xform-track"><div class="ripple" style="animation-delay:0.7s;"></div></div>
            <div class="ctrl-xform-node">{_esc(right_label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8. Equation reveal
# ============================================================

def render_equations(equations: list[str], stagger: float = 0.10) -> None:
    lines = []
    for i, eq in enumerate(equations):
        lines.append(f'<div class="ctrl-eq" style="animation-delay:{i * stagger:.2f}s;">{eq}</div>')
    st.markdown("".join(lines), unsafe_allow_html=True)


# ============================================================
# 9b. Real windowed matrix bracket notation
# ============================================================

def render_matrix_notation(rows, cols, values, n_rows_total: int, n_cols_total: int,
                            max_rows: int = 3, max_cols: int = 8) -> None:
    """
    Build an ASCII bracket-matrix preview using ONLY real (row, col,
    value) triples from the caller's actual sparse data -- windowed to
    `max_rows` rows and `max_cols` representative columns (first few +
    the last column that actually has data), because a 5000x10000 dense
    render is neither possible nor honest to fabricate.
    """
    by_row: dict[int, dict[int, float]] = {}
    shown_rows = sorted(set(r for r in rows if r < n_rows_total))[:max_rows]
    shown_rows_set = set(shown_rows)
    for r, c, v in zip(rows, cols, values):
        if r in shown_rows_set:
            by_row.setdefault(r, {})[c] = v

    all_cols_present = sorted(set(c for row in by_row.values() for c in row.keys()))
    if len(all_cols_present) > max_cols:
        head = all_cols_present[: max_cols - 1]
        display_cols = head + [all_cols_present[-1]]
    else:
        display_cols = all_cols_present

    def fmt(v):
        return f"{v:g}" if v is not None else "0"

    col_header = "  ".join(f"c{c}" if c != display_cols[-1] or len(display_cols) <= max_cols - 1 else "..." for c in display_cols)
    lines = ["        ┌" + " " * (8 * len(display_cols)) + "┐"]
    for i, r in enumerate(shown_rows):
        prefix = "A   =   │ " if i == 1 or (len(shown_rows) == 1 and i == 0) else "        │ "
        cells = []
        row_data = by_row.get(r, {})
        for c in display_cols:
            cells.append(f"{fmt(row_data.get(c, 0)):>6}")
        lines.append(prefix + " ".join(cells) + " │")
    if n_rows_total > max_rows:
        lines.append("        │ " + " ".join(["...".rjust(6)] * len(display_cols)) + " │")
    lines.append("        └" + " " * (8 * len(display_cols)) + "┘")

    matrix_text = "\n".join(lines)
    st.markdown(f'<div class="ctrl-matrix">{matrix_text}</div>', unsafe_allow_html=True)
    st.caption(
        f"Showing {len(shown_rows)} of {n_rows_total:,} rows, "
        f"{len(display_cols)} representative columns of {n_cols_total:,} total "
        "(real nonzero coefficients from the uploaded file)."
    )


# ============================================================
# 10. Sparse matrix heat map (real data, efficient scatter)
# ============================================================

def render_heatmap(rows, cols, values, n_rows_total: int, n_cols_total: int, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    marker_size = 0.5 if len(rows) > 20_000 else (1.2 if len(rows) > 2_000 else 4.0)
    vals = np.asarray(values, dtype=float) if len(values) else np.zeros(len(rows))
    sc = ax.scatter(cols, rows, c=vals if len(vals) else None, cmap="YlGn",
                     s=marker_size, marker="s", linewidths=0)
    if len(vals):
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(colors="#9aa696", labelsize=6)
        cb.set_label("coefficient magnitude", color="#9aa696", fontsize=7)

    ax.set_xlim(0, max(n_cols_total, 1))
    ax.set_ylim(0, max(n_rows_total, 1))
    ax.invert_yaxis()
    ax.set_xlabel("Variables (columns)", color="#9aa696", fontsize=9)
    ax.set_ylabel("Constraints (rows)", color="#9aa696", fontsize=9)
    ax.set_title(title, color=ACCENT, fontsize=11, fontweight="bold")
    ax.tick_params(colors="#667060", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a3324")

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    density = (len(rows) / (max(n_rows_total, 1) * max(n_cols_total, 1))) * 100
    st.markdown(
        f'<div style="text-align:center; font-family:JetBrains Mono,monospace;">'
        f'<span style="color:#9aa696;">{len(rows):,} NON-ZERO</span> &nbsp;|&nbsp; '
        f'<span style="color:#76b900; font-weight:900;">{100 - density:.2f}% ZERO</span> &nbsp;|&nbsp; '
        f'<span style="color:#ffb020; font-weight:900;">{density:.2f}% NON-ZERO</span></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 12. CUDA transfer -- two zones + travelling packets
# ============================================================

def render_cuda_transfer(nnz: int, size_kb: float) -> None:
    left, mid, right = st.columns([2, 1.3, 2])
    with left:
        st.markdown(
            f"""
            <div class="ctrl-zone">
                <div class="ctrl-zone-title">HOST / CPU</div>
                <div class="ctrl-zone-sub">Sparse Matrix A + b</div>
                <div class="ctrl-zone-sub">{nnz:,} non-zero entries</div>
                <div class="ctrl-zone-sub">{size_kb:,.1f} KB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mid:
        st.markdown(
            """
            <div class="ctrl-cuda-row" style="flex-direction:column; gap:6px;">
                <div style="font-size:10px; font-weight:800; color:#76b900; letter-spacing:1px;">CUDA TRANSFER</div>
                <div class="ctrl-cuda-track">
                    <div class="pkt"></div>
                    <div class="pkt" style="animation-delay:0.35s;"></div>
                    <div class="pkt" style="animation-delay:0.7s;"></div>
                    <div class="pkt" style="animation-delay:1.05s;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="ctrl-zone">
                <div class="ctrl-zone-title">DEVICE / GPU</div>
                <div class="ctrl-zone-sub">cudaMemcpy Host→Device</div>
                <div class="ctrl-zone-sub">GPU global memory</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 13. GPU solver block grid
# ============================================================

def render_gpu_grid(active: bool, n_blocks: int = 120, active_fraction: float = 0.6) -> None:
    n_active = int(n_blocks * active_fraction) if active else 0
    cells = []
    for i in range(n_blocks):
        on = active and (i % 3 != 0) and i < n_active * 1.6
        cls = "ctrl-block on" if on else "ctrl-block"
        style = f' style="animation-delay:{(i % 20) * 0.06:.2f}s;"' if on else ""
        cells.append(f'<div class="{cls}"{style}></div>')

    st.markdown(
        f"""
        <div class="ctrl-grid-wrap">
            <div class="ctrl-grid">{"".join(cells)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    label = "GPU ACTIVE — parallel kernel executing" if active else "GPU idle"
    st.caption(f"{label}  ·  visual indicator of parallel occupancy, not literal per-core telemetry")


# ============================================================
# 15. Iteration timeline -- real SVG from real (iter, objective) points
# ============================================================

def render_iteration_timeline(iterations: list[float], objectives: list[float]) -> None:
    if not iterations:
        st.info("No iteration trace available.")
        return

    W, H = 760, 190
    pad_l, pad_r, pad_t, pad_b = 40, 20, 20, 30

    it_min, it_max = min(iterations), max(iterations)
    ob_min, ob_max = min(objectives), max(objectives)
    it_span = (it_max - it_min) or 1
    ob_span = (ob_max - ob_min) or 1

    pts = []
    for i, o in zip(iterations, objectives):
        x = pad_l + (i - it_min) / it_span * (W - pad_l - pad_r)
        y = pad_t + (1 - (o - ob_min) / ob_span) * (H - pad_t - pad_b)
        pts.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    circles = []
    labels = []
    for idx, ((x, y), it, ob) in enumerate(zip(pts, iterations, objectives)):
        delay = idx * 0.18
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{ACCENT_BRIGHT}" '
            f'style="opacity:0; animation:ctrl-dot-in 0.4s ease forwards; animation-delay:{delay:.2f}s;" />'
        )
        labels.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" fill="#9aa696" font-size="9" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" '
            f'style="opacity:0; animation:ctrl-dot-in 0.4s ease forwards; animation-delay:{delay:.2f}s;">'
            f'{ob:,.0f}</text>'
        )
        labels.append(
            f'<text x="{x:.1f}" y="{H - 8}" fill="#4d5a47" font-size="8" text-anchor="middle">it {int(it)}</text>'
        )

    svg = f"""
    <div class="ctrl-svg-host">
    <svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">
        <style>
            @keyframes ctrl-dot-in {{ to {{ opacity: 1; }} }}
            @keyframes ctrl-line-in {{ to {{ stroke-dashoffset: 0; }} }}
        </style>
        <polyline points="{polyline}" fill="none" stroke="{ACCENT}" stroke-width="2"
            stroke-dasharray="2000" stroke-dashoffset="2000"
            style="animation: ctrl-line-in 1.6s ease forwards;" />
        {''.join(circles)}
        {''.join(labels)}
    </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


def render_cascade(values: list[float], fmt: str = "{:,.1f}", stagger: float = 0.22) -> None:
    parts = []
    for i, v in enumerate(values):
        is_final = i == len(values) - 1
        cls = "ctrl-cascade-item final" if is_final else "ctrl-cascade-item"
        parts.append(f'<div class="{cls}" style="animation-delay:{i * stagger:.2f}s;">{fmt.format(v)}</div>')
        if not is_final:
            parts.append('<div class="ctrl-cascade-arrow">↓</div>')
    st.markdown(f'<div class="ctrl-cascade">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_convergence_badge() -> None:
    st.markdown(
        '<div style="text-align:center; margin:14px 0;">'
        '<span class="ctrl-tag ctrl-tag-live" style="font-size:14px; padding:6px 18px;">✓ CONVERGED</span></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 20. CPU vs GPU race
# ============================================================

def render_race(cpu_ms: float, gpu_ms: float) -> None:
    max_ms = max(cpu_ms, gpu_ms, 1e-9)
    cpu_pct = cpu_ms / max_ms * 100
    gpu_pct = gpu_ms / max_ms * 100

    st.markdown(
        f"""
        <div class="ctrl-mono">
            <div style="font-size:11px; font-weight:800; color:#9aa696;">CPU</div>
            <div class="ctrl-race-track"><div class="ctrl-race-fill cpu" style="--target:{cpu_pct:.1f}%;">{cpu_ms:,.0f} ms</div></div>
            <div style="font-size:11px; font-weight:800; color:#9aa696; margin-top:10px;">GPU</div>
            <div class="ctrl-race-track"><div class="ctrl-race-fill gpu" style="--target:{gpu_pct:.1f}%;">{gpu_ms:,.0f} ms</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_speedup_badge(speedup: float) -> None:
    html_doc = f"""
    <html><head><style>
        body {{ margin:0; background:transparent; font-family:'JetBrains Mono',monospace; text-align:center; }}
        .num {{ font-size:56px; font-weight:900; color:#76b900; text-shadow:0 0 20px rgba(118,185,0,0.5); }}
        .lbl {{ font-size:12px; letter-spacing:2px; color:#9aa696; font-weight:800; }}
    </style></head>
    <body>
        <div class="num" id="n">0.00×</div>
        <div class="lbl">GPU SPEEDUP</div>
        <script>
            var target = {speedup};
            var el = document.getElementById('n');
            var start = null, duration = 1400;
            function step(ts) {{
                if (!start) start = ts;
                var p = Math.min((ts - start) / duration, 1);
                var current = target * (1 - Math.pow(1 - p, 3));
                el.textContent = current.toFixed(2) + '×';
                if (p < 1) requestAnimationFrame(step);
            }}
            requestAnimationFrame(step);
        </script>
    </body></html>
    """
    components.html(html_doc, height=110)


# ============================================================
# 21. Benchmark console
# ============================================================

def render_benchmark_console(lines: list[tuple[str, str]], stagger: float = 0.09) -> None:
    """lines: list of (tag, text). tag in {"info","ok","warn","demo",""}."""
    out = []
    for i, (tag, text) in enumerate(lines):
        cls = {"info": "tag-info", "ok": "tag-ok", "warn": "tag-warn", "demo": "tag-demo"}.get(tag, "")
        prefix = f'<span class="{cls}">[{tag.upper()}]</span> ' if tag else ""
        out.append(f'<div class="line" style="animation-delay:{i * stagger:.2f}s;">{prefix}{_esc(text)}</div>')
    st.markdown(f'<div class="ctrl-console">{"".join(out)}</div>', unsafe_allow_html=True)


# ============================================================
# 22. Benchmark timeline (phase-segmented)
# ============================================================

def render_benchmark_timeline(cpu_ms: float, gpu_phases: list[tuple[str, float, str]]) -> None:
    """gpu_phases: list of (phase_label, ms, color_hex)."""
    total_gpu = sum(p[1] for p in gpu_phases) or 1e-9

    st.markdown(
        f"""
        <div class="ctrl-tl-row">
            <div class="ctrl-tl-label">CPU</div>
            <div class="ctrl-tl-track"><div class="ctrl-tl-seg" style="width:100%; background:linear-gradient(90deg,#555,#999);"></div></div>
            <div style="width:70px; font-size:11px; color:#9aa696;">{cpu_ms:,.1f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    segs = "".join(
        f'<div class="ctrl-tl-seg" style="width:{(ms / total_gpu) * 100:.1f}%; background:{color};" title="{_esc(label)}: {ms:.2f} ms"></div>'
        for label, ms, color in gpu_phases
    )
    st.markdown(
        f"""
        <div class="ctrl-tl-row">
            <div class="ctrl-tl-label">GPU</div>
            <div class="ctrl-tl-track">{segs}</div>
            <div style="width:70px; font-size:11px; color:#9aa696;">{total_gpu:,.1f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    legend = "  ".join(f'<span style="color:{c};">■</span> {_esc(l)}' for l, _, c in gpu_phases)
    st.markdown(f'<div style="font-size:10px; color:#9aa696;">{legend}</div>', unsafe_allow_html=True)
