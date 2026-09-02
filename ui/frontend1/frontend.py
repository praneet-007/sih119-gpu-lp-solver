import streamlit as st
import subprocess
import sys
import tempfile
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Refinery GPU LP Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent
CONVERTER = ROOT / "mps_converter.py"

if os.name == "nt":
    SOLVER = ROOT / "gpu_solver.exe"
else:
    SOLVER = ROOT / "gpu_solver"


# ==========================================================
# SESSION STATE INITIALIZATION
# ==========================================================

_defaults = {
    "result": None,
    "raw_output": "",
    "history": [],
    "run_count": 0,
    "last_run_time": None,
}

for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================================
# GLOBAL CSS — GLOWING / FLOATING / SHINING THEME
# ==========================================================

st.markdown(
    """
    <style>
    /* ====================================================
       ROOT VARIABLES
       ==================================================== */
    :root {
        --nv-green: #76b900;
        --nv-green-bright: #9be03c;
        --nv-green-dim: rgba(118,185,0,0.12);
        --nv-green-glow: rgba(118,185,0,0.55);
        --surface: #060a06;
        --surface-card: rgba(10,16,10,0.88);
        --text-primary: #f0f4ec;
        --text-secondary: #8f9b8a;
    }

    /* ====================================================
       ANIMATED BACKGROUND — AURORA / SHINING EFFECT
       ==================================================== */
    @keyframes aurora {
        0%   { background-position: 0% 50%; }
        25%  { background-position: 100% 0%; }
        50%  { background-position: 100% 100%; }
        75%  { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes float-particle {
        0%, 100% { transform: translateY(0px) scale(1); opacity: 0.3; }
        50%      { transform: translateY(-30px) scale(1.15); opacity: 0.7; }
    }

    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 15px rgba(118,185,0,0.25), 0 0 45px rgba(118,185,0,0.08); }
        50%      { box-shadow: 0 0 25px rgba(118,185,0,0.45), 0 0 80px rgba(118,185,0,0.15); }
    }

    @keyframes btn-glow {
        0%, 100% {
            box-shadow: 0 0 12px rgba(118,185,0,0.4),
                        0 0 35px rgba(118,185,0,0.15),
                        inset 0 1px 0 rgba(255,255,255,0.15);
        }
        50% {
            box-shadow: 0 0 22px rgba(118,185,0,0.7),
                        0 0 60px rgba(118,185,0,0.25),
                        0 0 100px rgba(118,185,0,0.10),
                        inset 0 1px 0 rgba(255,255,255,0.2);
        }
    }

    @keyframes border-flow {
        0%   { border-color: rgba(118,185,0,0.25); }
        50%  { border-color: rgba(118,185,0,0.55); }
        100% { border-color: rgba(118,185,0,0.25); }
    }

    @keyframes float-badge {
        0%, 100% { transform: translateY(0px); }
        50%      { transform: translateY(-6px); }
    }

    .stApp {
        background:
            radial-gradient(ellipse at 20% 0%, rgba(118,185,0,0.10), transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(118,185,0,0.06), transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(20,40,10,0.15), transparent 60%),
            linear-gradient(135deg, #020402 0%, #050a05 30%, #030503 60%, #010201 100%);
        background-size: 200% 200%;
        animation: aurora 20s ease infinite;
        color: var(--text-primary);
        min-height: 100vh;
    }

    /* Floating particles overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(2px 2px at 15% 25%, rgba(118,185,0,0.35), transparent),
            radial-gradient(2px 2px at 45% 65%, rgba(118,185,0,0.25), transparent),
            radial-gradient(3px 3px at 75% 15%, rgba(118,185,0,0.30), transparent),
            radial-gradient(2px 2px at 85% 80%, rgba(118,185,0,0.20), transparent),
            radial-gradient(2px 2px at 30% 90%, rgba(118,185,0,0.28), transparent),
            radial-gradient(3px 3px at 60% 40%, rgba(118,185,0,0.18), transparent),
            radial-gradient(2px 2px at 10% 70%, rgba(118,185,0,0.22), transparent),
            radial-gradient(2px 2px at 90% 45%, rgba(118,185,0,0.15), transparent);
        animation: float-particle 8s ease-in-out infinite;
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1rem;
        padding-bottom: 3rem;
        position: relative;
        z-index: 1;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ====================================================
       SIDEBAR — GLASS MORPHISM
       ==================================================== */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg,
                rgba(5,10,5,0.97),
                rgba(2,4,2,0.99)) !important;
        border-right: 1px solid rgba(118,185,0,0.22) !important;
        backdrop-filter: blur(12px);
    }

    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--nv-green);
        text-shadow: 0 0 20px rgba(118,185,0,0.3);
    }

    /* ====================================================
       HERO BANNER — GLOWING + FLOATING
       ==================================================== */
    .hero {
        text-align: center;
        padding: 50px 30px 42px;
        margin-bottom: 35px;
        border-radius: 28px;
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg,
                rgba(118,185,0,0.12) 0%,
                rgba(0,0,0,0.85) 40%,
                rgba(118,185,0,0.06) 100%);
        border: 1px solid rgba(118,185,0,0.30);
        animation: pulse-glow 4s ease-in-out infinite;
    }

    /* Shimmer sweep across hero */
    .hero::before {
        content: "";
        position: absolute;
        top: 0; left: -100%; right: -100%; bottom: 0;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(118,185,0,0.06) 45%,
            rgba(118,185,0,0.12) 50%,
            rgba(118,185,0,0.06) 55%,
            transparent 100%
        );
        background-size: 200% 100%;
        animation: shimmer 6s linear infinite;
        pointer-events: none;
    }

    .hero-title {
        color: var(--nv-green);
        font-size: 48px;
        font-weight: 950;
        letter-spacing: 3px;
        line-height: 1.1;
        text-shadow: 0 0 40px rgba(118,185,0,0.35),
                     0 0 80px rgba(118,185,0,0.12);
        position: relative;
    }

    .hero-subtitle {
        color: #aeb8a7;
        font-size: 13px;
        letter-spacing: 2px;
        margin-top: 16px;
        text-transform: uppercase;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 22px;
        padding: 10px 26px;
        border-radius: 28px;
        border: 1px solid rgba(118,185,0,0.45);
        background: rgba(118,185,0,0.10);
        color: var(--nv-green-bright);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1.2px;
        animation: float-badge 3s ease-in-out infinite;
        position: relative;
    }

    .hero-version {
        display: inline-block;
        margin-top: 10px;
        color: #5a6e4e;
        font-size: 10px;
        letter-spacing: 1px;
    }

    /* ====================================================
       SECTION HEADERS — GREEN ACCENT
       ==================================================== */
    .section-header {
        margin-top: 35px;
        margin-bottom: 6px;
        font-size: 22px;
        font-weight: 900;
        color: var(--nv-green);
        border-left: 4px solid var(--nv-green);
        padding-left: 14px;
        text-shadow: 0 0 18px rgba(118,185,0,0.18);
    }

    .section-sub {
        color: var(--text-secondary);
        font-size: 13px;
        margin-bottom: 18px;
        padding-left: 18px;
    }

    /* ====================================================
       METRIC CARDS — GLASS + GLOW
       ==================================================== */
    .metric-card {
        min-height: 115px;
        padding: 18px 20px;
        border-radius: 18px;
        background:
            linear-gradient(145deg,
                rgba(118,185,0,0.09),
                rgba(10,14,10,0.92));
        border: 1px solid rgba(118,185,0,0.22);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.04),
            0 8px 30px rgba(0,0,0,0.30);
        animation: border-flow 5s ease-in-out infinite;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow:
            0 0 20px rgba(118,185,0,0.25),
            0 12px 40px rgba(0,0,0,0.35);
    }

    .metric-label {
        color: #7a8c72;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--nv-green);
        font-size: 26px;
        font-weight: 950;
        margin-top: 8px;
        white-space: nowrap;
        text-shadow: 0 0 12px rgba(118,185,0,0.15);
    }

    .metric-unit {
        color: #5a6e4e;
        font-size: 12px;
        font-weight: 600;
    }

    /* ====================================================
       STATUS CARDS
       ==================================================== */
    .status-card {
        padding: 16px 20px;
        border-radius: 16px;
        background:
            linear-gradient(90deg,
                rgba(118,185,0,0.10),
                rgba(0,0,0,0.55));
        border: 1px solid rgba(118,185,0,0.25);
        animation: border-flow 4s ease-in-out infinite;
    }

    .status-online {
        color: var(--nv-green-bright);
        font-weight: 900;
        text-shadow: 0 0 8px rgba(118,185,0,0.4);
    }

    /* ====================================================
       INFO BOX
       ==================================================== */
    .info-box {
        padding: 20px 22px;
        border-radius: 16px;
        background: rgba(118,185,0,0.045);
        border: 1px solid rgba(118,185,0,0.18);
        color: #b7c2b1;
        font-size: 13px;
        line-height: 1.7;
    }

    /* ====================================================
       FILE UPLOADER — GLOWING BORDER
       ==================================================== */
    [data-testid="stFileUploader"] {
        background:
            linear-gradient(135deg,
                rgba(118,185,0,0.06),
                rgba(0,0,0,0.50));
        border: 2px dashed rgba(118,185,0,0.45) !important;
        border-radius: 20px;
        padding: 14px;
        animation: border-flow 4s ease-in-out infinite;
        transition: border-color 0.3s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(118,185,0,0.7) !important;
        box-shadow: 0 0 25px rgba(118,185,0,0.12);
    }

    /* ====================================================
       BUTTONS — MASSIVE GLOW ANIMATION
       ==================================================== */
    div.stButton > button {
        min-height: 58px;
        background:
            linear-gradient(90deg,
                #76b900 0%,
                #8dd620 50%,
                #9be03c 100%) !important;
        color: #050805 !important;
        border: none !important;
        border-radius: 14px;
        font-weight: 950;
        font-size: 16px;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: btn-glow 2.5s ease-in-out infinite;
        transition: transform 0.15s, filter 0.15s;
        position: relative;
        overflow: hidden;
    }

    div.stButton > button::before {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 200%; height: 100%;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255,255,255,0.15) 50%,
            transparent 100%
        );
        animation: shimmer 3s linear infinite;
        pointer-events: none;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        filter: brightness(1.15);
    }

    div.stButton > button:active {
        transform: translateY(1px) scale(0.99);
    }

    /* ====================================================
       DOWNLOAD BUTTON — FLOATING GLOW
       ==================================================== */
    div.stDownloadButton > button {
        background: linear-gradient(90deg,
            rgba(118,185,0,0.18),
            rgba(118,185,0,0.08)) !important;
        color: var(--nv-green-bright) !important;
        border: 1px solid rgba(118,185,0,0.35) !important;
        border-radius: 12px;
        font-weight: 800;
        animation: border-flow 4s ease-in-out infinite;
        transition: transform 0.15s;
    }

    div.stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(118,185,0,0.2);
    }

    /* ====================================================
       DATA FRAMES
       ==================================================== */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(118,185,0,0.12);
    }

    /* ====================================================
       EXPANDERS
       ==================================================== */
    details {
        border: 1px solid rgba(118,185,0,0.15) !important;
        border-radius: 14px !important;
        background: rgba(5,10,5,0.6) !important;
    }

    /* ====================================================
       NUMBER INPUTS & SELECTS
       ==================================================== */
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] > div {
        background: rgba(10,16,10,0.7) !important;
        border-color: rgba(118,185,0,0.2) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    /* ====================================================
       FLOATING ACTION INDICATOR
       ==================================================== */
    .floating-indicator {
        position: fixed;
        bottom: 30px;
        right: 30px;
        padding: 14px 22px;
        border-radius: 20px;
        background: linear-gradient(135deg,
            rgba(118,185,0,0.15),
            rgba(0,0,0,0.85));
        border: 1px solid rgba(118,185,0,0.40);
        color: var(--nv-green-bright);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        z-index: 9999;
        animation: float-badge 3s ease-in-out infinite,
                   pulse-glow 3s ease-in-out infinite;
        backdrop-filter: blur(10px);
    }

    /* ====================================================
       SCROLLBAR
       ==================================================== */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #020402;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(118,185,0,0.25);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(118,185,0,0.45);
    }

    /* ====================================================
       PLOTLY CHART CONTAINERS
       ==================================================== */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(118,185,0,0.10);
    }

    /* ====================================================
       SIDEBAR PIPELINE — VERTICAL TIMELINE
       ==================================================== */
    @keyframes pipeline-pulse {
        0%, 100% { box-shadow: 0 0 6px rgba(118,185,0,0.3); }
        50%      { box-shadow: 0 0 14px rgba(118,185,0,0.7), 0 0 28px rgba(118,185,0,0.2); }
    }

    @keyframes line-flow {
        0%   { background-position: 0 0; }
        100% { background-position: 0 20px; }
    }

    .pipeline-wrap {
        padding: 8px 0 4px 0;
    }

    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        position: relative;
        padding-bottom: 0;
    }

    /* Glowing dot */
    .pipeline-dot {
        flex-shrink: 0;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: radial-gradient(circle,
            #9be03c 0%,
            #76b900 60%,
            rgba(118,185,0,0.5) 100%);
        border: 2px solid rgba(118,185,0,0.6);
        margin-top: 3px;
        animation: pipeline-pulse 3s ease-in-out infinite;
        position: relative;
        z-index: 2;
    }

    .pipeline-step:nth-child(1) .pipeline-dot { animation-delay: 0s; }
    .pipeline-step:nth-child(2) .pipeline-dot { animation-delay: 0.4s; }
    .pipeline-step:nth-child(3) .pipeline-dot { animation-delay: 0.8s; }
    .pipeline-step:nth-child(4) .pipeline-dot { animation-delay: 1.2s; }
    .pipeline-step:nth-child(5) .pipeline-dot { animation-delay: 1.6s; }
    .pipeline-step:nth-child(6) .pipeline-dot { animation-delay: 2.0s; }

    /* Content */
    .pipeline-content {
        flex: 1;
        padding-bottom: 18px;
    }

    .pipeline-num {
        display: inline-block;
        background: rgba(118,185,0,0.15);
        color: #9be03c;
        font-size: 9px;
        font-weight: 900;
        padding: 2px 7px;
        border-radius: 6px;
        letter-spacing: 0.8px;
        margin-bottom: 3px;
        border: 1px solid rgba(118,185,0,0.25);
    }

    .pipeline-title {
        color: #e8f0e0;
        font-size: 12.5px;
        font-weight: 700;
        line-height: 1.35;
        margin-top: 2px;
    }

    .pipeline-desc {
        color: #6b7a62;
        font-size: 10px;
        margin-top: 2px;
        line-height: 1.3;
    }

    /* Animated connecting line */
    .pipeline-connector {
        width: 2px;
        height: 16px;
        margin-left: 8px;
        margin-bottom: 0;
        background:
            repeating-linear-gradient(
                to bottom,
                rgba(118,185,0,0.5) 0px,
                rgba(118,185,0,0.5) 4px,
                transparent 4px,
                transparent 8px
            );
        background-size: 2px 20px;
        animation: line-flow 1.5s linear infinite;
        position: relative;
        z-index: 1;
    }

    /* Hover lift on steps */
    .pipeline-step:hover .pipeline-content {
        transform: translateX(3px);
        transition: transform 0.2s;
    }
    .pipeline-step:hover .pipeline-dot {
        box-shadow: 0 0 18px rgba(118,185,0,0.8), 0 0 35px rgba(118,185,0,0.3);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


def render_metric(label: str, value: str, unit: str = ""):
    """Render a glowing metric card."""
    unit_html = f'<span class="metric-unit"> {unit}</span>' if unit else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}{unit_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_result(result):
    """Validate the solver JSON output structure."""
    if not isinstance(result, dict):
        raise ValueError("Solver output is not a valid JSON object.")
    if "matrix" not in result:
        raise ValueError("Solver JSON is missing the 'matrix' key.")
    if "performance" not in result:
        raise ValueError("Solver JSON is missing the 'performance' key.")
    return True


def plotly_dark_layout(fig, height=480, **kwargs):
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        showlegend=kwargs.get("showlegend", False),
        margin=kwargs.get("margin", dict(l=60, r=30, t=80, b=100)),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(118,185,0,0.06)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(118,185,0,0.06)",
        zeroline=False,
    )
    return fig


# ==========================================================
# LIVE SUBPROCESS EXECUTION  (Popen streaming)
# ==========================================================


def execute_live(command, output_placeholder):
    """
    Run a subprocess with Popen and stream stdout lines
    into a Streamlit st.code() block in real-time.
    """
    lines = []

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Executable not found: {command[0]}\n"
            "Ensure the solver binary is compiled and in the project root."
        )
    except PermissionError:
        raise RuntimeError(
            f"Permission denied executing: {command[0]}\n"
            "Check file permissions (chmod +x on Linux/WSL)."
        )

    while True:
        line = process.stdout.readline()
        if line:
            line = line.rstrip()
            lines.append(line)
            # Show last 200 lines to keep the console manageable
            output_placeholder.code(
                "\n".join(lines[-200:]),
                language="text",
            )
        elif process.poll() is not None:
            break
        time.sleep(0.01)

    return_code = process.wait()
    return return_code, "\n".join(lines)


# ==========================================================
# COMPLETE PIPELINE
# ==========================================================


def run_pipeline(uploaded_file, console_placeholder, progress_bar):
    """
    Full pipeline:
      1. Save uploaded MPS file
      2. Convert MPS → .lpdata via Python converter
      3. Execute C++/CUDA solver with live streaming
      4. Parse JSON output
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        temp = Path(temp_dir)
        mps_file = temp / uploaded_file.name
        lp_file = temp / "solver_input.lpdata"
        output_file = temp / "solver_output.json"

        # ── Save uploaded MPS ─────────────────────────────
        uploaded_file.seek(0)
        mps_file.write_bytes(uploaded_file.getvalue())

        # ── STEP 1: MPS Conversion ───────────────────────
        if not CONVERTER.exists():
            raise RuntimeError(f"Missing converter:\n{CONVERTER}")

        progress_bar.progress(8)
        console_placeholder.code(
            "╔══════════════════════════════════════════════╗\n"
            "║    ⚡ REFINERY GPU OPTIMIZATION ENGINE ⚡    ║\n"
            "╚══════════════════════════════════════════════╝\n\n"
            "▶ Step 1/3 — Parsing MPS model...\n"
            f"  Input:  {uploaded_file.name}\n"
            f"  Output: solver_input.lpdata\n",
            language="text",
        )

        converter = subprocess.run(
            [sys.executable, str(CONVERTER), str(mps_file), str(lp_file)],
            capture_output=True,
            text=True,
            errors="replace",
        )

        converter_output = converter.stdout or ""
        if converter_output:
            console_placeholder.code(converter_output, language="text")

        if converter.returncode != 0:
            raise RuntimeError(
                converter.stderr or converter.stdout or "MPS conversion failed."
            )

        if not lp_file.exists():
            raise RuntimeError(
                "MPS converter completed but solver_input.lpdata was not created."
            )

        # ── STEP 2: CUDA Solver ──────────────────────────
        if not SOLVER.exists():
            raise RuntimeError(f"Missing solver binary:\n{SOLVER}")

        progress_bar.progress(30)
        console_placeholder.code(
            "✓ Step 1/3 — MPS model converted successfully.\n\n"
            "▶ Step 2/3 — Launching C++ / CUDA engine...\n"
            f"  Binary:  {SOLVER.name}\n"
            f"  Input:   solver_input.lpdata\n"
            f"  Output:  solver_output.json\n",
            language="text",
        )

        return_code, raw_output = execute_live(
            [str(SOLVER), str(lp_file), str(output_file)],
            console_placeholder,
        )

        if return_code != 0:
            raise RuntimeError(f"CUDA solver exited with code {return_code}.\n\n{raw_output}")

        # ── STEP 3: Parse JSON ───────────────────────────
        progress_bar.progress(80)
        console_placeholder.code(
            raw_output + "\n\n▶ Step 3/3 — Loading solver results...\n",
            language="text",
        )

        if not output_file.exists():
            raise RuntimeError(
                "CUDA solver completed but solver_output.json was not generated."
            )

        try:
            result = json.loads(output_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid solver JSON:\n{exc}")

        validate_result(result)

        progress_bar.progress(100)
        console_placeholder.code(
            raw_output + "\n\n✓ Optimization completed successfully. ⚡\n",
            language="text",
        )

        return result, raw_output


# ==========================================================
#                        HERO BANNER
# ==========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">⚡ REFINERY GPU LP OPTIMIZER</div>
        <div class="hero-subtitle">
            LARGE-SCALE LINEAR PROGRAMMING &nbsp;•&nbsp;
            SPARSE CSR MATRIX COMPRESSION &nbsp;•&nbsp;
            CUDA ACCELERATION
        </div>
        <div class="hero-badge">
            MPS &nbsp;→&nbsp; PYTHON CONVERTER &nbsp;→&nbsp;
            C++ / CUDA ENGINE &nbsp;→&nbsp; GPU KERNEL &nbsp;→&nbsp;
            JSON &nbsp;→&nbsp; ANALYTICS
        </div>
        <br>
        <div class="hero-version">
            PETROLEUM REFINERY BLEND OPTIMIZATION &nbsp;|&nbsp;
            RTX ACCELERATED
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SIDEBAR — CONTROL CENTER
# ==========================================================

with st.sidebar:

    st.markdown("## ⚡ CONTROL CENTER")

    # ── System Status ─────────────────────────────────────
    st.markdown("### 🔧 SYSTEM STATUS")

    if CONVERTER.exists():
        st.success("✓ Python MPS converter ready")
    else:
        st.error("✗ mps_converter.py missing")

    if SOLVER.exists():
        st.success(f"✓ {SOLVER.name} ready")
    else:
        st.error(f"✗ {SOLVER.name} missing")

    st.divider()

    # ── Execution Pipeline ────────────────────────────────
    st.markdown("### 🔄 EXECUTION PIPELINE")
    st.markdown(
'<div class="pipeline-wrap">'
'<div class="pipeline-step">'
'<div class="pipeline-dot"></div>'
'<div class="pipeline-content">'
'<span class="pipeline-num">STEP 01</span>'
'<div class="pipeline-title">📄 MPS Model Upload</div>'
'<div class="pipeline-desc">Drag-and-drop refinery .mps file</div>'
'</div></div>'
'<div class="pipeline-connector"></div>'
'<div class="pipeline-step">'
'<div class="pipeline-dot"></div>'
'<div class="pipeline-content">'
'<span class="pipeline-num">STEP 02</span>'
'<div class="pipeline-title">🐍 Python MPS Converter</div>'
'<div class="pipeline-desc">MPS → .lpdata matrix format</div>'
'</div></div>'
'<div class="pipeline-connector"></div>'
'<div class="pipeline-step">'
'<div class="pipeline-dot"></div>'
'<div class="pipeline-content">'
'<span class="pipeline-num">STEP 03</span>'
'<div class="pipeline-title">⚙️ C++ / CUDA Engine</div>'
'<div class="pipeline-desc">Sparse CSR optimization solver</div>'
'</div></div>'
'<div class="pipeline-connector"></div>'
'<div class="pipeline-step">'
'<div class="pipeline-dot"></div>'
'<div class="pipeline-content">'
'<span class="pipeline-num">STEP 04</span>'
'<div class="pipeline-title">🖥️ NVIDIA GPU Execution</div>'
'<div class="pipeline-desc">RTX CUDA kernel acceleration</div>'
'</div></div>'
'<div class="pipeline-connector"></div>'
'<div class="pipeline-step">'
'<div class="pipeline-dot"></div>'
'<div class="pipeline-content">'
'<span class="pipeline-num">STEP 05</span>'
'<div class="pipeline-title">📦 JSON Result Output</div>'
'<div class="pipeline-desc">Structured solver results</div>'
'</div></div>'
'<div class="pipeline-connector"></div>'
'<div class="pipeline-step">'
'<div class="pipeline-dot"></div>'
'<div class="pipeline-content">'
'<span class="pipeline-num">STEP 06</span>'
'<div class="pipeline-title">📊 Analytics Dashboard</div>'
'<div class="pipeline-desc">Interactive charts &amp; benchmarks</div>'
'</div></div>'
'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Engineering Capabilities ──────────────────────────
    st.markdown("### 🏗️ SYSTEMS ENGINEERING")
    capabilities = [
        "Sparse matrix analysis",
        "CSR memory compression",
        "NNZ / sparsity profiling",
        "CUDA kernel execution",
        "Live console streaming",
        "Matrix constraint heatmap",
        "Tank capacity gauges",
        "Bottleneck penalty charts",
        "CPU vs GPU benchmarking",
        "Session history tracking",
        "CSV benchmark export",
    ]
    for cap in capabilities:
        st.caption(f"✓ {cap}")

    st.divider()

    # ── Run Statistics ────────────────────────────────────
    st.markdown("### 📈 SESSION STATS")
    st.caption(f"Total runs: **{st.session_state.run_count}**")
    if st.session_state.last_run_time:
        st.caption(f"Last run: **{st.session_state.last_run_time}**")
    st.caption(
        f"History entries: **{len(st.session_state.history)}**"
    )

    st.divider()
    st.caption("⚡ Refinery GPU LP Optimization Engine")


# ==========================================================
#  FLOATING STATUS INDICATOR (bottom-right)
# ==========================================================

if SOLVER.exists() and CONVERTER.exists():
    indicator_text = "⚡ ENGINE ONLINE"
else:
    indicator_text = "⚠ ENGINE OFFLINE"

st.markdown(
    f'<div class="floating-indicator">{indicator_text}</div>',
    unsafe_allow_html=True,
)


# ==========================================================
# MODEL INPUT — FILE UPLOADER
# ==========================================================

st.markdown(
    '<div class="section-header">📄 MODEL INPUT</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-sub">'
    "Upload a standard MPS refinery optimization model to begin analysis."
    "</div>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload LP Matrix (.mps)",
    type=["mps", "txt"],
    label_visibility="collapsed",
)

if uploaded:
    file_size_kb = len(uploaded.getvalue()) / 1024

    st.markdown(
        f"""
        <div class="status-card">
            <span class="status-online">● MODEL LOADED</span>
            &nbsp;&nbsp;
            <strong>{uploaded.name}</strong>
            &nbsp;&nbsp;
            <span class="metric-unit">({file_size_kb:.1f} KB)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button("⚡ RUN GPU OPTIMIZATION", use_container_width=True):

        console_placeholder = st.empty()
        progress = st.progress(0)

        try:
            with st.status(
                "⚡ Running refinery optimization pipeline...",
                expanded=True,
            ):
                result, raw_output = run_pipeline(
                    uploaded, console_placeholder, progress
                )

            st.session_state.result = result
            st.session_state.raw_output = raw_output
            st.session_state.run_count += 1
            st.session_state.last_run_time = datetime.now().strftime(
                "%H:%M:%S"
            )

            # ── Build history entry ───────────────────────
            matrix = result.get("matrix", {})
            performance = result.get("performance", {})

            history_entry = {
                "Run": st.session_state.run_count,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model": uploaded.name,
                "Rows": safe_int(matrix.get("rows")),
                "Columns": safe_int(matrix.get("cols")),
                "NNZ": safe_int(matrix.get("nnz")),
                "Sparsity %": round(safe_float(matrix.get("sparsity_percent")), 2),
                "CSR MB": round(safe_float(matrix.get("csr_mb")), 4),
                "RAM Saved MB": round(safe_float(matrix.get("ram_saved_mb")), 4),
                "Objective": round(
                    safe_float(result.get("objective_value")), 6
                ),
                "CPU ms": round(
                    safe_float(performance.get("cpu_time_ms")), 2
                ),
                "GPU ms": round(
                    safe_float(performance.get("gpu_compute_time_ms")), 2
                ),
                "Total ms": round(
                    safe_float(performance.get("total_time_ms")), 2
                ),
                "Speedup": round(
                    safe_float(performance.get("speedup")), 2
                ),
            }

            st.session_state.history.append(history_entry)
            st.rerun()

        except Exception as exc:
            st.error(f"🚨 Engine failure: {exc}")
            with st.expander("Show technical error details"):
                st.exception(exc)


# ==========================================================
# ████  RESULTS DASHBOARD  ████
# ==========================================================

if st.session_state.result:

    result = st.session_state.result
    matrix = result.get("matrix", {})
    performance = result.get("performance", {})

    # ===========================================================
    # 1.  MATRIX & HARDWARE PROFILE
    # ===========================================================

    st.markdown(
        '<div class="section-header">🖥️ MATRIX &amp; SYSTEM PROFILE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "Matrix dimensions, sparsity, CSR compression metrics, and total execution time."
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    metrics_data = [
        ("MATRIX ROWS", f"{safe_int(matrix.get('rows')):,}", ""),
        ("MATRIX COLUMNS", f"{safe_int(matrix.get('cols')):,}", ""),
        ("NON-ZERO VALUES", f"{safe_int(matrix.get('nnz')):,}", ""),
        ("SPARSITY", f"{safe_float(matrix.get('sparsity_percent')):.2f}", "%"),
        ("CSR MEMORY", f"{safe_float(matrix.get('csr_mb')):.4f}", "MB"),
        (
            "TOTAL TIME",
            f"{safe_float(performance.get('total_time_ms')):.2f}",
            "ms",
        ),
    ]
    for col, (label, val, unit) in zip(cols, metrics_data):
        with col:
            render_metric(label, val, unit)

    # ── CSR Memory Savings Info Box ───────────────────────
    dense_mb = safe_float(matrix.get("dense_mb"))
    csr_mb = safe_float(matrix.get("csr_mb"))
    saved_mb = safe_float(matrix.get("ram_saved_mb"))
    saved_percent = safe_float(matrix.get("ram_saved_percent"))

    st.markdown(
        f"""
        <div class="info-box">
            <b>⚡ CSR Compression Analysis:</b><br><br>
            Dense array: <b>{dense_mb:.4f} MB</b>
            &nbsp;→&nbsp;
            CSR sparse: <b>{csr_mb:.4f} MB</b>
            &nbsp;→&nbsp;
            <span style="color: var(--nv-green-bright);">
                RAM saved: <b>{saved_mb:.4f} MB</b>
                ({saved_percent:.2f}% reduction)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ===========================================================
    # 2.  OPTIMIZATION SUMMARY
    # ===========================================================

    st.markdown(
        '<div class="section-header">🎯 OPTIMIZATION SUMMARY</div>',
        unsafe_allow_html=True,
    )

    objective = safe_float(result.get("objective_value"))
    cpu_time = safe_float(performance.get("cpu_time_ms"))
    gpu_time = safe_float(performance.get("gpu_compute_time_ms"))
    total_time = safe_float(performance.get("total_time_ms"))
    speedup = safe_float(performance.get("speedup"))

    cols = st.columns(5)
    summary_data = [
        ("OBJECTIVE VALUE", f"{objective:,.4f}", ""),
        ("CPU REFERENCE", f"{cpu_time:.2f}", "ms"),
        ("GPU KERNEL", f"{gpu_time:.2f}", "ms"),
        ("END-TO-END", f"{total_time:.2f}", "ms"),
        ("SPEEDUP", f"{speedup:.2f}", "×"),
    ]
    for col, (label, val, unit) in zip(cols, summary_data):
        with col:
            render_metric(label, val, unit)

    status = result.get("status", "COMPLETED")
    if str(status).upper() in ["OPTIMAL", "SUCCESS", "COMPLETED"]:
        st.success(f"✓ Optimization status: **{status}**")
    else:
        st.warning(f"⚠ Optimization status: **{status}**")

    # ── CPU vs GPU Objective Accuracy ─────────────────────
    cpu_objective = performance.get("cpu_objective")
    gpu_objective = performance.get("gpu_objective", objective)

    if cpu_objective is not None:
        st.markdown(
            '<div class="section-header">🎯 CPU vs GPU ACCURACY PROOF</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric("CPU OBJECTIVE", f"{safe_float(cpu_objective):,.6f}")
        with c2:
            render_metric("GPU OBJECTIVE", f"{safe_float(gpu_objective):,.6f}")
        diff = abs(safe_float(cpu_objective) - safe_float(gpu_objective))
        with c3:
            render_metric("ABSOLUTE DIFF", f"{diff:.8f}")

    # ===========================================================
    # 3.  TANK CAPACITY UTILIZATION  (Primal Variables x)
    # ===========================================================

    primal = result.get("primal_variables", [])

    if primal:
        st.markdown(
            '<div class="section-header">🏭 TANK CAPACITY UTILIZATION</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-sub">'
            "Decision variables (x) visualized as refinery tank capacity gauges. "
            "Higher utilization indicates stronger resource commitment."
            "</div>",
            unsafe_allow_html=True,
        )

        xdf = pd.DataFrame(primal)

        if "name" in xdf.columns and "utilization_percent" in xdf.columns:
            xdf["utilization_percent"] = pd.to_numeric(
                xdf["utilization_percent"], errors="coerce"
            )

            chart_df = (
                xdf.dropna(subset=["utilization_percent"])
                .sort_values("utilization_percent", ascending=False)
                .head(30)
            )

            if not chart_df.empty:
                # ── Gauge-style bar chart ─────────────────
                fig = go.Figure()

                colors = [
                    f"rgba({max(0, int(155 - v*0.8))}, "
                    f"{min(255, int(100 + v*1.5))}, "
                    f"0, 0.85)"
                    for v in chart_df["utilization_percent"]
                ]

                fig.add_trace(
                    go.Bar(
                        x=chart_df["name"],
                        y=chart_df["utilization_percent"],
                        text=[f"{v:.1f}%" for v in chart_df["utilization_percent"]],
                        textposition="outside",
                        marker=dict(
                            color=colors,
                            line=dict(
                                color="rgba(118,185,0,0.6)", width=1
                            ),
                        ),
                        hovertemplate=(
                            "<b>Tank:</b> %{x}<br>"
                            "<b>Utilization:</b> %{y:.2f}%"
                            "<extra></extra>"
                        ),
                    )
                )

                # Capacity ceiling line
                fig.add_hline(
                    y=100,
                    line_dash="dot",
                    line_color="rgba(255,80,80,0.5)",
                    annotation_text="100% Capacity",
                    annotation_position="top right",
                    annotation_font_color="rgba(255,80,80,0.7)",
                )

                fig.update_yaxes(
                    range=[
                        0,
                        max(115, float(chart_df["utilization_percent"].max()) + 12),
                    ]
                )

                fig.update_layout(
                    title=dict(
                        text="Refinery Tank Capacity Utilization (%)",
                        font=dict(size=16),
                    ),
                )
                fig.update_xaxes(tickangle=-45)
                plotly_dark_layout(fig, height=520)

                st.plotly_chart(fig, use_container_width=True)

        # ── Data table ────────────────────────────────────
        table_cols = ["name", "value", "capacity", "utilization_percent"]
        existing = [c for c in table_cols if c in xdf.columns]
        if existing:
            table = xdf[existing].copy()
            table.rename(
                columns={
                    "name": "Variable",
                    "value": "Level",
                    "capacity": "Capacity",
                    "utilization_percent": "Utilization %",
                },
                inplace=True,
            )
            with st.expander("📋 View Full Decision Variable Table"):
                st.dataframe(table, use_container_width=True, hide_index=True)

    # ===========================================================
    # 4.  BOTTLENECK PENALTIES  (Dual Variables y)
    # ===========================================================

    dual = result.get("dual_variables", [])

    if dual:
        st.markdown(
            '<div class="section-header">🚧 REFINERY BOTTLENECK ANALYSIS</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-sub">'
            "Dual variables (y) represent shadow prices — the profit lost per unit "
            "of violated constraint. Higher penalties indicate critical bottlenecks."
            "</div>",
            unsafe_allow_html=True,
        )

        ydf = pd.DataFrame(dual)

        if "name" in ydf.columns and "penalty" in ydf.columns:
            ydf["penalty"] = pd.to_numeric(ydf["penalty"], errors="coerce")

            chart_df = (
                ydf.dropna(subset=["penalty"])
                .sort_values("penalty", ascending=False)
                .head(25)
            )

            if not chart_df.empty:
                fig = go.Figure()

                # Gradient from amber (high penalty) to green (low)
                max_penalty = chart_df["penalty"].max()
                min_penalty = chart_df["penalty"].min()
                rng = max(max_penalty - min_penalty, 0.001)

                colors = []
                for p in chart_df["penalty"]:
                    ratio = (p - min_penalty) / rng
                    r = int(200 * ratio + 50)
                    g = int(185 * (1 - ratio * 0.5))
                    colors.append(f"rgba({r},{g},0,0.85)")

                fig.add_trace(
                    go.Bar(
                        x=chart_df["name"],
                        y=chart_df["penalty"],
                        text=[f"{v:.4f}" for v in chart_df["penalty"]],
                        textposition="outside",
                        marker=dict(
                            color=colors,
                            line=dict(
                                color="rgba(118,185,0,0.4)", width=1
                            ),
                        ),
                        hovertemplate=(
                            "<b>Constraint:</b> %{x}<br>"
                            "<b>Penalty:</b> %{y:.6f}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    title=dict(
                        text="Constraint Bottleneck Penalties (Shadow Prices)",
                        font=dict(size=16),
                    ),
                )
                fig.update_xaxes(tickangle=-45)
                plotly_dark_layout(fig, height=520)

                st.plotly_chart(fig, use_container_width=True)

    # ===========================================================
    # 5.  CONSTRAINT MATRIX HEATMAP
    # ===========================================================

    heatmap_data = result.get("heatmap", {})
    values = heatmap_data.get("values", [])
    row_labels = heatmap_data.get("row_labels", [])
    col_labels = heatmap_data.get("col_labels", [])

    if values and row_labels and col_labels:

        st.markdown(
            '<div class="section-header">🔥 CONSTRAINT MATRIX HEATMAP</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-sub">'
            "Interactive 2D visualization of the LP constraint matrix density and structure."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            total_rows = min(len(values), len(row_labels))
            valid_rows = [
                row
                for row in values[:total_rows]
                if isinstance(row, list) and len(row) > 0
            ]

            if not valid_rows:
                raise ValueError("Heatmap contains no valid rows.")

            total_cols = min(
                min(len(row) for row in valid_rows),
                len(col_labels),
            )

            if total_cols <= 0:
                raise ValueError("Heatmap contains no columns.")

            # Build clean numeric matrix
            matrix_values = []
            for row in values[:total_rows]:
                clean_row = []
                for val in row[:total_cols]:
                    try:
                        clean_row.append(float(val))
                    except (TypeError, ValueError):
                        clean_row.append(0.0)
                while len(clean_row) < total_cols:
                    clean_row.append(0.0)
                matrix_values.append(clean_row)

            hdf = pd.DataFrame(
                matrix_values,
                index=row_labels[:total_rows],
                columns=col_labels[:total_cols],
            )

            # ── Heatmap Controls ──────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                row_limit = st.number_input(
                    "Rows to display",
                    min_value=1,
                    max_value=total_rows,
                    value=min(80, total_rows),
                    step=10,
                )
            with c2:
                col_limit = st.number_input(
                    "Columns to display",
                    min_value=1,
                    max_value=total_cols,
                    value=min(80, total_cols),
                    step=10,
                )
            with c3:
                heat_mode = st.selectbox(
                    "Visualization mode",
                    [
                        "Coefficient values",
                        "Absolute values",
                        "Non-zero structure (sparsity pattern)",
                    ],
                )

            shown = hdf.iloc[: int(row_limit), : int(col_limit)]
            display_values = shown.to_numpy(dtype=float)

            if "Absolute" in heat_mode:
                display_values = np.abs(display_values)
            elif "Non-zero" in heat_mode:
                display_values = (display_values != 0).astype(int)

            if "Non-zero" in heat_mode:
                colorscale = [
                    [0.0, "#020402"],
                    [0.01, "#020402"],
                    [1.0, "#76b900"],
                ]
                color_title = "Non-zero"
            else:
                colorscale = [
                    [0.00, "#050805"],
                    [0.15, "#102000"],
                    [0.30, "#254200"],
                    [0.50, "#3d6b00"],
                    [0.70, "#5a9500"],
                    [0.85, "#76b900"],
                    [1.00, "#d9ff8a"],
                ]
                color_title = "Coefficient"

            fig = go.Figure(
                data=go.Heatmap(
                    z=display_values,
                    x=list(shown.columns),
                    y=list(shown.index),
                    colorscale=colorscale,
                    colorbar=dict(
                        title=color_title,
                        thickness=15,
                        len=0.75,
                    ),
                    hovertemplate=(
                        "<b>Constraint:</b> %{y}<br>"
                        "<b>Variable:</b> %{x}<br>"
                        "<b>Value:</b> %{z:.6f}"
                        "<extra></extra>"
                    ),
                    xgap=0.8,
                    ygap=0.8,
                )
            )

            fig.update_layout(
                title=dict(
                    text=(
                        f"Constraint Matrix A — "
                        f"{shown.shape[0]} × {shown.shape[1]}"
                    ),
                    x=0.02,
                    font=dict(size=16),
                ),
                height=680,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#020402",
                font=dict(color="white", size=11),
                margin=dict(l=120, r=60, t=80, b=160),
            )
            fig.update_xaxes(
                title="Decision Variables",
                tickangle=-55,
                showgrid=False,
                zeroline=False,
            )
            fig.update_yaxes(
                title="Constraints",
                autorange="reversed",
                showgrid=False,
                zeroline=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "displayModeBar": True,
                },
            )

            # ── Heatmap Statistics ────────────────────────
            displayed = shown.shape[0] * shown.shape[1]
            nonzero = int((shown.to_numpy() != 0).sum())
            display_sparsity = 100 * (1 - nonzero / max(displayed, 1))

            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric("DISPLAYED CELLS", f"{displayed:,}")
            with c2:
                render_metric("DISPLAYED NNZ", f"{nonzero:,}")
            with c3:
                render_metric("DISPLAYED SPARSITY", f"{display_sparsity:.2f}", "%")

            st.caption(
                f"Showing {shown.shape[0]} of {total_rows} rows "
                f"and {shown.shape[1]} of {total_cols} columns."
            )

        except Exception as exc:
            st.error(f"Heatmap rendering error: {exc}")

    # ===========================================================
    # 6.  GPU PERFORMANCE BENCHMARK  (CPU vs GPU bar chart)
    # ===========================================================

    st.markdown(
        '<div class="section-header">📈 GPU PERFORMANCE BENCHMARK</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "Direct comparison: CPU reference execution vs GPU kernel vs end-to-end pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    perf_df = pd.DataFrame(
        {
            "Metric": ["CPU Reference", "GPU Kernel", "End-to-End"],
            "Time (ms)": [cpu_time, gpu_time, total_time],
        }
    )

    fig = go.Figure()
    bar_colors = ["#e05050", "#76b900", "#4a90d9"]

    fig.add_trace(
        go.Bar(
            x=perf_df["Metric"],
            y=perf_df["Time (ms)"],
            text=[f"{t:.2f} ms" for t in perf_df["Time (ms)"]],
            textposition="outside",
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Time: <b>%{y:.2f} ms</b>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"CPU vs GPU Execution Time &nbsp;&nbsp;"
            f'<span style="color:#76b900;font-size:14px;">'
            f"({speedup:.2f}× speedup)</span>",
            font=dict(size=16),
        ),
    )
    plotly_dark_layout(fig, height=470)

    st.plotly_chart(fig, use_container_width=True)

    # ── Speedup highlight ─────────────────────────────────
    if speedup > 1:
        st.markdown(
            f"""
            <div class="info-box" style="text-align:center;">
                ⚡ <b>GPU acceleration achieved
                <span style="color:var(--nv-green-bright);font-size:18px;">
                    {speedup:.2f}×
                </span>
                speedup</b> over CPU reference implementation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ===========================================================
    # 7.  RAW CONSOLE & JSON OUTPUT
    # ===========================================================

    with st.expander("⚙️ VIEW LIVE C++ / CUDA CONSOLE OUTPUT"):
        st.code(st.session_state.raw_output or "(no output)", language="text")

    with st.expander("📦 VIEW COMPLETE SOLVER JSON"):
        st.json(result)


# ==========================================================
# SESSION HISTORY & EXPORT
# ==========================================================

if st.session_state.history:

    st.markdown(
        '<div class="section-header">🗂️ SESSION HISTORY</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "Complete record of every optimization run in this session. "
        "Export to CSV for benchmarking reports."
        "</div>",
        unsafe_allow_html=True,
    )

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    # ── Multi-run speedup trend ───────────────────────────
    if len(st.session_state.history) > 1:
        trend_df = history_df[["Run", "CPU ms", "GPU ms", "Speedup"]].copy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend_df["Run"],
                y=trend_df["CPU ms"],
                mode="lines+markers",
                name="CPU (ms)",
                line=dict(color="#e05050", width=2),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trend_df["Run"],
                y=trend_df["GPU ms"],
                mode="lines+markers",
                name="GPU (ms)",
                line=dict(color="#76b900", width=2),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title=dict(
                text="Performance Trend Across Runs",
                font=dict(size=16),
            ),
            xaxis_title="Run #",
            yaxis_title="Time (ms)",
        )
        plotly_dark_layout(fig, height=400, showlegend=True)

        st.plotly_chart(fig, use_container_width=True)

    # ── CSV Export ────────────────────────────────────────
    csv_data = history_df.to_csv(index=False)
    c1, c2 = st.columns([3, 1])
    with c2:
        st.download_button(
            "📥 EXPORT BENCHMARK CSV",
            data=csv_data,
            file_name="refinery_gpu_benchmark.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c1:
        if st.button("🗑️ Clear History", use_container_width=False):
            st.session_state.history = []
            st.session_state.run_count = 0
            st.rerun()


# ==========================================================
# EMPTY STATE — READY SCREEN
# ==========================================================

if not st.session_state.result:
    st.markdown(
        """
        <div class="info-box">
            <b>⚡ READY FOR GPU OPTIMIZATION</b>
            <br><br>
            Upload an MPS refinery optimization model and launch the
            CUDA acceleration pipeline.
            <br><br>
            <b>Full Dashboard Capabilities:</b>
            <br><br>
            • End-to-end MPS → CUDA → JSON pipeline
            <br>
            • Live C++ console streaming (Popen)
            <br>
            • CPU vs GPU objective accuracy verification
            <br>
            • Sparse matrix CSR compression analysis
            <br>
            • Matrix dimensions, NNZ, sparsity profiling
            <br>
            • Interactive constraint matrix heatmap
            <br>
            • Tank capacity utilization gauges (primal x)
            <br>
            • Refinery bottleneck penalties (dual y / shadow prices)
            <br>
            • GPU performance benchmark bar charts
            <br>
            • Multi-run session history tracking
            <br>
            • CSV benchmark report export
        </div>
        """,
        unsafe_allow_html=True,
    )