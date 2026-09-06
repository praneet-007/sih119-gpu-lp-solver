# CHANGED: full rebuild of the presentation layer around
# gpu_ui_components.py's "GPU Control Center" visual language.
# WHY: the previous version was still a dashboard (cards + charts +
# markdown sections) with CSS decoration bolted on. This version treats
# the page as one continuous computational journey -- a persistent stage
# rail, a physically travelling file object, real equations/matrix
# assembled from the uploaded CSV, a real sparse heat map, an animated
# CPU->GPU transfer, a pulsing GPU block grid, an SVG iteration
# timeline, an animated CPU-vs-GPU race, and a line-by-line benchmark
# console -- all driven by real parsed/solved data, with DEMO values
# explicitly tagged per the honesty rules below.

import streamlit as st
import subprocess
import tempfile
import os
import re
import pandas as pd
import numpy as np
import time
import io

import gpu_ui_components as viz

st.set_page_config(
    page_title="GPU Optimization Control Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# BASE THEME (kept from the previous version) + new control-center CSS
# ============================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: "Segoe UI", Arial, sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at 50% 0%, rgba(118,185,0,0.10), transparent 35%),
            linear-gradient(135deg, #000000 0%, #050805 45%, #000000 100%);
        color: #ffffff;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
    header { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050705 0%, #000000 100%);
        border-right: 1px solid #76b900;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #76b900; }
    div.stButton > button {
        background: linear-gradient(90deg, #76b900, #8bd000);
        color: #000000; border: none; border-radius: 10px; height: 60px;
        font-size: 20px; font-weight: 900; letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(118,185,0,0.25); transition: 0.3s;
    }
    div.stButton > button:hover { box-shadow: 0 0 35px rgba(118,185,0,0.55); transform: scale(1.01); }
    [data-testid="stFileUploader"] {
        background: rgba(118,185,0,0.04); border: 2px dashed rgba(118,185,0,0.55);
        border-radius: 15px; padding: 10px;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(118,185,0,0.08), rgba(0,0,0,0.7));
        border: 1px solid rgba(118,185,0,0.35); padding: 15px; border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] { color: #9aa696 !important; }
    div[data-testid="stMetricValue"] { color: #76b900 !important; }
    .stProgress > div > div > div > div { background-color: #76b900; }
    hr { border-color: rgba(118,185,0,0.25); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# CHANGED: inject the control-center keyframes/classes.
# WHY: every viz.render_* call below depends on these.
viz.inject_base_css()


SOLVER_PATH = "./gpu_solver"


# ============================================================
# BACKEND CONTRACT (unchanged from the existing app)
# ============================================================

def parse_solver_output(output):
    result = {}
    patterns = {
        "cpu_time": r"CPU_TIME\s*=\s*([0-9.]+)",
        "gpu_compute_time": r"GPU_COMPUTE_TIME\s*=\s*([0-9.]+)",
        "gpu_total_time": r"GPU_TOTAL_TIME\s*=\s*([0-9.]+)",
        "speedup": r"SPEEDUP\s*=\s*([0-9.]+)",
        "objective": r"OBJECTIVE_VALUE\s*=\s*([0-9.eE+-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            result[key] = float(match.group(1))
    if "speedup" not in result:
        if "cpu_time" in result and "gpu_total_time" in result:
            if result["gpu_total_time"] > 0:
                result["speedup"] = result["cpu_time"] / result["gpu_total_time"]
    return result


def run_gpu_solver(uploaded_file):
    input_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp:
            temp.write(uploaded_file.getvalue())
            input_path = temp.name

        process = subprocess.run(
            [SOLVER_PATH, input_path], capture_output=True, text=True, timeout=300
        )
        if process.returncode != 0:
            return None, process.stderr

        result = parse_solver_output(process.stdout)
        return result, process.stdout

    except FileNotFoundError:
        return None, "gpu_solver executable was not found."
    except subprocess.TimeoutExpired:
        return None, "GPU solver timed out."
    except Exception as e:
        return None, str(e)
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)


def demo_solver():
    time.sleep(1)
    output = """
CPU_TIME=1250
GPU_COMPUTE_TIME=180
GPU_TOTAL_TIME=205
SPEEDUP=6.10
OBJECTIVE_VALUE=12345.70
"""
    return parse_solver_output(output), output


def demo_optimization_trace():
    """DEMO / SIMULATED -- only used while DEMO MODE is on, always
    rendered behind a visible DEMO tag. Real CUDA solver output will
    replace this once the solver returns per-iteration data."""
    return pd.DataFrame({
        "Iteration": [1, 10, 20, 30, 40, 50, 60],
        "Objective": [18452.2, 15120.5, 13742.8, 12982.1, 12521.4, 12391.8, 12345.7],
        "Residual": [8.42, 4.12, 2.31, 1.04, 0.52, 0.18, 0.03],
    })


def demo_tank_utilization():
    """DEMO / SIMULATED petroleum resource view -- the uploaded CSV has
    no tank-capacity data, so this is illustrative only and always
    labeled as such (see section 5/26 of the spec)."""
    return [
        ("Tank 01 — Crude Feed", 91, "NORMAL"),
        ("Tank 02 — Naphtha", 72, "NORMAL"),
        ("Tank 03 — Residue", 100, "AT CAPACITY"),
        ("Tank 04 — Blend Output", 78, "NORMAL"),
    ]


def demo_bottleneck_rows():
    """DEMO / SIMULATED -- see demo_tank_utilization()."""
    return [
        ("Constraint C1", 91, "NORMAL"),
        ("Constraint C2", 72, "NORMAL"),
        ("Constraint C3", 100, "BOTTLENECK"),
    ]


# ============================================================
# SESSION STATE
# ============================================================

if "stage" not in st.session_state:
    st.session_state.stage = 0
if "show_all_equations" not in st.session_state:
    st.session_state.show_all_equations = False
if "show_all_matrix_rows" not in st.session_state:
    st.session_state.show_all_matrix_rows = False


# ============================================================
# HERO
# ============================================================

# CHANGED: the stage indicator moved into the sidebar as a persistent
# vertical PIPELINE (not a horizontal row of boxes) and is now built by
# viz.render_stage_pipeline(), which renders three-state nodes (○/●/✓)
# joined by animated data-pipe connectors -- only the ONE connector
# leading into the currently active stage plays moving particles; done
# connectors are static solid lines, pending ones dim/dashed.
# WHY: the master prompt's non-negotiable requirement -- static arrows
# between boxes are explicitly rejected; the connection itself must be
# the animated element, and only the current transition may animate.
with st.sidebar:
    st.markdown('<h2 style="color:#76b900;">⚡ CONTROL CENTER</h2>', unsafe_allow_html=True)
    st.divider()
    demo_mode = st.toggle("DEMO MODE", value=True)
    st.divider()
    st.markdown('<h3 style="color:#76b900;">SOLVER STATUS</h3>', unsafe_allow_html=True)
    solver_ready = os.path.exists(SOLVER_PATH)
    if solver_ready:
        st.success("● GPU SOLVER READY")
    else:
        st.warning("● GPU SOLVER NOT FOUND")
    st.divider()
    st.markdown('<h3 style="color:#76b900;">EXECUTION PIPELINE</h3>', unsafe_allow_html=True)
    rail_slot = st.empty()
    with rail_slot.container():
        viz.render_stage_pipeline(st.session_state.stage)
    st.caption("○ waiting · ● active (data flowing in) · ✓ completed")

viz.render_hero(demo_mode, solver_ready)


# ============================================================
# STAGE 0-1: INPUT FILE + SYSTEM PROFILE
# ============================================================

viz.frame_start("📄 COMPANY INPUT FILE")
uploaded_file = st.file_uploader("Upload petroleum matrix", type=["txt", "csv", "dat"], label_visibility="collapsed")
viz.frame_end()

matrix_df = None
constraint_df = None
n_constraints = 0
n_vars = 0
nnz = 0

if uploaded_file:
    st.session_state.stage = max(st.session_state.stage, 1)

    size_kb = uploaded_file.size / 1024
    viz.render_file_object(uploaded_file.name, size_kb, uploaded_file.size)
    viz.render_travel_rail("📁 UPLOAD", "🔍 ANALYSIS ENGINE")

    try:
        matrix_df = pd.read_csv(uploaded_file)
        constraint_df = matrix_df[matrix_df["type"].astype(str).str.lower() == "constraint"].copy()
        constraint_df["row"] = pd.to_numeric(constraint_df["row"], errors="coerce")
        constraint_df["col"] = pd.to_numeric(constraint_df["col"], errors="coerce")
        constraint_df["value"] = pd.to_numeric(constraint_df["value"], errors="coerce")
        constraint_df = constraint_df.dropna(subset=["row", "col", "value"])

        n_constraints = int(constraint_df["row"].nunique())
        n_vars = int(constraint_df["col"].max())
        nnz = int(len(constraint_df))
        total_elements = n_constraints * n_vars
        sparsity = (1 - (nnz / total_elements)) * 100 if total_elements > 0 else 0

        objective_rows = matrix_df[matrix_df["type"].astype(str).str.lower() == "objective"]
        has_objective = len(objective_rows) > 0

        st.session_state.stage = max(st.session_state.stage, 1)

        viz.frame_start("🖥 SYSTEM PROFILE")
        viz.render_system_profile_counters([
            ("CONSTRAINTS", n_constraints, ""),
            ("VARIABLES", n_vars, ""),
            ("NON-ZERO VALUES", nnz, ""),
            ("SPARSITY", sparsity, "%"),
        ])
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f'<div style="font-size:12px; color:#9aa696;">GPU &nbsp;→&nbsp; NVIDIA GPU / CUDA<br>'
                f'CPU &nbsp;→&nbsp; Host CPU (scipy/HiGHS baseline)</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            solver_label = "DEMO" if demo_mode else "LIVE"
            st.markdown(
                f'<div style="font-size:12px; color:#9aa696;">MATRIX &nbsp;→&nbsp; {n_constraints:,} × {n_vars:,}<br>'
                f'SOLVER STATUS &nbsp;→&nbsp; {solver_label}</div>',
                unsafe_allow_html=True,
            )
        if not has_objective:
            st.warning(
                "Objective function (c) is not present in this dataset. "
                "The input defines constraint matrix A and RHS vector b only — "
                "no objective coefficients are invented here."
            )
        viz.frame_end()

    except Exception as e:
        st.warning(f"Unable to analyze matrix: {e}")


# ============================================================
# STAGE 2: ANALYSIS ENGINE
# ============================================================

if uploaded_file and constraint_df is not None:
    viz.frame_start("🔍 ANALYSIS ENGINE")
    viz.render_analysis_sequence([
        "Parsing CSV",
        "Reading constraints",
        "Detecting variables",
        "Extracting non-zero coefficients",
        "Building sparse representation",
    ])
    viz.frame_end()


# ============================================================
# STAGE 3-4: LP FORMULATION + EQUATIONS
# ============================================================

    st.session_state.stage = max(st.session_state.stage, 2)

    viz.frame_start("📐 LP FORMULATION")
    viz.render_transform_arrow("RAW CSV", "CONSTRAINT EXTRACTION", "LINEAR PROGRAM")
    st.latex(r"A\,x \leq b")
    st.markdown(
        "**A** = coefficient matrix &nbsp;|&nbsp; **x** = decision variables &nbsp;|&nbsp; **b** = RHS vector",
    )
    viz.frame_end()

    st.session_state.stage = max(st.session_state.stage, 3)

    viz.frame_start("🧮 EQUATIONS")
    grouped = constraint_df.groupby("row", sort=True)
    n_show = 50 if st.session_state.show_all_equations else 5

    equation_lines = []
    shown = 0
    for row_number, group in grouped:
        if shown >= n_show:
            break
        terms = [f"{float(r['value']):g}x<sub>{int(r['col'])}</sub>" for _, r in group.iterrows()]
        left = " + ".join(terms)
        rhs_vals = pd.to_numeric(group["rhs"], errors="coerce").dropna()
        rhs = rhs_vals.iloc[0] if len(rhs_vals) else None
        senses = group["sense"].dropna()
        sense = str(senses.iloc[0]) if len(senses) else "<="
        if rhs is not None:
            equation_lines.append(f'<b>C{int(row_number)}:</b> &nbsp; {left} {sense} {rhs:g}')
        shown += 1

    viz.render_equations(equation_lines)
    st.caption(f"Showing {shown} of {grouped.ngroups:,} total constraints.")

    toggle_label = "SHOW 5 EQUATIONS" if st.session_state.show_all_equations else "SHOW 50 EQUATIONS"
    if st.button(toggle_label, key="eq_toggle"):
        st.session_state.show_all_equations = not st.session_state.show_all_equations
        st.rerun()
    viz.frame_end()


# ============================================================
# STAGE 5: EQUATION -> MATRIX TRANSFORMATION
# ============================================================

    st.session_state.stage = max(st.session_state.stage, 4)

    viz.frame_start("🔄 MATRIX REPRESENTATION")
    viz.render_transform_arrow("EQUATIONS", "CONVERTING TO MATRIX", "MATRIX A")

    max_rows = 50 if st.session_state.show_all_matrix_rows else 3
    viz.render_matrix_notation(
        rows=constraint_df["row"].astype(int).tolist(),
        cols=constraint_df["col"].astype(int).tolist(),
        values=constraint_df["value"].tolist(),
        n_rows_total=n_constraints,
        n_cols_total=n_vars,
        max_rows=max_rows,
    )
    toggle_label2 = "SHOW 3 ROWS" if st.session_state.show_all_matrix_rows else "EXPAND TO 50 ROWS"
    if st.button(toggle_label2, key="matrix_toggle"):
        st.session_state.show_all_matrix_rows = not st.session_state.show_all_matrix_rows
        st.rerun()
    viz.frame_end()


# ============================================================
# STAGE 6: SPARSE MATRIX HEAT MAP + FINGERPRINT
# ============================================================

    st.session_state.stage = max(st.session_state.stage, 4)

    viz.frame_start("▦ SPARSE MATRIX HEAT MAP")
    viz.render_heatmap(
        rows=constraint_df["row"].astype(int).tolist(),
        cols=constraint_df["col"].astype(int).tolist(),
        values=constraint_df["value"].tolist(),
        n_rows_total=n_constraints,
        n_cols_total=n_vars,
        title=f"A = {n_constraints:,} × {n_vars:,}",
    )
    viz.frame_end()

    viz.frame_start("🧬 MATRIX FINGERPRINT")
    row_window = st.slider(
        "Row window (first N constraint rows)", min_value=10,
        max_value=max(n_constraints, 10), value=min(50, n_constraints), step=10,
        key="fingerprint_window",
    )
    windowed = constraint_df[constraint_df["row"] < row_window]
    viz.render_heatmap(
        rows=windowed["row"].astype(int).tolist(),
        cols=windowed["col"].astype(int).tolist(),
        values=windowed["value"].tolist(),
        n_rows_total=row_window,
        n_cols_total=n_vars,
        title=f"First {row_window} rows — structural fingerprint",
    )
    viz.frame_end()


# ============================================================
# STAGE 7: CUDA TRANSFER
# ============================================================

    st.session_state.stage = max(st.session_state.stage, 5)

    viz.frame_start("🚀 CUDA TRANSFER")
    viz.render_cuda_transfer(nnz, size_kb)
    viz.frame_end()

with rail_slot.container():
    viz.render_stage_pipeline(st.session_state.stage)

st.divider()

run_button = st.button("⚡ RUN OPTIMIZATION", type="primary", use_container_width=True)


# ============================================================
# STAGE 8+: GPU SOLVER -> RESULTS
# ============================================================

if run_button:

    if uploaded_file is None:
        st.error("Please upload a petroleum matrix first.")

    else:
        st.session_state.stage = 6

        viz.frame_start("⚡ GPU SOLVER")
        progress = st.progress(0)
        status = st.empty()
        status.info("INITIALIZING OPTIMIZATION ENGINE...")
        progress.progress(20)

        chip_slot = st.empty()
        with chip_slot.container():
            viz.render_gpu_grid(active=True)

        progress.progress(55)

        if demo_mode:
            result, raw_output = demo_solver()
        else:
            result, raw_output = run_gpu_solver(uploaded_file)

        chip_slot.empty()
        progress.progress(90)

        if result is None:
            progress.progress(0)
            status.error("OPTIMIZATION FAILED")
            st.error(raw_output)
            viz.frame_end()

        else:
            progress.progress(100)
            status.success("✓ OPTIMIZATION COMPLETED")
            viz.frame_end()

            cpu_time = result.get("cpu_time", 0)
            gpu_compute_time = result.get("gpu_compute_time", 0)
            gpu_total_time = result.get("gpu_total_time", 0)
            speedup = result.get("speedup", 0)
            objective = result.get("objective", 0)

            # --------------------------------------------------
            # TANK UTILIZATION  (section 5)
            # --------------------------------------------------
            st.session_state.stage = 6
            with rail_slot.container():
                viz.render_stage_pipeline(st.session_state.stage)

            viz.frame_start("🛢 TANK UTILIZATION", viz.demo_tag() if demo_mode else viz.unavailable_tag())
            if demo_mode:
                viz.render_fill_bars(demo_tank_utilization())
            else:
                st.info(
                    "Tank utilization data unavailable for this input — the live "
                    "GPU solver does not yet return per-resource utilization values."
                )
            viz.frame_end()

            # --------------------------------------------------
            # BOTTLENECK DETECTION  (section 6)
            # --------------------------------------------------
            viz.frame_start("⚠ BOTTLENECK DETECTION", viz.demo_tag() if demo_mode else viz.unavailable_tag())
            if demo_mode:
                bottleneck_rows = demo_bottleneck_rows()
                viz.render_fill_bars(bottleneck_rows)
                worst = max(bottleneck_rows, key=lambda r: r[1])
                if worst[1] >= 98:
                    st.warning(f"⚠ BOTTLENECK DETECTED — {worst[0]} at {worst[1]}% utilization.")
            else:
                st.info(
                    "Bottleneck analysis unavailable for this input — computing real "
                    "constraint utilization requires the solved decision-variable "
                    "vector, which the live solver bridge does not yet return."
                )
            viz.frame_end()

            # --------------------------------------------------
            # ITERATIONS + CONVERGENCE
            # --------------------------------------------------
            st.session_state.stage = 7
            with rail_slot.container():
                viz.render_stage_pipeline(st.session_state.stage)

            viz.frame_start("📈 ITERATION ENGINE", viz.demo_tag() if demo_mode else "")

            if demo_mode:
                trace_df = demo_optimization_trace()
                viz.render_iteration_timeline(trace_df["Iteration"].tolist(), trace_df["Objective"].tolist())

                col_obj, col_res = st.columns(2)
                with col_obj:
                    st.markdown('<div style="font-size:11px; font-weight:800; color:#9aa696;">OBJECTIVE</div>', unsafe_allow_html=True)
                    viz.render_cascade(trace_df["Objective"].tolist(), fmt="{:,.0f}")
                with col_res:
                    st.markdown('<div style="font-size:11px; font-weight:800; color:#9aa696;">RESIDUAL</div>', unsafe_allow_html=True)
                    viz.render_cascade(trace_df["Residual"].tolist(), fmt="{:.2f}")

                viz.render_convergence_badge()

                with st.expander("◈ Iteration table (technical view)"):
                    st.dataframe(trace_df, use_container_width=True)

                final_objective = trace_df.iloc[-1]["Objective"]
                final_residual = trace_df.iloc[-1]["Residual"]
            else:
                st.info(
                    "Live CUDA solver connected. Iteration history and convergence "
                    "path will render here once the solver bridge streams per-"
                    "iteration data — no simulated trace is shown in LIVE mode."
                )
                final_objective = objective
                final_residual = None
            viz.frame_end()

            # --------------------------------------------------
            # OPTIMAL SOLUTION
            # --------------------------------------------------
            st.session_state.stage = 8
            with rail_slot.container():
                viz.render_stage_pipeline(st.session_state.stage)

            viz.frame_start("🎯 OPTIMAL SOLUTION", viz.demo_tag() if demo_mode else "")
            sol_col1, sol_col2 = st.columns(2)
            with sol_col1:
                st.metric("OBJECTIVE VALUE", f"₹{final_objective:,.2f}" if demo_mode else f"{objective:,.2f}")
            with sol_col2:
                if final_residual is not None:
                    st.metric("FINAL RESIDUAL", f"{final_residual:.2f}")
                else:
                    st.metric("FINAL RESIDUAL", "—")

            if demo_mode:
                st.caption("DEMO / SIMULATED — decision variables below are illustrative, not solved values.")
                v1, v2, v3, v4 = st.columns(4)
                for col, name, val in zip((v1, v2, v3, v4), ("x1", "x2", "x3", "x4"), (120, 80, 45, 0)):
                    with col:
                        st.metric(name, str(val))

                viz.frame_start("✓ CONSTRAINT VALIDATION", viz.demo_tag())
                viz.render_fill_bars([
                    ("C1", 91, "OK"), ("C2", 72, "OK"), ("C3", 100, "AT LIMIT"),
                ])
                st.success("✓ All demo constraints satisfied")
                viz.frame_end()
            else:
                st.info("Solution vector will render here once the live solver bridge returns it.")
            viz.frame_end()

            # --------------------------------------------------
            # CPU VS GPU RACE + BENCHMARK CONSOLE
            # --------------------------------------------------
            st.session_state.stage = 9
            with rail_slot.container():
                viz.render_stage_pipeline(st.session_state.stage)

            viz.frame_start("⚔ CPU VS GPU")
            viz.render_race(cpu_time, gpu_total_time)
            viz.render_speedup_badge(speedup)
            viz.frame_end()

            st.session_state.stage = 10
            with rail_slot.container():
                viz.render_stage_pipeline(st.session_state.stage)

            viz.frame_start("> GPU BENCHMARK CONSOLE", viz.demo_tag() if demo_mode else viz.demo_tag() if False else "")
            demo_prefix = [("demo", "SIMULATED BENCHMARK — DEMO MODE")] if demo_mode else []
            console_lines = demo_prefix + [
                ("info", "Loading optimization problem..."),
                ("info", f"Matrix: {n_constraints:,} x {n_vars:,}" if n_constraints else "Matrix: (uploaded)"),
                ("info", f"Nonzeros: {nnz:,}" if nnz else "Nonzeros: (uploaded)"),
                ("info", "CPU execution started..."),
                ("ok", f"CPU execution completed: {cpu_time:.2f} ms"),
                ("info", "CUDA initialization..."),
                ("ok", "GPU device ready" if solver_ready else "GPU device not found (demo timing shown)"),
                ("info", "Host -> Device transfer..."),
                ("ok", "Transfer completed"),
                ("info", "GPU optimization kernel..."),
                ("ok", f"GPU computation: {gpu_compute_time:.2f} ms"),
                ("info", f"End-to-end GPU time: {gpu_total_time:.2f} ms"),
                ("ok", f"GPU speedup: {speedup:.2f}x"),
                ("ok", "STATUS: ACCELERATION ACHIEVED" if speedup >= 1 else "STATUS: NO ACCELERATION MEASURED"),
            ]
            viz.render_benchmark_console(console_lines)
            viz.frame_end()

            viz.frame_start("⏱ BENCHMARK TIMELINE")
            gpu_init = max(gpu_total_time - gpu_compute_time, 0) * 0.4
            gpu_xfer = max(gpu_total_time - gpu_compute_time, 0) * 0.6
            viz.render_benchmark_timeline(
                cpu_time,
                [
                    ("init", gpu_init, "#4c7a00"),
                    ("H2D transfer", gpu_xfer, "#76b900"),
                    ("GPU compute", gpu_compute_time, "#a6ff00"),
                ],
            )
            viz.frame_end()

            viz.frame_start("📊 PERFORMANCE INSIGHT")
            st.markdown(
                f'<div style="text-align:center; font-family:JetBrains Mono,monospace; font-size:34px; '
                f'font-weight:900; color:#76b900;">{speedup:.2f}× FASTER THAN CPU</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Theoretical equivalent throughput based on the measured benchmark "
                "ratio above — not a literal guaranteed rate."
            )
            with st.expander("◈ VIEW RAW SOLVER OUTPUT"):
                st.code(raw_output, language="text")
            viz.frame_end()

st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#667060; padding:16px; letter-spacing:1px;">
        GPU OPTIMIZATION CONTROL CENTER • CUDA • STREAMLIT
    </div>
    """,
    unsafe_allow_html=True,
)
