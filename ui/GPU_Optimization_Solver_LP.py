import streamlit as st
import subprocess
import tempfile
import os
import re
import pandas as pd
import time

st.set_page_config(
    page_title="GPU Optimization Solver",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(118, 185, 0, 0.12),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #000000 0%,
            #050805 45%,
            #000000 100%
        );
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

header {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #050705 0%,
            #000000 100%
        );
    border-right: 1px solid #76b900;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #76b900;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    letter-spacing: 2px;
    color: #76b900;
    text-shadow:
        0 0 10px rgba(118, 185, 0, 0.5),
        0 0 25px rgba(118, 185, 0, 0.25);
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #b8c7ad;
    font-size: 18px;
    letter-spacing: 1px;
    margin-bottom: 35px;
}

.section-title {
    color: #76b900;
    font-size: 25px;
    font-weight: 800;
    border-left: 4px solid #76b900;
    padding-left: 12px;
    margin-top: 20px;
}

.result-box {
    background:
        linear-gradient(
            145deg,
            rgba(118, 185, 0, 0.10),
            rgba(0, 0, 0, 0.75)
        );
    border: 1px solid rgba(118, 185, 0, 0.45);
    border-radius: 14px;
    padding: 22px 10px;
    text-align: center;
    box-shadow:
        0 0 15px rgba(118, 185, 0, 0.08);
    transition: 0.3s;
}

.result-box:hover {
    border-color: #76b900;
    box-shadow:
        0 0 25px rgba(118, 185, 0, 0.22);
    transform: translateY(-2px);
}

.result-value {
    color: #76b900;
    font-size: 30px;
    font-weight: 900;
    margin-top: 5px;
}

.result-label {
    color: #9aa696;
    font-size: 13px;
    letter-spacing: 1px;
    font-weight: 700;
}

div.stButton > button {
    background:
        linear-gradient(
            90deg,
            #76b900,
            #8bd000
        );
    color: #000000;
    border: none;
    border-radius: 10px;
    height: 65px;
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 2px;
    box-shadow:
        0 0 20px rgba(118, 185, 0, 0.25);
    transition: 0.3s;
}

div.stButton > button:hover {
    background:
        linear-gradient(
            90deg,
            #8bd000,
            #76b900
        );
    box-shadow:
        0 0 35px rgba(118, 185, 0, 0.55);
    color: #000000;
    transform: scale(1.01);
}

[data-testid="stFileUploader"] {
    background: rgba(118, 185, 0, 0.04);
    border: 2px dashed rgba(118, 185, 0, 0.55);
    border-radius: 15px;
    padding: 10px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #76b900;
    background: rgba(118, 185, 0, 0.08);
}

.stTextInput input,
.stTextArea textarea {
    background-color: #050705 !important;
    color: white !important;
    border: 1px solid #76b900 !important;
}

.stSelectbox div,
.stMultiSelect div {
    background-color: #050705 !important;
}

div[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(118, 185, 0, 0.08),
            rgba(0, 0, 0, 0.7)
        );
    border: 1px solid rgba(118, 185, 0, 0.35);
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #9aa696 !important;
}

div[data-testid="stMetricValue"] {
    color: #76b900 !important;
}

.stProgress > div > div > div > div {
    background-color: #76b900;
}

.stProgress > div > div {
    background-color: #17200d;
}

hr {
    border-color: rgba(118, 185, 0, 0.25);
}

.stExpander {
    border: 1px solid rgba(118, 185, 0, 0.3);
    border-radius: 10px;
    background: rgba(118, 185, 0, 0.03);
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

footer {
    visibility: hidden;
}

.green-text {
    color: #76b900;
    font-weight: 800;
}

.status-online {
    color: #76b900;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


SOLVER_PATH = "./gpu_solver"


def parse_solver_output(output):
    result = {}

    patterns = {
        "cpu_time": r"CPU_TIME\s*=\s*([0-9.]+)",
        "gpu_compute_time": r"GPU_COMPUTE_TIME\s*=\s*([0-9.]+)",
        "gpu_total_time": r"GPU_TOTAL_TIME\s*=\s*([0-9.]+)",
        "speedup": r"SPEEDUP\s*=\s*([0-9.]+)",
        "objective": r"OBJECTIVE_VALUE\s*=\s*([0-9.eE+-]+)"
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


def demo_solver():
    time.sleep(1.5)
    output = """
CPU_TIME=1250
GPU_COMPUTE_TIME=180
GPU_TOTAL_TIME=205
SPEEDUP=6.10
OBJECTIVE_VALUE=84523.45
"""
    return parse_solver_output(output), output


def build_lp_model(components, total_production, min_octane, max_sulfur):
    """
    Convert user-friendly refinery inputs into the LP:

        minimize     c^T x

        subject to   A x <= b
                     x >= 0

    Constraint rows:
      1. Total production equality
      2. Minimum octane
      3. Maximum sulfur
      4..n+3. Component capacity limits
    """
    names = [row["Component"] for row in components]
    costs = [float(row["Cost"]) for row in components]
    octanes = [float(row["Octane"]) for row in components]
    sulfurs = [float(row["Sulfur"]) for row in components]
    capacities = [float(row["Max Flow"]) for row in components]

    c = costs

    # Equality is kept separately so the UI can clearly show the original LP.
    A_eq = [[1.0 for _ in names]]
    b_eq = [float(total_production)]

    # Convert >= octane to <= form for the matrix representation.
    A_ub = [
        [-x for x in octanes],
        sulfurs,
    ]
    b_ub = [
        -float(min_octane) * float(total_production),
        float(max_sulfur) * float(total_production),
    ]

    # x_i <= capacity_i
    for i in range(len(names)):
        row = [0.0] * len(names)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(capacities[i])

    return {
        "names": names,
        "c": c,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_ub": A_ub,
        "b_ub": b_ub,
    }


def lp_equations_html(model, total_production, min_octane, max_sulfur):
    names = model["names"]
    costs = model["c"]
    terms = " + ".join(
        f"{cost:g}x<sub>{i+1}</sub>" for i, cost in enumerate(costs)
    )

    production = " + ".join(
        f"x<sub>{i+1}</sub>" for i in range(len(names))
    )

    octane = " + ".join(
        f"{float(next(r['Octane'] for r in st.session_state.components if r['Component'] == name)):g}x<sub>{i+1}</sub>"
        for i, name in enumerate(names)
    )

    sulfur = " + ".join(
        f"{float(next(r['Sulfur'] for r in st.session_state.components if r['Component'] == name)):g}x<sub>{i+1}</sub>"
        for i, name in enumerate(names)
    )

    return f"""
    <div class="result-box" style="text-align:left;">
        <div class="result-label">OBJECTIVE</div>
        <div style="font-size:18px; margin-top:8px;">
            Minimize&nbsp;&nbsp; <b>{terms}</b>
        </div>
        <hr>
        <div class="result-label">CONSTRAINTS</div>
        <div style="font-size:16px; line-height:2;">
            {production} = {total_production:g}<br>
            {octane} ≥ {min_octane:g} × {total_production:g}<br>
            {sulfur} ≤ {max_sulfur:g} × {total_production:g}<br>
            x<sub>i</sub> ≥ 0
        </div>
    </div>
    """


def matrix_text(model):
    """
    Human-readable internal LP representation.

    NOTE:
    The exact serialization expected by ./gpu_solver may differ.
    Keep this function as the single adapter point if the CUDA solver
    uses a different input format.
    """
    names = model["names"]
    c = model["c"]
    A_eq = model["A_eq"]
    b_eq = model["b_eq"]
    A_ub = model["A_ub"]
    b_ub = model["b_ub"]

    lines = [
        "# LP generated by GPU Optimization Solver",
        f"# VARIABLES {len(names)}",
        f"# EQUALITY_CONSTRAINTS {len(A_eq)}",
        f"# INEQUALITY_CONSTRAINTS {len(A_ub)}",
        "",
        "VARIABLES",
        " ".join(names),
        "",
        "OBJECTIVE",
        " ".join(f"{x:.10g}" for x in c),
        "",
        "A_EQ",
    ]
    lines += [" ".join(f"{x:.10g}" for x in row) for row in A_eq]
    lines += [
        "",
        "B_EQ",
        " ".join(f"{x:.10g}" for x in b_eq),
        "",
        "A_UB",
    ]
    lines += [" ".join(f"{x:.10g}" for x in row) for row in A_ub]
    lines += [
        "",
        "B_UB",
        " ".join(f"{x:.10g}" for x in b_ub),
    ]

    return "\n".join(lines)


def run_gpu_solver_from_lp(model):
    input_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".txt",
            mode="w",
            encoding="utf-8"
        ) as temp:
            temp.write(matrix_text(model))
            input_path = temp.name

        process = subprocess.run(
            [SOLVER_PATH, input_path],
            capture_output=True,
            text=True,
            timeout=300
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


# ---------------------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------------------

st.markdown(
    '<div class="main-title">⚡ GPU OPTIMIZATION SOLVER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'INDIGENOUS GPU-ACCELERATED LINEAR PROGRAMMING ENGINE'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<h2 style="color:#76b900;">⚡ GPU CONTROL CENTER</h2>',
        unsafe_allow_html=True
    )

    st.divider()

    demo_mode = st.toggle(
        "DEMO MODE",
        value=True
    )

    if demo_mode:
        st.info("Demo mode uses simulated solver results.")
    else:
        st.success("Live GPU solver mode enabled.")

    st.divider()

    st.markdown(
        '<h3 style="color:#76b900;">SOLVER SCOPE</h3>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="line-height:1.8;">
        <b>Problem:</b> Linear Programming (LP)<br>
        <b>Model:</b> User input → LP → Matrix<br>
        <b>Method:</b> Interior Point Method<br>
        <b>Linear Algebra:</b> Newton + Factorization<br>
        <b>Acceleration:</b> NVIDIA CUDA
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<h3 style="color:#76b900;">PIPELINE</h3>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            color:white;
            font-size:15px;
            font-weight:700;
            line-height:1.2;
        ">
            <div style="margin:8px 0;">🛢️ INDUSTRIAL INPUT</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">📐 LP EQUATION</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">🔢 MATRIX FORM</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">⚙️ INTERIOR POINT</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">🚀 NVIDIA GPU</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">📊 OPTIMAL RESULT</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    status_text = (
        "● GPU SOLVER READY"
        if os.path.exists(SOLVER_PATH)
        else "● GPU SOLVER NOT FOUND"
    )

    if os.path.exists(SOLVER_PATH):
        st.success(status_text)
    else:
        st.warning(status_text)


# ---------------------------------------------------------------------
# LP INPUT
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-title">LINEAR PROGRAMMING INPUT</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter the refinery problem in business terms. The application "
    "automatically builds the LP equation and matrix."
)

left, right = st.columns(2)

with left:
    total_production = st.number_input(
        "Required Production (barrels/hr)",
        min_value=1.0,
        value=10000.0,
        step=100.0
    )

with right:
    min_octane = st.number_input(
        "Minimum Octane",
        min_value=0.0,
        value=92.0,
        step=0.5
    )

max_sulfur = st.number_input(
    "Maximum Sulfur (ppm)",
    min_value=0.0,
    value=10.0,
    step=0.5
)

st.markdown("### Blending Components")

default_components = pd.DataFrame([
    {"Component": "Alkylate", "Cost": 80.0, "Octane": 98.0, "Sulfur": 2.0, "Max Flow": 3000.0},
    {"Component": "Reformate", "Cost": 60.0, "Octane": 95.0, "Sulfur": 8.0, "Max Flow": 4000.0},
    {"Component": "FCC Gasoline", "Cost": 45.0, "Octane": 90.0, "Sulfur": 20.0, "Max Flow": 5000.0},
    {"Component": "Naphtha", "Cost": 35.0, "Octane": 75.0, "Sulfur": 30.0, "Max Flow": 3000.0},
])

edited_components = st.data_editor(
    default_components,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Component": st.column_config.TextColumn("Component"),
        "Cost": st.column_config.NumberColumn("Cost / unit", min_value=0.0),
        "Octane": st.column_config.NumberColumn("Octane", min_value=0.0),
        "Sulfur": st.column_config.NumberColumn("Sulfur (ppm)", min_value=0.0),
        "Max Flow": st.column_config.NumberColumn("Maximum Flow", min_value=0.0),
    }
)

components = edited_components.to_dict("records")
st.session_state.components = components

valid_components = (
    len(components) > 0
    and all(str(row["Component"]).strip() for row in components)
    and all(
        float(row["Cost"]) >= 0
        and float(row["Octane"]) >= 0
        and float(row["Sulfur"]) >= 0
        and float(row["Max Flow"]) >= 0
        for row in components
    )
)

if not valid_components:
    st.error("Please provide valid component names and non-negative numeric values.")
    st.stop()

model = build_lp_model(
    components,
    total_production,
    min_octane,
    max_sulfur
)

st.divider()

# ---------------------------------------------------------------------
# LP EQUATION
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-title">GENERATED LP EQUATION</div>',
    unsafe_allow_html=True
)

st.markdown(
    lp_equations_html(
        model,
        total_production,
        min_octane,
        max_sulfur
    ),
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# INTERNAL MATRIX
# ---------------------------------------------------------------------

st.markdown(
    '<div class="section-title">INTERNAL MATRIX REPRESENTATION</div>',
    unsafe_allow_html=True
)

st.caption(
    "This matrix is generated automatically from the LP inputs. "
    "The user does not need to enter it manually."
)

with st.expander("◈ VIEW GENERATED A, b, c MATRICES"):
    st.code(matrix_text(model), language="text")

st.divider()

run_button = st.button(
    "⚡ SOLVE LP ON GPU",
    type="primary",
    use_container_width=True
)

# ---------------------------------------------------------------------
# SOLVE
# ---------------------------------------------------------------------

if run_button:
    st.markdown(
        '<div class="section-title">OPTIMIZATION ENGINE</div>',
        unsafe_allow_html=True
    )

    progress = st.progress(0)
    status = st.empty()

    status.info("BUILDING LINEAR PROGRAM...")
    progress.progress(20)
    time.sleep(0.4)

    status.info("GENERATING MATRIX FORM...")
    progress.progress(40)
    time.sleep(0.4)

    if demo_mode:
        result, raw_output = demo_solver()
    else:
        status.info("RUNNING GPU LP SOLVER...")
        result, raw_output = run_gpu_solver_from_lp(model)

    progress.progress(80)

    if result is None:
        progress.progress(0)
        status.error("OPTIMIZATION FAILED")
        st.error(raw_output)

    else:
        progress.progress(100)
        status.success("✓ LP OPTIMIZATION COMPLETED")

        cpu_time = result.get("cpu_time", 0)
        gpu_compute_time = result.get("gpu_compute_time", 0)
        gpu_total_time = result.get("gpu_total_time", 0)
        speedup = result.get("speedup", 0)
        objective = result.get("objective", 0)

        st.divider()

        st.markdown(
            '<div class="section-title">OPTIMIZATION RESULT</div>',
            unsafe_allow_html=True
        )

        r1, r2 = st.columns(2)

        with r1:
            st.metric("OBJECTIVE VALUE", f"{objective:,.2f}")

        with r2:
            st.metric("SOLVER STATUS", "OPTIMAL / COMPLETED")

        st.divider()

        st.markdown(
            '<div class="section-title">SOLVER PERFORMANCE</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">CPU EXECUTION</div>
                    <div class="result-value">{cpu_time:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">GPU COMPUTE</div>
                    <div class="result-value">{gpu_compute_time:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">GPU END-TO-END</div>
                    <div class="result-value">{gpu_total_time:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">GPU SPEEDUP</div>
                    <div class="result-value">{speedup:.2f}×</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        st.markdown(
            '<div class="section-title">CPU vs GPU PERFORMANCE</div>',
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({
            "Platform": [
                "CPU",
                "GPU Compute",
                "GPU End-to-End"
            ],
            "Execution Time (ms)": [
                cpu_time,
                gpu_compute_time,
                gpu_total_time
            ]
        })

        st.bar_chart(
            chart_data,
            x="Platform",
            y="Execution Time (ms)",
            use_container_width=True
        )

        st.divider()

        with st.expander("◈ VIEW RAW SOLVER OUTPUT"):
            st.code(raw_output, language="text")

        st.success(
            f"⚡ GPU acceleration achieved {speedup:.2f}× speedup."
        )


st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#667060;
        padding:20px;
        letter-spacing:1px;
    ">
        GPU OPTIMIZATION SOLVER • LP • CUDA • STREAMLIT
    </div>
    """,
    unsafe_allow_html=True
)
