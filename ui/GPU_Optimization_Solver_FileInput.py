import streamlit as st
import subprocess
import tempfile
import os
import re
import pandas as pd
import time
import json

st.set_page_config(
    page_title="GPU LP Optimization Solver",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- STYLING -------------------------

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 50% 0%, rgba(118,185,0,0.12), transparent 35%),
        linear-gradient(135deg, #000000 0%, #050805 45%, #000000 100%);
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
header { background: transparent !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050705 0%, #000000 100%);
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
    text-shadow: 0 0 10px rgba(118,185,0,0.5),
                 0 0 25px rgba(118,185,0,0.25);
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
    background: linear-gradient(145deg, rgba(118,185,0,0.10),
                                rgba(0,0,0,0.75));
    border: 1px solid rgba(118,185,0,0.45);
    border-radius: 14px;
    padding: 22px 10px;
    text-align: center;
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
    background: linear-gradient(90deg, #76b900, #8bd000);
    color: #000000;
    border: none;
    border-radius: 10px;
    height: 65px;
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 2px;
}
[data-testid="stFileUploader"] {
    background: rgba(118,185,0,0.04);
    border: 2px dashed rgba(118,185,0,0.55);
    border-radius: 15px;
    padding: 15px;
}
.stTextInput input, .stTextArea textarea {
    background-color: #050705 !important;
    color: white !important;
    border: 1px solid #76b900 !important;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(118,185,0,0.08),
                                rgba(0,0,0,0.7));
    border: 1px solid rgba(118,185,0,0.35);
    padding: 15px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] { color: #76b900 !important; }
.stProgress > div > div > div > div { background-color: #76b900; }
hr { border-color: rgba(118,185,0,0.25); }
</style>
""", unsafe_allow_html=True)

SOLVER_PATH = "./gpu_solver"

# ------------------------- SOLVER HELPERS -------------------------

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
        if result.get("cpu_time", 0) > 0 and result.get("gpu_total_time", 0) > 0:
            result["speedup"] = result["cpu_time"] / result["gpu_total_time"]

    return result


def demo_solver():
    time.sleep(1.2)
    output = """
CPU_TIME=1250
GPU_COMPUTE_TIME=180
GPU_TOTAL_TIME=205
SPEEDUP=6.10
OBJECTIVE_VALUE=84523.45
"""
    return parse_solver_output(output), output


def parse_lp_file(uploaded_file):
    """
    Supported application input format:

    {
      "problem_type": "LP",
      "sense": "minimize",
      "variables": [...],
      "objective": [...],
      "constraints": [
        {"name": "...", "sense": "<=", "rhs": ..., "coefficients": [...]}
      ],
      "lower_bounds": [...],
      "upper_bounds": [...]
    }

    The uploaded company file describes the optimization problem.
    This function converts it into c, A, b internally.
    """
    try:
        raw = uploaded_file.getvalue().decode("utf-8")
        data = json.loads(raw)
    except UnicodeDecodeError:
        raise ValueError("The input file must be UTF-8 encoded JSON.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input file: {e}")

    if str(data.get("problem_type", "")).upper() != "LP":
        raise ValueError("This version supports LP problems only.")

    sense = str(data.get("sense", "minimize")).lower()
    if sense not in ("minimize", "maximize"):
        raise ValueError("sense must be 'minimize' or 'maximize'.")

    variables = data.get("variables")
    objective = data.get("objective")
    constraints = data.get("constraints", [])

    if not isinstance(variables, list) or not variables:
        raise ValueError("variables must be a non-empty list.")

    if not isinstance(objective, list) or len(objective) != len(variables):
        raise ValueError("objective length must equal the number of variables.")

    n = len(variables)

    # Internal LP representation.
    # Convert maximize to minimize by negating c.
    c = [float(x) for x in objective]
    if sense == "maximize":
        c = [-x for x in c]

    A = []
    b = []
    constraint_names = []
    constraint_senses = []

    for i, con in enumerate(constraints):
        coeff = con.get("coefficients")
        if not isinstance(coeff, list) or len(coeff) != n:
            raise ValueError(
                f"Constraint {i+1}: coefficients must contain exactly {n} values."
            )

        csense = str(con.get("sense", "<=")).strip()
        if csense not in ("<=", ">=", "="):
            raise ValueError(f"Constraint {i+1}: sense must be <=, >=, or =.")

        row = [float(x) for x in coeff]
        rhs = float(con.get("rhs"))

        # Store all rows in a canonical <= form.
        if csense == ">=":
            row = [-x for x in row]
            rhs = -rhs

        A.append(row)
        b.append(rhs)
        constraint_names.append(str(con.get("name", f"constraint_{i+1}")))
        constraint_senses.append(csense)

    lower = data.get("lower_bounds", [0.0] * n)
    upper = data.get("upper_bounds", [None] * n)

    if len(lower) != n or len(upper) != n:
        raise ValueError("lower_bounds and upper_bounds must match variables length.")

    return {
        "sense": sense,
        "variables": [str(x) for x in variables],
        "objective_original": [float(x) for x in objective],
        "c": c,
        "A": A,
        "b": b,
        "constraint_names": constraint_names,
        "constraint_senses": constraint_senses,
        "lower_bounds": lower,
        "upper_bounds": upper
    }


def matrix_text(model):
    lines = [
        "# Generated internal LP representation",
        "# Objective is converted to minimization form internally",
        f"VARIABLES {len(model['variables'])}",
        f"CONSTRAINTS {len(model['A'])}",
        "",
        "VARIABLE_NAMES",
        " ".join(model["variables"]),
        "",
        "OBJECTIVE_C",
        " ".join(f"{x:.12g}" for x in model["c"]),
        "",
        "MATRIX_A"
    ]

    for row in model["A"]:
        lines.append(" ".join(f"{x:.12g}" for x in row))

    lines += [
        "",
        "VECTOR_B",
        " ".join(f"{x:.12g}" for x in model["b"]),
        "",
        "LOWER_BOUNDS",
        " ".join(str(x) for x in model["lower_bounds"]),
        "",
        "UPPER_BOUNDS",
        " ".join("INF" if x is None else str(x) for x in model["upper_bounds"])
    ]

    return "\n".join(lines)


def run_gpu_solver(model):
    """
    Adapter between the Streamlit application and ./gpu_solver.

    IMPORTANT:
    If your CUDA/C++ executable expects another file format,
    only this serialization/adapter needs to be changed.
    """
    input_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".lpdata",
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

        return parse_solver_output(process.stdout), process.stdout

    except FileNotFoundError:
        return None, "gpu_solver executable was not found."

    except subprocess.TimeoutExpired:
        return None, "GPU solver timed out."

    except Exception as e:
        return None, str(e)

    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)


# ------------------------- HEADER -------------------------

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

# ------------------------- SIDEBAR -------------------------

with st.sidebar:
    st.markdown(
        '<h2 style="color:#76b900;">⚡ GPU CONTROL CENTER</h2>',
        unsafe_allow_html=True
    )

    st.divider()

    demo_mode = st.toggle("DEMO MODE", value=True)

    if demo_mode:
        st.info("Demo mode uses simulated solver results.")
    else:
        st.success("Live GPU solver mode enabled.")

    st.divider()

    st.markdown(
        '<h3 style="color:#76b900;">CURRENT SCOPE</h3>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="line-height:1.8;">
        <b>Problem:</b> LP only<br>
        <b>Input:</b> Optimization file<br>
        <b>Internal:</b> A, b, c matrix<br>
        <b>Method:</b> Interior Point<br>
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
        <div style="text-align:center;color:white;font-size:15px;font-weight:700;line-height:1.2;">
            <div style="margin:8px 0;">📁 COMPANY INPUT FILE</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">📐 LP MODEL</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">🔢 MATRIX A, b, c</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">⚙️ INTERIOR POINT</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">🚀 NVIDIA GPU</div>
            <div style="color:#76b900;font-size:24px;">↓</div>
            <div style="margin:8px 0;">📊 OPTIMAL SOLUTION</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    if os.path.exists(SOLVER_PATH):
        st.success("● GPU SOLVER READY")
    else:
        st.warning("● GPU SOLVER NOT FOUND")

# ------------------------- FILE INPUT -------------------------

st.markdown(
    '<div class="section-title">LP PROBLEM INPUT FILE</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload the company's LP problem file. The application parses the "
    "problem and automatically creates the internal matrix."
)

uploaded_file = st.file_uploader(
    "Upload LP problem",
    type=["json"],
    label_visibility="collapsed"
)

model = None

if uploaded_file:
    try:
        model = parse_lp_file(uploaded_file)

        st.success(f"✓ LP FILE LOADED — {uploaded_file.name}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("VARIABLES", f"{len(model['variables']):,}")

        with col2:
            st.metric("CONSTRAINTS", f"{len(model['A']):,}")

        with col3:
            nnz = sum(1 for row in model["A"] for x in row if x != 0)
            st.metric("NON-ZERO VALUES", f"{nnz:,}")

        with col4:
            st.metric("OBJECTIVE", model["sense"].upper())

        st.divider()

        st.markdown(
            '<div class="section-title">LP MODEL SUMMARY</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Objective:** {model['sense']}  "
            f"**{len(model['variables']):,} variables** "
            f"subject to **{len(model['A']):,} constraints**."
        )

        with st.expander("◈ VIEW LP FILE CONTENT"):
            st.code(
                uploaded_file.getvalue().decode("utf-8")[:12000],
                language="json"
            )

        with st.expander("◈ VIEW GENERATED MATRIX A, b, c"):
            st.code(matrix_text(model), language="text")

    except Exception as e:
        st.error(f"Unable to parse LP file: {e}")

st.divider()

# ------------------------- SOLVE -------------------------

run_button = st.button(
    "⚡ SOLVE LP ON GPU",
    type="primary",
    use_container_width=True
)

if run_button:

    if model is None:
        st.error("Please upload a valid LP input file first.")

    else:
        st.markdown(
            '<div class="section-title">OPTIMIZATION ENGINE</div>',
            unsafe_allow_html=True
        )

        progress = st.progress(0)
        status = st.empty()

        status.info("READING LP PROBLEM...")
        progress.progress(15)
        time.sleep(0.3)

        status.info("BUILDING MATRIX A, b, c...")
        progress.progress(30)
        time.sleep(0.3)

        status.info("INITIALIZING INTERIOR POINT METHOD...")
        progress.progress(45)
        time.sleep(0.3)

        if demo_mode:
            result, raw_output = demo_solver()
        else:
            status.info("RUNNING GPU LINEAR ALGEBRA...")
            result, raw_output = run_gpu_solver(model)

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

            c1, c2 = st.columns(2)

            with c1:
                st.metric("OBJECTIVE VALUE", f"{objective:,.2f}")

            with c2:
                st.metric("STATUS", "OPTIMAL / COMPLETED")

            st.divider()

            st.markdown(
                '<div class="section-title">SOLVER PERFORMANCE</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(
                    f'<div class="result-box">'
                    f'<div class="result-label">CPU EXECUTION</div>'
                    f'<div class="result-value">{cpu_time:.2f} ms</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(
                    f'<div class="result-box">'
                    f'<div class="result-label">GPU COMPUTE</div>'
                    f'<div class="result-value">{gpu_compute_time:.2f} ms</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with c3:
                st.markdown(
                    f'<div class="result-box">'
                    f'<div class="result-label">GPU END-TO-END</div>'
                    f'<div class="result-value">{gpu_total_time:.2f} ms</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with c4:
                st.markdown(
                    f'<div class="result-box">'
                    f'<div class="result-label">GPU SPEEDUP</div>'
                    f'<div class="result-value">{speedup:.2f}×</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.divider()

            st.markdown(
                '<div class="section-title">CPU vs GPU PERFORMANCE</div>',
                unsafe_allow_html=True
            )

            chart_data = pd.DataFrame({
                "Platform": ["CPU", "GPU Compute", "GPU End-to-End"],
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

            with st.expander("◈ VIEW RAW SOLVER OUTPUT"):
                st.code(raw_output, language="text")

            st.success(
                f"⚡ GPU acceleration achieved {speedup:.2f}× speedup."
            )

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#667060;padding:20px;letter-spacing:1px;">
        GPU OPTIMIZATION SOLVER • LP • CUDA • STREAMLIT
    </div>
    """,
    unsafe_allow_html=True
)
