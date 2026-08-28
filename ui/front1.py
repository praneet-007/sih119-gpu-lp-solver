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

                result["speedup"] = (
                    result["cpu_time"] /
                    result["gpu_total_time"]
                )

    return result


def run_gpu_solver(uploaded_file):

    input_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".txt"
        ) as temp:

            temp.write(uploaded_file.getvalue())
            input_path = temp.name

        process = subprocess.run(
            [SOLVER_PATH, input_path],
            capture_output=True,
            text=True,
            timeout=300
        )

        if process.returncode != 0:

            return None, process.stderr

        result = parse_solver_output(
            process.stdout
        )

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

    time.sleep(2)

    output = """
CPU_TIME=1250
GPU_COMPUTE_TIME=180
GPU_TOTAL_TIME=205
SPEEDUP=6.10
OBJECTIVE_VALUE=84523.45
"""

    return parse_solver_output(output), output


st.markdown(
    '<div class="main-title">⚡ GPU OPTIMIZATION SOLVER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'INDIGENOUS GPU-ACCELERATED PETROLEUM OPTIMIZATION ENGINE'
    '</div>',
    unsafe_allow_html=True
)


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

        st.info
        (
            "Demo mode is using simulated solver results."
        )

    else:

        st.success(
            "Live solver mode enabled."
        )

    st.divider()

    st.markdown(
        '<h3 style="color:#76b900;">SOLVER STATUS</h3>',
        unsafe_allow_html=True
    )

    if os.path.exists(SOLVER_PATH):

        st.success(
            "● GPU SOLVER READY"
        )

    else:

        st.warning(
            "● GPU SOLVER NOT FOUND"
        )

    st.divider()

    st.markdown(
    '<h3 style="color:#76b900;">PIPELINE</h3>',
    unsafe_allow_html=True)

    st.markdown("""
    <div style="
        text-align: center;
        color: white;
        font-size: 15px;
        font-weight: 700;
        line-height: 1.2;
    ">

    <div style="margin: 8px 0;">
        📁 MATRIX INPUT
    </div>

    <div style="
        color: #76b900;
        font-size: 24px;
        font-weight: 900;
        margin: 2px 0;
    ">
        ↓
    </div>

    <div style="margin: 8px 0;">
        🐍 PYTHON BRIDGE
    </div>

    <div style="
        color: #76b900;
        font-size: 24px;
        font-weight: 900;
        margin: 2px 0;
    ">
        ↓
    </div>

    <div style="margin: 8px 0;">
        ⚙️ CUDA SOLVER
    </div>

    <div style="
        color: #76b900;
        font-size: 24px;
        font-weight: 900;
        margin: 2px 0;
    ">
        ↓
    </div>

    <div style="margin: 8px 0;">
        🚀 NVIDIA GPU
    </div>

    <div style="
        color: #76b900;
        font-size: 24px;
        font-weight: 900;
        margin: 2px 0;
    ">
        ↓
    </div>

    <div style="margin: 8px 0;">
        📊 PERFORMANCE
    </div>

    </div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">PETROLEUM MATRIX INPUT</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload petroleum matrix",
    type=["txt", "csv", "dat"],
    label_visibility="collapsed"
)


if uploaded_file:

    st.success(
        f"✓ MATRIX LOADED — {uploaded_file.name}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "FILE SIZE",
            f"{uploaded_file.size / 1024:.2f} KB"
        )

    with col2:

        st.metric(
            "FILE TYPE",
            uploaded_file.type or "TEXT"
        )

    with st.expander(
        "◈ PREVIEW MATRIX DATA"
    ):

        try:

            content = uploaded_file.getvalue().decode(
                "utf-8"
            )

            lines = content.splitlines()

            st.code(
                "\n".join(lines[:20])
            )

        except Exception:

            st.warning(
                "Unable to preview the matrix."
            )


st.divider()

run_button = st.button(
    "⚡ RUN OPTIMIZATION",
    type="primary",
    use_container_width=True
)


if run_button:

    if uploaded_file is None:

        st.error(
            "Please upload a petroleum matrix first."
        )

    else:

        st.markdown(
            '<div class="section-title">'
            'OPTIMIZATION ENGINE'
            '</div>',
            unsafe_allow_html=True
        )

        progress = st.progress(0)

        status = st.empty()

        status.info(
            "INITIALIZING OPTIMIZATION ENGINE..."
        )

        progress.progress(20)

        time.sleep(0.5)

        status.info(
            "PROCESSING MATRIX..."
        )

        progress.progress(40)

        if demo_mode:

            result, raw_output = demo_solver()

        else:

            result, raw_output = run_gpu_solver(
                uploaded_file
            )

        progress.progress(80)

        if result is None:

            progress.progress(0)

            status.error(
                "OPTIMIZATION FAILED"
            )

            st.error(raw_output)

        else:

            progress.progress(100)

            status.success(
                "✓ OPTIMIZATION COMPLETED"
            )

            cpu_time = result.get(
                "cpu_time",
                0
            )

            gpu_compute_time = result.get(
                "gpu_compute_time",
                0
            )

            gpu_total_time = result.get(
                "gpu_total_time",
                0
            )

            speedup = result.get(
                "speedup",
                0
            )

            objective = result.get(
                "objective",
                0
            )

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'PERFORMANCE DASHBOARD'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            CPU EXECUTION
                        </div>
                        <div class="result-value">
                            {cpu_time:.2f} ms
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            GPU COMPUTE
                        </div>
                        <div class="result-value">
                            {gpu_compute_time:.2f} ms
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            GPU END-TO-END
                        </div>
                        <div class="result-value">
                            {gpu_total_time:.2f} ms
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col4:

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-label">
                            GPU SPEEDUP
                        </div>
                        <div class="result-value">
                            {speedup:.2f}×
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()

            st.markdown(
                '<div class="section-title">'
                'CPU vs GPU PERFORMANCE'
                '</div>',
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

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="section-title">'
                    'OPTIMIZATION RESULT'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.metric(
                    "OBJECTIVE VALUE",
                    f"{objective:,.2f}"
                )

            with col2:

                st.markdown(
                    '<div class="section-title">'
                    'ACCELERATION'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.metric(
                    "GPU SPEEDUP",
                    f"{speedup:.2f}×"
                )

            st.divider()

            with st.expander(
                "◈ VIEW RAW SOLVER OUTPUT"
            ):

                st.code(
                    raw_output,
                    language="text"
                )

            st.success(
                f"⚡ GPU acceleration achieved "
                f"{speedup:.2f}× speedup."
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
        GPU OPTIMIZATION SOLVER • CUDA • STREAMLIT
    </div>
    """,
    unsafe_allow_html=True
)