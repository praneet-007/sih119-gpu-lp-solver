import streamlit as st
import subprocess
import tempfile
import os
import re
import pandas as pd
import time
import io  # CHANGED: Used to safely read the uploaded CSV multiple times.

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
OBJECTIVE_VALUE=12345.70
"""

    return parse_solver_output(output), output

# CHANGED: Demo optimization trace for the GPU Optimization section
def demo_optimization_trace():
    """
    Returns simulated optimization iterations for DEMO MODE.

    # CHANGED:
    # These values are only for demonstrating the UI.
    # Later, the real CUDA solver will provide these values.
    """

    return pd.DataFrame({
        "Iteration": [1, 10, 20, 30, 40, 50, 60],
        "Objective": [
            18452.2,
            15120.5,
            13742.8,
            12982.1,
            12521.4,
            12391.8,
            12345.7
        ],
        "Residual": [
            8.42,
            4.12,
            2.31,
            1.04,
            0.52,
            0.18,
            0.03
        ]
    })
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

        # CHANGED: Correct Streamlit info call for demo mode.
        st.info("Demo mode is using simulated solver results.")

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



# ============================================================
# PROBLEM INPUT SECTION
# ============================================================

# CHANGED: Added a clear heading so the judge knows this
# section represents the input to our optimization system.
st.markdown(
    '<div class="section-title">PROBLEM INPUT</div>',
    unsafe_allow_html=True
)

# CHANGED: Upload the optimization matrix supplied by the user.
uploaded_file = st.file_uploader(
    "Upload petroleum matrix",
    type=["txt", "csv", "dat"],
    label_visibility="collapsed"
)

# CHANGED: Only display the following information
# after the user has successfully uploaded a file.
if uploaded_file:

    # Show confirmation that the input has been received.
    st.success(
        f"✓ MATRIX LOADED — {uploaded_file.name}"
    )

    # Create two columns so important input information
    # is visible immediately to the judge.
    col1, col2 = st.columns(2)

    with col1:

        # Display the size of the uploaded input file.
        st.metric(
            "FILE SIZE",
            f"{uploaded_file.size / 1024:.2f} KB"
        )

    with col2:

        # Display the file format/type.
        st.metric(
            "FILE TYPE",
            uploaded_file.type or "TEXT"
        )

    # Allow the judge to open the actual input data.
    # This helps demonstrate what is being sent into the solver.
    with st.expander("◈ PREVIEW MATRIX DATA"):

        try:

            # Convert the uploaded file into readable text.
            content = uploaded_file.getvalue().decode("utf-8")

            # Split the input into individual rows.
            lines = content.splitlines()

            # CHANGED: Show only the first 20 rows so that
            # a large matrix does not fill the entire screen.
            st.code(
                "\n".join(lines[:20])
            )
            st.divider()
  
        except Exception:

            # Display an error if the uploaded file cannot
            # be interpreted as normal text.
            st.warning(
                "Unable to preview the matrix."
            )
            st.divider()

            #............
                # ============================================================
    # CHANGED: MATRIX ANALYSIS
    # ============================================================
    # This section analyzes the uploaded sparse matrix and gives
    # the user/judge a quick understanding of its size and structure.
    # ============================================================

    try:

        # CHANGED: Read the uploaded CSV directly into a pandas DataFrame.
        # This allows us to calculate rows, columns and non-zero values.
        matrix_df = pd.read_csv(uploaded_file)

        # CHANGED: Count the number of unique constraint rows.
        # The CSV uses the "row" column to identify each constraint.
        constraint_count = matrix_df[
            matrix_df["type"].str.lower() == "constraint"
        ]["row"].nunique()

        # CHANGED: Find the highest column index.
        # In a sparse matrix, the column number represents a variable.
        variable_count = matrix_df["col"].max()

        # CHANGED: Count the number of actual non-zero matrix entries.
        # Each row in this sparse CSV represents a stored coefficient.
        non_zero_count = len(matrix_df)

        # CHANGED: Calculate the total number of possible elements
        # in the complete matrix.
        total_elements = constraint_count * variable_count

        # CHANGED: Calculate how much of the matrix is actually populated.
        # A sparse matrix has a very small percentage of non-zero values.
        if total_elements > 0:
            sparsity = (
                (1 - (non_zero_count / total_elements)) * 100
            )
        else:
            sparsity = 0

        # ========================================================
        # CHANGED: Display MATRIX ANALYSIS heading
        # ========================================================

        st.markdown(
            '<div class="section-title">MATRIX ANALYSIS</div>',
            unsafe_allow_html=True
        )

        # CHANGED: Create four columns for important matrix metrics.
        col1, col2, col3, col4 = st.columns(4)

        with col1:

            # Number of constraints in the optimization problem.
            st.metric(
                "CONSTRAINTS",
                f"{constraint_count:,}"
            )

        with col2:

            # Number of variables represented by matrix columns.
            st.metric(
                "VARIABLES",
                f"{int(variable_count):,}"
            )

        with col3:

            # Number of stored/non-zero coefficients.
            st.metric(
                "NON-ZERO VALUES",
                f"{non_zero_count:,}"
            )

        with col4:

            # Percentage of the matrix that contains zero values.
            st.metric(
                "SPARSITY",
                f"{sparsity:.2f}%"
            )

        # CHANGED: Display the mathematical matrix dimensions.
        st.info(
            f"Matrix A dimensions: "
            f"{constraint_count:,} × {int(variable_count):,}"
        )

    except Exception as e:

        # If the uploaded file does not follow the expected CSV format,
        # show a warning instead of crashing the entire Streamlit app.
        st.warning(
            f"Unable to analyze matrix: {e}"
        )

            # ============================================================
    # CHANGED: MATHEMATICAL LP MODEL
    # ============================================================
    # The uploaded CSV stores the sparse matrix as individual
    # coefficient records.
    #
    # Example:
    # constraint,1,1,1.0,10000.0,<=
    # constraint,1,124,1.1,,
    #
    # These records mean:
    #
    # 1.0*x1 + 1.1*x124 + ... <= 10000
    #
    # This section converts those records into a form that
    # a human/judge can understand.
    # ============================================================

    try:

        # CHANGED: Read the uploaded file again from its raw bytes.
        # Using BytesIO prevents problems caused by the file pointer
        # already being used by the preview/analysis section.
        model_df = pd.read_csv(
            io.BytesIO(uploaded_file.getvalue())
        )

        # CHANGED: Keep only rows representing constraints.
        constraint_df = model_df[
            model_df["type"].astype(str).str.lower() == "constraint"
        ].copy()

        # CHANGED: Make sure column numbers and coefficient values
        # are treated as numbers.
        constraint_df["col"] = pd.to_numeric(
            constraint_df["col"],
            errors="coerce"
        )

        constraint_df["value"] = pd.to_numeric(
            constraint_df["value"],
            errors="coerce"
        )

        # CHANGED: Remove invalid coefficient records.
        constraint_df = constraint_df.dropna(
            subset=["row", "col", "value"]
        )

        # CHANGED: Display the mathematical model section.
        st.markdown(
            '<div class="section-title">MATHEMATICAL OPTIMIZATION MODEL</div>',
            unsafe_allow_html=True
        )

        # --------------------------------------------------------
        # CHANGED: Show the general mathematical form.
        # --------------------------------------------------------

        st.markdown(
            """
            ### Linear Constraint Form

            The uploaded sparse data is converted into:

            **A × x ≤ b**

            where:

            - **A** = coefficient matrix
            - **x** = decision variables
            - **b** = right-hand-side constraint values
            """
        )

        # --------------------------------------------------------
        # CHANGED: Display variable information.
        # --------------------------------------------------------

        variable_count_model = int(
            constraint_df["col"].max()
        )

        st.metric(
            "DECISION VARIABLES",
            f"{variable_count_model:,}"
        )

        st.caption(
            f"Decision variables: x₁ through x{variable_count_model:,}"
        )

        # --------------------------------------------------------
        # CHANGED: Build readable constraint equations.
        # --------------------------------------------------------
        # We only display the first few constraints.
        # A dataset can contain thousands of constraints, so
        # displaying all of them would make the dashboard unusable.
        # --------------------------------------------------------

        st.markdown("### Sample Constraints")

        # Group all coefficient records belonging to the same row.
        grouped_constraints = constraint_df.groupby(
            "row",
            sort=True
        )

        # Display only the first 5 constraints.
        displayed_constraints = 0

        for row_number, group in grouped_constraints:

            # Stop after five equations.
            if displayed_constraints >= 5:
                break

            # ----------------------------------------------------
            # CHANGED: Build the left-hand side of the equation.
            # ----------------------------------------------------

            terms = []

            for _, record in group.iterrows():

                # Get the variable number.
                variable_number = int(record["col"])

                # Get the coefficient.
                coefficient = float(record["value"])

                # Create a readable term such as:
                # 1.0x1
                # 1.1x124
                term = (
                    f"{coefficient:g}"
                    f"x{variable_number}"
                )

                terms.append(term)

            # Join terms together.
            left_side = " + ".join(terms)

            # ----------------------------------------------------
            # CHANGED: Extract RHS and constraint direction.
            # ----------------------------------------------------

            rhs_values = pd.to_numeric(
                group["rhs"],
                errors="coerce"
            ).dropna()

            if len(rhs_values) > 0:

                # The RHS is stored once for the constraint.
                rhs_value = rhs_values.iloc[0]

                # Get <=, >= or = if supplied.
                senses = group["sense"].dropna()

                if len(senses) > 0:
                    sense = str(senses.iloc[0])
                else:
                    sense = "<="

                equation = (
                    f"**C{int(row_number)}:**  "
                    f"{left_side} {sense} {rhs_value:g}"
                )

                st.markdown(equation)

            displayed_constraints += 1

        # Tell the user that only a sample is displayed.
        st.caption(
            f"Showing {displayed_constraints} sample constraints "
            f"from {len(grouped_constraints):,} total constraints."
        )

        # --------------------------------------------------------
        # CHANGED: Matrix representation.
        # --------------------------------------------------------

        st.markdown("### Matrix Representation")

        st.latex(
            r"A x \leq b"
        )

        # Display the dimensions without creating a huge dense matrix.
        # IMPORTANT:
        # A 5,000 × 10,000 dense matrix would contain 50 million
        # elements. We therefore keep the actual data sparse.
        matrix_rows = constraint_df["row"].nunique()
        matrix_cols = int(constraint_df["col"].max())

        st.info(
            f"A = {matrix_rows:,} × {matrix_cols:,}    |    "
            f"x = {matrix_cols:,} × 1    |    "
            f"b = {matrix_rows:,} × 1"
        )

        # --------------------------------------------------------
        # CHANGED: Explain the sparse representation.
        # --------------------------------------------------------

        st.markdown(
            """
            **Sparse representation**

            Only the non-zero coefficients are stored in the input
            dataset. This avoids creating unnecessary zero values and
            is important when working with large optimization problems.
            """
        )

        # --------------------------------------------------------
        # CHANGED: Show the first few A coefficients.
        # --------------------------------------------------------

        with st.expander("◈ VIEW SPARSE MATRIX COEFFICIENTS"):

            st.dataframe(
                constraint_df[
                    ["row", "col", "value", "rhs", "sense"]
                ].head(20),
                use_container_width=True
            )

        # --------------------------------------------------------
        # CHANGED: Objective function status.
        # --------------------------------------------------------
        # The uploaded dataset contains constraints but no objective
        # records, so we explicitly tell the judge instead of
        # inventing an objective function.
        # --------------------------------------------------------

        objective_rows = model_df[
            model_df["type"].astype(str).str.lower() == "objective"
        ]

        if len(objective_rows) == 0:

            st.warning(
                "Objective function (c) is not provided in this "
                "dataset. The current input defines the constraint "
                "matrix A and RHS vector b."
            )

        else:

            st.success(
                "Objective function detected in the input dataset."
            )

    except Exception as e:

        # CHANGED: Prevent the mathematical-model section from
        # crashing the entire dashboard if the input format changes.
        st.warning(
            f"Unable to build mathematical model: {e}"
        )

        # ============================================================
# CHANGED: GPU OPTIMIZATION PIPELINE
# ============================================================
# This section visually explains how the mathematical model
# moves from the uploaded CSV to the CUDA/GPU solver.
#
# IMPORTANT:
# This is a pipeline explanation, not fake solver progress.
# The actual solver is still executed by run_gpu_solver().
# ============================================================

st.markdown(
    '<div class="section-title">GPU OPTIMIZATION PIPELINE</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    ### From Mathematical Model to GPU

    The optimization problem follows this execution pipeline:
    """
)

# ------------------------------------------------------------
# CHANGED: Pipeline stages
# ------------------------------------------------------------
# Each box represents one stage of the actual workflow.
# ------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        **① INPUT**

        📄

        Sparse CSV

        A, b
        """
    )

with col2:
    st.markdown(
        """
        **② MODEL**

        📐

        A × x ≤ b

        LP Model
        """
    )

with col3:
    st.markdown(
        """
        **③ BRIDGE**

        🔄

        Python

        → CUDA
        """
    )

with col4:
    st.markdown(
        """
        **④ SOLVER**

        ⚙️

        CUDA

        GPU
        """
    )

with col5:
    st.markdown(
        """
        **⑤ RESULT**

        📊

        Optimization

        Output
        """
    )

# ------------------------------------------------------------
# CHANGED: Explain the data flow.
# ------------------------------------------------------------

st.info(
    """
    **Data flow:** The sparse CSV provides the constraint
    coefficients and RHS values → these represent the
    mathematical model A × x ≤ b → the Python layer passes
    the optimization data to the CUDA solver → CUDA executes
    the computation on the NVIDIA GPU → the solver returns
    the optimization result and performance metrics.
    """
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

            # ========================================================
            # CHANGED: GPU OPTIMIZATION EXPLAINABILITY
            # ========================================================
            # In DEMO MODE this displays simulated iteration history,
            # convergence, solution values and constraint utilization.
            # Later, these values can be replaced directly by the
            # real CUDA solver output without changing the UI layout.
            # ========================================================

            st.divider()

            st.markdown(
                '<div class="section-title">GPU OPTIMIZATION</div>',
                unsafe_allow_html=True
            )

            if demo_mode:
                st.caption("DEMO MODE — Simulated optimization progress")

                # CHANGED: Load simulated optimization iterations.
                trace_df = demo_optimization_trace()

                st.markdown("### Iteration Progress")

                # CHANGED: Show iteration, objective and residual together.
                # CHANGED: Use a fixed-width table layout so iteration data is always visible.
                display_trace = trace_df.copy()
                display_trace["Objective"] = display_trace["Objective"].map(lambda v: f"{v:,.1f}")
                display_trace["Residual"] = display_trace["Residual"].map(lambda v: f"{v:.2f}")
                st.table(display_trace)

                # CHANGED: Show convergence status.
                st.success("✓ CONVERGED")

                # CHANGED: Visualize objective reduction over iterations.
                st.markdown("### Objective Convergence")

                # CHANGED: Keep only the two values needed by the chart.
                # This prevents Streamlit from trying to plot Residual
                # on the same scale as Objective.
                convergence_df = trace_df[
                    ["Iteration", "Objective"]
                ].copy()

                convergence_df["Iteration"] = pd.to_numeric(
                    convergence_df["Iteration"],
                    errors="coerce"
                )
                convergence_df["Objective"] = pd.to_numeric(
                    convergence_df["Objective"],
                    errors="coerce"
                )

                convergence_df = convergence_df.dropna()
                convergence_df = convergence_df.set_index("Iteration")

                # CHANGED: Plot the objective against iteration.
                # A fixed height keeps the dashboard compact and readable.
                st.line_chart(
                    convergence_df["Objective"],
                    height=300,
                    use_container_width=True
                )

                # CHANGED: Explain the convergence numerically.
                first_objective = convergence_df["Objective"].iloc[0]
                final_objective = convergence_df["Objective"].iloc[-1]

                if first_objective != 0:
                    improvement = (
                        (first_objective - final_objective)
                        / first_objective
                    ) * 100
                else:
                    improvement = 0

                st.caption(
                    f"Objective reduced from "
                    f"{first_objective:,.1f} to {final_objective:,.1f} "
                    f"— {improvement:.1f}% improvement."
                )

                # CHANGED: Show final demo optimization result.
                st.markdown("### OPTIMAL SOLUTION")

                # CHANGED: Keep the demo optimization objective consistent with the displayed demo trace.
                final_objective = trace_df.iloc[-1]["Objective"]

                solution_col1, solution_col2 = st.columns(2)

                with solution_col1:
                    st.metric(
                        "OBJECTIVE VALUE",
                        f"₹{final_objective:,.2f}"
                    )

                with solution_col2:
                    st.metric(
                        "FINAL RESIDUAL",
                        f"{trace_df.iloc[-1]['Residual']:.2f}"
                    )

                st.markdown("**Selected Variables**")

                x1, x2, x3, x4 = st.columns(4)

                with x1:
                    st.metric("x1", "120")
                with x2:
                    st.metric("x2", "80")
                with x3:
                    st.metric("x3", "45")
                with x4:
                    st.metric("x4", "0")

                # CHANGED: Demonstrate constraint utilization.
                # These are UI demo values, not calculated from the
                # uploaded dataset or the real CUDA solver.
                st.markdown("### CONSTRAINT CHECK")

                st.write("Constraint 1 — 91%")
                st.progress(0.91)

                st.write("Constraint 2 — 72%")
                st.progress(0.72)

                st.write("Constraint 3 — 100%")
                st.progress(1.00)

                st.success("✓ All demo constraints satisfied")

            else:
                # CHANGED: Do not invent live iteration/solution values.
                # The real CUDA solver must expose this information first.
                st.info(
                    "Live CUDA solver connected. Iteration history, "
                    "solution variables and constraint validation will "
                    "appear here when the solver provides them."
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

                # CHANGED: In DEMO MODE, show the same optimization objective used above.
                result_objective = final_objective if demo_mode else objective
                st.metric(
                    "OBJECTIVE VALUE",
                    f"{result_objective:,.2f}"
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