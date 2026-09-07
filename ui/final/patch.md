# Patches for frontend_cuda_crusaders_v2.py

The frontend file is ~3.5MB (it embeds base64 images), too large to safely
rewrite in full here. Below are exact find/replace patches — apply each
with your editor's find/replace (or `patch`/manual copy-paste). Every
"FIND" block is copied verbatim from your file, so a plain string search
will locate it uniquely.

---

## Patch 1 — LPProblem: add a source_format field

FIND (around line 711):
```python
@dataclass
class LPProblem:
    sense: str                 # "max" or "min"
    var_names: list
    c: np.ndarray               # objective coefficients, aligned to var_names, in MAX sense
    A: "sparse.csr_matrix"
    b: np.ndarray
    row_labels: list = field(default_factory=list)
    source_text: str = ""

    @property
    def m(self):
        return self.A.shape[0]
```

REPLACE WITH:
```python
@dataclass
class LPProblem:
    sense: str                 # "max" or "min"
    var_names: list
    c: np.ndarray               # objective coefficients, aligned to var_names, in MAX sense
    A: "sparse.csr_matrix"
    b: np.ndarray
    row_labels: list = field(default_factory=list)
    source_text: str = ""
    source_format: str = "LP"   # "MPS" / "LP" / "CSV" -- which uploaded format this came from
    objective_is_absent: bool = False  # CSV only: True if the CSV had no objective rows

    @property
    def m(self):
        return self.A.shape[0]
```

---

## Patch 2 — insert parse_csv_text() after parse_mps_text()

FIND (around line 998-1006, the end of parse_mps_text and the following
section header):
```python
    # The raw MPS text can be tens of MB; nothing in the UI reads it back,
    # so it isn't kept in session state for large uploads.
    kept_source = text if len(text) <= 200_000 else ""
    return LPProblem(sense="min", var_names=var_order, c=c, A=A_signed.tocsr(),
                      b=np.asarray(b_signed, dtype=float),
                      row_labels=row_labels, source_text=kept_source)


# ============================================================================
# REFERENCE PDHG SOLVER (real algorithm, real measured timing per stage)
# ============================================================================
```

REPLACE WITH (this inserts the new function between the two, changing
nothing else):
```python
    # The raw MPS text can be tens of MB; nothing in the UI reads it back,
    # so it isn't kept in session state for large uploads.
    kept_source = text if len(text) <= 200_000 else ""
    return LPProblem(sense="min", var_names=var_order, c=c, A=A_signed.tocsr(),
                      b=np.asarray(b_signed, dtype=float),
                      row_labels=row_labels, source_text=kept_source)


CSV_REQUIRED_COLUMNS = ["type", "row", "col", "value", "rhs", "sense"]
CSV_VALID_TYPES = {"constraint", "objective"}
CSV_SENSE_ALIASES = {"<=": "<=", "le": "<=", ">=": ">=", "ge": ">=", "=": "=", "==": "=", "eq": "="}


def _csv_float(raw, line_no, field_name):
    raw = (raw or "").strip()
    if raw == "":
        raise ValueError(f"Missing value for '{field_name}' on CSV row {line_no}.")
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Invalid numeric value for '{field_name}' on CSV row {line_no}: {raw!r}") from None


def _csv_index(raw, line_no, field_name):
    raw = (raw or "").strip()
    if raw == "":
        raise ValueError(f"Missing value for '{field_name}' on CSV row {line_no}.")
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"Invalid index for '{field_name}' on CSV row {line_no}: {raw!r} (must be a non-negative integer)") from None
    if val < 0:
        raise ValueError(f"Invalid index for '{field_name}' on CSV row {line_no}: {val} (indices must be >= 0)")
    return val


def parse_csv_text(text: str) -> LPProblem:
    """Parses the type,row,col,value,rhs,sense CSV schema into an LPProblem,
    using the SAME canonical A x <= b, x >= 0 representation parse_lp_text()
    and parse_mps_text() produce (equalities mirrored into a <= pair, >=
    rows sign-flipped) -- so every downstream stage (GPU prep, PDHG, CPU
    benchmark, analytics) stays completely format-independent.

    CSV objective rows are interpreted directly as MINIMIZE c^T x
    coefficients (the same convention gpu_format.py's minimize-only format
    uses). If the CSV has no 'objective' rows at all, the objective is left
    all-zero (never fabricated) and problem.objective_is_absent is set so
    the caller can warn the user.

        type,row,col,value,rhs,sense
        objective,,0,3,,
        constraint,0,0,1,10,<=
    """
    if not text or not text.strip():
        raise ValueError("CSV file is empty.")

    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV file is empty.") from None

    header_norm = [h.strip().lower() for h in header]
    missing = [c for c in CSV_REQUIRED_COLUMNS if c not in header_norm]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
    col_idx = {c: header_norm.index(c) for c in CSV_REQUIRED_COLUMNS}

    constraints: dict = {}
    objective: dict = {}
    max_col = -1
    any_data_row = False

    for line_no, fields in enumerate(reader, start=2):
        if not fields or all((f or "").strip() == "" for f in fields):
            continue
        if len(fields) < len(header):
            raise ValueError(f"Malformed CSV row {line_no}: expected {len(header)} fields, got {len(fields)}.")
        any_data_row = True

        row_type = (fields[col_idx["type"]] or "").strip().lower()
        if row_type not in CSV_VALID_TYPES:
            raise ValueError(
                f"Invalid 'type' value on CSV row {line_no}: {fields[col_idx['type']]!r} "
                f"(expected 'constraint' or 'objective')."
            )

        col = _csv_index(fields[col_idx["col"]], line_no, "col")
        value = _csv_float(fields[col_idx["value"]], line_no, "value")
        max_col = max(max_col, col)

        if row_type == "objective":
            objective[col] = objective.get(col, 0.0) + value
            continue

        row = _csv_index(fields[col_idx["row"]], line_no, "row")
        rhs = _csv_float(fields[col_idx["rhs"]], line_no, "rhs")
        sense_raw = (fields[col_idx["sense"]] or "").strip().lower()
        sense = CSV_SENSE_ALIASES.get(sense_raw)
        if sense is None:
            raise ValueError(
                f"Invalid constraint sense on CSV row {line_no}: {fields[col_idx['sense']]!r} "
                f"(expected one of '<=', '>=', '=')."
            )
        entry = constraints.setdefault(row, {"coeffs": {}, "rhs": rhs, "sense": sense})
        if entry["rhs"] != rhs:
            raise ValueError(
                f"Inconsistent rhs for constraint row {row} on CSV row {line_no}: "
                f"{entry['rhs']!r} vs {rhs!r} (every line for the same row must agree)."
            )
        if entry["sense"] != sense:
            raise ValueError(
                f"Inconsistent sense for constraint row {row} on CSV row {line_no}: "
                f"{entry['sense']!r} vs {sense!r} (every line for the same row must agree)."
            )
        entry["coeffs"][col] = entry["coeffs"].get(col, 0.0) + value

    if not any_data_row:
        raise ValueError("CSV has a header but no data rows.")
    if not constraints:
        raise ValueError("CSV must contain at least one constraint row (type=constraint).")

    row_ids = sorted(constraints.keys())
    expected = list(range(len(row_ids)))
    if row_ids != expected:
        found_missing = sorted(set(expected) - set(row_ids))
        raise ValueError(
            f"CSV is missing constraint row(s) {found_missing} -- 'row' indices must be "
            f"a contiguous 0-based range with no gaps (found rows {row_ids})."
        )

    n = max(max_col + 1, (max(objective.keys()) + 1) if objective else 0)
    if n == 0:
        raise ValueError("CSV has no variable columns.")
    var_names = [f"x{j}" for j in range(n)]

    data, indices, indptr = [], [], [0]
    b = []
    row_labels = []
    for row in row_ids:
        entry = constraints[row]
        sense = entry["sense"]
        sign = -1.0 if sense == ">=" else 1.0
        pairs = sorted(((c, v * sign) for c, v in entry["coeffs"].items()), key=lambda p: p[0])
        for j, val in pairs:
            indices.append(j); data.append(val)
        indptr.append(len(data))
        b.append(entry["rhs"] * sign)
        row_labels.append(f"CSV constraint row {row} ({sense})")
        if sense == "=":
            pairs2 = sorted(((c, -v) for c, v in entry["coeffs"].items()), key=lambda p: p[0])
            for j, val in pairs2:
                indices.append(j); data.append(val)
            indptr.append(len(data))
            b.append(-entry["rhs"])
            row_labels.append(f"CSV constraint row {row} (=, mirrored)")

    A = sparse.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, n))
    b_arr = np.array(b, dtype=float)

    c = np.zeros(n)
    for col, val in objective.items():
        c[col] = val

    kept_source = text if len(text) <= 200_000 else ""
    problem = LPProblem(sense="min", var_names=var_names, c=c, A=A, b=b_arr,
                         row_labels=row_labels, source_text=kept_source,
                         source_format="CSV", objective_is_absent=(len(objective) == 0))
    return problem


# ============================================================================
# REFERENCE PDHG SOLVER (real algorithm, real measured timing per stage)
# ============================================================================
```

**Also add one stdlib import** — `io` is already imported at the top of
the file (line 37: `import io`), but `csv` is not. Find:
```python
import io
import base64
```
and change it to:
```python
import io
import csv
import base64
```

---

## Patch 3 — file uploader: accept .csv, detect + show format

FIND (around line 1840):
```python
        uploaded=st.file_uploader("Upload File (.lp / .mps)", type=["lp","mps","txt"], label_visibility="collapsed")
        if uploaded is not None:
            content=uploaded.read().decode("utf-8",errors="ignore")
            if uploaded.name.lower().endswith('.mps'):
                st.session_state['_uploaded_mps']=content; st.session_state['_uploaded_lp']=None
            else:
                lp_text=content; st.session_state['_uploaded_lp']=content; st.session_state['_uploaded_mps']=None
```

REPLACE WITH:
```python
        uploaded=st.file_uploader("Upload File (.lp / .mps / .csv)", type=["lp","mps","csv","txt"], label_visibility="collapsed")
        if uploaded is not None:
            content=uploaded.read().decode("utf-8",errors="ignore")
            name_lower = uploaded.name.lower()
            if name_lower.endswith('.mps'):
                st.session_state['_uploaded_mps']=content; st.session_state['_uploaded_lp']=None; st.session_state['_uploaded_csv']=None
                st.session_state['_uploaded_format']='MPS'
            elif name_lower.endswith('.csv'):
                st.session_state['_uploaded_csv']=content; st.session_state['_uploaded_mps']=None; st.session_state['_uploaded_lp']=None
                st.session_state['_uploaded_format']='CSV'
            elif name_lower.endswith('.lp') or name_lower.endswith('.txt'):
                lp_text=content; st.session_state['_uploaded_lp']=content; st.session_state['_uploaded_mps']=None; st.session_state['_uploaded_csv']=None
                st.session_state['_uploaded_format']='LP'
            else:
                st.error(f"Unsupported format: '{uploaded.name}'. Supported formats: MPS, LP, CSV.")
                st.session_state['_uploaded_format']=None
            if st.session_state.get('_uploaded_format'):
                st.markdown(
                    f"<div class='cc-muted' style='margin-top:-4px'>Detected format: "
                    f"<b>{st.session_state['_uploaded_format']}</b> &nbsp;·&nbsp; {_html.escape(uploaded.name)}</div>",
                    unsafe_allow_html=True,
                )
```

---

## Patch 4 — solve button: dispatch by detected format, friendlier errors

FIND (around line 1861):
```python
    if solve_clicked:
        st.session_state.lp_text=lp_text
        uploaded_mps = st.session_state.get('_uploaded_mps')
        try:
            if uploaded_mps:
                # Large real-world MPS files (tens of MB, tens of
                # thousands of rows/cols) can take a few seconds even with
                # a fast parser -- show real progress instead of a
                # silent, seemingly-frozen page.
                with st.spinner(f"Parsing MPS file ({len(uploaded_mps)/1e6:.1f} MB)…"):
                    problem = parse_mps_text(uploaded_mps)
            else:
                problem = parse_lp_text(lp_text)
        except Exception as e:
            st.error(f"Could not parse the LP problem: {e}"); return
        st.session_state.problem=problem; st.session_state.gpu_result=None; st.session_state.cpu_result=None
```

REPLACE WITH:
```python
    if solve_clicked:
        st.session_state.lp_text=lp_text
        uploaded_mps = st.session_state.get('_uploaded_mps')
        uploaded_csv = st.session_state.get('_uploaded_csv')
        input_format = st.session_state.get('_uploaded_format') or 'LP'
        try:
            if uploaded_mps:
                # Large real-world MPS files (tens of MB, tens of
                # thousands of rows/cols) can take a few seconds even with
                # a fast parser -- show real progress instead of a
                # silent, seemingly-frozen page.
                with st.spinner(f"Parsing MPS file ({len(uploaded_mps)/1e6:.1f} MB)…"):
                    problem = parse_mps_text(uploaded_mps)
                problem.source_format = 'MPS'
            elif uploaded_csv:
                with st.spinner(f"Parsing CSV file ({len(uploaded_csv)/1e6:.1f} MB)…"):
                    problem = parse_csv_text(uploaded_csv)
            else:
                problem = parse_lp_text(lp_text)
                problem.source_format = 'LP'
        except ValueError as e:
            st.error(f"Unable to parse {input_format} input file.\n\n{e}"); return
        except Exception as e:
            st.error(f"Conversion failed:\n{e}"); return
        if getattr(problem, "objective_is_absent", False):
            st.warning("CSV contained no 'objective' rows — the objective vector is all-zero (nothing was fabricated).")
        st.session_state['_input_format']=problem.source_format
        st.session_state.problem=problem; st.session_state.gpu_result=None; st.session_state.cpu_result=None
```

---

## Patch 5 — run_real_solver_pipeline: also drive CSV through the real
converter binary path (csv_to_txt.py), same way MPS already does

FIND (around line 144-207):
```python
def run_real_solver_pipeline(problem: LPProblem, tol: float, max_iter: int,
                              mps_text: Optional[str] = None,
                              console_cb=None):
```
... (docstring unchanged) ...
```python
    if not real_solver_available():
        return None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gpu_input_file = tmp / "solver_input.txt"
            meta_file = gpu_input_file.with_suffix(".meta.json")
            output_file = tmp / "solver_output.json"

            objective_constant = 0.0
            was_maximize = (problem.sense == "max")

            if mps_text is not None and _CONVERTER_PATH.exists():
                mps_file = tmp / "input.mps"
                mps_file.write_text(mps_text, encoding="utf-8")
                if console_cb:
                    console_cb("Step 1/3 -- canonicalizing MPS via mps_to_txt.py (highspy)...")
                conv = subprocess.run(
                    [sys.executable, str(_CONVERTER_PATH), str(mps_file), "--out", str(gpu_input_file)],
                    capture_output=True, text=True, errors="replace", timeout=120,
                )
                if conv.returncode != 0 or not gpu_input_file.exists():
                    return None
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    objective_constant = float(meta.get("objective_constant", 0.0))
                    was_maximize = bool(meta.get("was_maximize", False))
            elif _user_gpu_format is not None and hasattr(_user_gpu_format, "export_to_gpu_format"):
```

REPLACE the signature and the `if mps_text is not None ...` branch with:
```python
def run_real_solver_pipeline(problem: LPProblem, tol: float, max_iter: int,
                              mps_text: Optional[str] = None,
                              csv_text: Optional[str] = None,
                              console_cb=None):
```
... (docstring unchanged; optionally mention csv_text mirrors mps_text) ...
```python
    if not real_solver_available():
        return None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gpu_input_file = tmp / "solver_input.txt"
            meta_file = gpu_input_file.with_suffix(".meta.json")
            output_file = tmp / "solver_output.json"

            objective_constant = 0.0
            was_maximize = (problem.sense == "max")
            _csv_converter = _APP_DIR / "csv_to_txt.py"

            if mps_text is not None and _CONVERTER_PATH.exists():
                mps_file = tmp / "input.mps"
                mps_file.write_text(mps_text, encoding="utf-8")
                if console_cb:
                    console_cb("Step 1/3 -- canonicalizing MPS via mps_to_txt.py (highspy)...")
                conv = subprocess.run(
                    [sys.executable, str(_CONVERTER_PATH), str(mps_file), "--out", str(gpu_input_file)],
                    capture_output=True, text=True, errors="replace", timeout=120,
                )
                if conv.returncode != 0 or not gpu_input_file.exists():
                    return None
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    objective_constant = float(meta.get("objective_constant", 0.0))
                    was_maximize = bool(meta.get("was_maximize", False))
            elif csv_text is not None and _csv_converter.exists():
                csv_file = tmp / "input.csv"
                csv_file.write_text(csv_text, encoding="utf-8")
                if console_cb:
                    console_cb("Step 1/3 -- canonicalizing CSV via csv_to_txt.py...")
                conv = subprocess.run(
                    [sys.executable, str(_csv_converter), str(csv_file), "--out", str(gpu_input_file)],
                    capture_output=True, text=True, errors="replace", timeout=120,
                )
                if conv.returncode != 0 or not gpu_input_file.exists():
                    return None
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    objective_constant = float(meta.get("objective_constant", 0.0))
                    was_maximize = bool(meta.get("was_maximize", False))
            elif _user_gpu_format is not None and hasattr(_user_gpu_format, "export_to_gpu_format"):
```

(Everything after this line is unchanged.)

Then update the ONE call site (Patch 6) to pass `csv_text=`.

---

## Patch 6 — pass csv_text at the call site in page_gpu_exec()

FIND (around line 2307):
```python
        _pipeline = run_real_solver_pipeline(
            problem, tol=cfg["tol"], max_iter=cfg["max_iter"],
            mps_text=st.session_state.get('_uploaded_mps'),
        )
```

REPLACE WITH:
```python
        _pipeline = run_real_solver_pipeline(
            problem, tol=cfg["tol"], max_iter=cfg["max_iter"],
            mps_text=st.session_state.get('_uploaded_mps'),
            csv_text=st.session_state.get('_uploaded_csv'),
        )
```

---

## Patch 7 — show "Input Format" in the Problem Summary panels

FIND (around line 2012, in render_input_data):
```python
def render_input_data(problem: LPProblem):
    left=[
        ("Objective", f"{problem.sense.upper()} — {_objective_display(problem)}"),
```

REPLACE WITH:
```python
def render_input_data(problem: LPProblem):
    left=[
        ("Input Format", getattr(problem, "source_format", "LP")),
        ("Objective", f"{problem.sense.upper()} — {_objective_display(problem)}"),
```

FIND (around line 1884, in render_preparation_summary):
```python
def render_preparation_summary(problem: LPProblem):
    """Show Problem Summary and Solver Configuration as synchronized row-by-row animations."""
    obj_str = " + ".join(f"{c:g}{v}" for v, c in zip(problem.var_names, problem.c) if c != 0) or "0"
    left=[
        ("Objective", f"{problem.sense.upper()}: {obj_str}"),
```

REPLACE WITH:
```python
def render_preparation_summary(problem: LPProblem):
    """Show Problem Summary and Solver Configuration as synchronized row-by-row animations."""
    obj_str = " + ".join(f"{c:g}{v}" for v, c in zip(problem.var_names, problem.c) if c != 0) or "0"
    left=[
        ("Input Format", getattr(problem, "source_format", "LP")),
        ("Objective", f"{problem.sense.upper()}: {obj_str}"),
```

---

## Patch 8 — session-state defaults (optional but tidy)

FIND (around line 1308, init_state):
```python
def init_state():
    defaults = {
        "page": "home",
        "lp_text": "",
        "problem": None,
```

REPLACE WITH:
```python
def init_state():
    defaults = {
        "page": "home",
        "lp_text": "",
        "problem": None,
        "_uploaded_mps": None,
        "_uploaded_lp": None,
        "_uploaded_csv": None,
        "_uploaded_format": None,
```

---

## Summary of behavior after these patches

- `.mps` → `parse_mps_text()` (unchanged) → `source_format="MPS"`.
- `.lp` / `.txt` (or pasted text) → `parse_lp_text()` (unchanged) → `source_format="LP"`.
- `.csv` → new `parse_csv_text()` → `source_format="CSV"`, `sense="min"`.
- All three produce the same `LPProblem(sense, var_names, c, A, b, ...)`
  shape, so every downstream page (`page_input`, `page_gpu_prep`,
  `page_gpu_exec`, `page_cpu_bench`, `page_analytics`, `page_results`) and
  every solver path (`reference_pdhg`, `reference_cpu_solve`,
  `try_real_gpu_solve`, `try_real_cpu_solve`, `run_real_solver_pipeline`)
  runs completely unchanged and format-independent — exactly matching
  the target "one canonical model → one downstream pipeline" architecture.
- If `mps_to_txt.py` / `csv_to_txt.py` / `gpu_solver` binaries are present
  beside the frontend file, `run_real_solver_pipeline()` now drives BOTH
  MPS and CSV through their respective real converter scripts and the
  compiled GPU solver; LP (and CSV when `csv_to_txt.py` isn't present)
  still goes through the existing direct `gpu_format.export_to_gpu_format`
  branch, unchanged.
