# Reference copy of the new parse_csv_text() function that gets inserted
# into frontend_cuda_crusaders_v2.py (see PATCH_INSTRUCTIONS.md, patch #4).
# Kept here as a real, importable/testable .py file so it can be unit
# tested standalone before being pasted into the big frontend file.

from __future__ import annotations
import csv as _csv
import io
import numpy as np
from scipy import sparse
try:
    from dataclasses import dataclass, field
except ImportError:
    pass


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


def parse_csv_text(text):
    """
    Parse the type,row,col,value,rhs,sense CSV schema into an LPProblem,
    using the SAME canonical A x <= b, x >= 0 representation parse_lp_text()
    and parse_mps_text() produce (equalities mirrored into a <= pair, >=
    rows sign-flipped) -- so page_home()'s downstream code (run_real_solver_pipeline,
    reference_pdhg, reference_cpu_solve, all the render_* stages) does not
    need to know or care that the source was a CSV.

    CSV objective rows are interpreted directly as MINIMIZE c^T x
    coefficients (matching gpu_format.py's minimize-only convention and
    what canonicalize_via_highspy() itself produces after negating a
    maximize-sense MPS/LP model) -- so the returned problem's sense is
    always "min". If the CSV has no maximize/minimize marker for your
    original problem, negate your coefficients yourself before uploading.
    If the CSV has no 'objective' rows at all, the objective is left as
    all-zero (never fabricated) and the caller should warn the user.
    """
    if not text or not text.strip():
        raise ValueError("CSV file is empty.")

    reader = _csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV file is empty.") from None

    header_norm = [h.strip().lower() for h in header]
    missing = [c for c in CSV_REQUIRED_COLUMNS if c not in header_norm]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
    col_idx = {c: header_norm.index(c) for c in CSV_REQUIRED_COLUMNS}

    constraints = {}   # row -> {"coeffs": {col: val}, "rhs": float, "sense": str}
    objective = {}      # col -> value
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
        sign = 1.0 if sense in ("<=",) else (-1.0 if sense == ">=" else 1.0)
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

    problem = _make_lp_problem(
        sense="min", var_names=var_names, c=c, A=A, b=b_arr,
        row_labels=row_labels, source_text=text if len(text) <= 200_000 else "",
    )
    problem.objective_is_absent = len(objective) == 0
    return problem


def _make_lp_problem(**kwargs):
    """Placeholder used only for standalone testing of this snippet file
    (the real patch calls LPProblem(**kwargs) directly, since LPProblem is
    already defined earlier in frontend_cuda_crusaders_v2.py)."""
    from dataclasses import dataclass, field as _field

    @dataclass
    class _LPProblemStub:
        sense: str
        var_names: list
        c: np.ndarray
        A: object
        b: np.ndarray
        row_labels: list = _field(default_factory=list)
        source_text: str = ""
        source_format: str = "LP"
        objective_is_absent: bool = False

        @property
        def m(self): return self.A.shape[0]
        @property
        def n(self): return self.A.shape[1]
        @property
        def nnz(self): return int(self.A.nnz)

    return _LPProblemStub(**kwargs)


if __name__ == "__main__":
    text = """type,row,col,value,rhs,sense
objective,,0,-3,,
objective,,1,-2,,
objective,,2,-5,,
constraint,0,0,1,10,<=
constraint,0,1,2,10,<=
constraint,0,2,1,10,<=
constraint,1,0,2,8,<=
constraint,1,1,1,8,<=
constraint,2,0,1,6,<=
constraint,2,2,1,6,<=
"""
    p = parse_csv_text(text)
    print("m,n,nnz =", p.m, p.n, p.nnz)
    print("A =\n", p.A.toarray())
    print("b =", p.b)
    print("c =", p.c)
    print("objective_is_absent =", p.objective_is_absent)
