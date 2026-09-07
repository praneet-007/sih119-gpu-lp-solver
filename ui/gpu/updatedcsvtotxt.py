"""
CSV -> GPU-format canonicalizer.

Schema (one nonzero / one objective coefficient per row)
---------------------------------------------------------
    type,row,col,value,rhs,sense

    type   "constraint" or "objective"
    row    0-indexed constraint row (constraint rows only; ignored for
           objective rows)
    col    0-indexed variable column (always required)
    value  coefficient (A[row, col] for a constraint row, c[col] for an
           objective row)
    rhs    right-hand side of the constraint (constraint rows only; every
           line for the same row must agree on rhs)
    sense  "<=", ">=" or "=" (constraint rows only; every line for the
           same row must agree on sense)

Example
-------
    type,row,col,value,rhs,sense
    objective,,0,3,,
    objective,,1,2,,
    constraint,0,0,1,10,<=
    constraint,0,1,2,10,<=
    constraint,1,0,1,4,<=

Objective coefficients are interpreted directly as minimize c^T x
coefficients -- the same convention gpu_format.py's minimize-only format
uses, and the same convention canonicalize_via_highspy() produces after
its own maximize -> minimize negation. If your problem is a maximization,
negate the objective coefficients before writing the CSV (or use the
.lp/.mps path, which carries an explicit sense). If a CSV has no
"objective" rows at all, the objective is reported as all-zero rather
than fabricated -- see the "objective_is_absent" flag returned below.

Only the constraint senses/matrix are canonicalized here (into A_ub/b_ub,
A_eq/b_eq); there is no bounds section in this schema, so -- exactly like
gpu_format.py's own format -- x >= 0 is the only variable bound and no
shifting/splitting is needed (unlike the MPS/LP path, which must
canonicalize arbitrary variable bounds away).
"""

from __future__ import annotations

import argparse
import csv as _csv
import io
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from gpu_format import export_to_gpu_format

REQUIRED_COLUMNS = ["type", "row", "col", "value", "rhs", "sense"]
VALID_TYPES = {"constraint", "objective"}
SENSE_ALIASES = {"<=": "<=", "le": "<=", ">=": ">=", "ge": ">=", "=": "=", "==": "=", "eq": "="}


class CsvFormatError(ValueError):
    """Raised for any malformed/invalid CSV input -- always carries a clear, user-facing message."""


def _require_float(raw: str, line_no: int, field: str) -> float:
    raw = (raw or "").strip()
    if raw == "":
        raise CsvFormatError(f"Missing value for '{field}' on CSV row {line_no}.")
    try:
        return float(raw)
    except ValueError:
        raise CsvFormatError(f"Invalid numeric value for '{field}' on CSV row {line_no}: {raw!r}") from None


def _require_index(raw: str, line_no: int, field: str) -> int:
    raw = (raw or "").strip()
    if raw == "":
        raise CsvFormatError(f"Missing value for '{field}' on CSV row {line_no}.")
    try:
        val = int(raw)
    except ValueError:
        raise CsvFormatError(f"Invalid index for '{field}' on CSV row {line_no}: {raw!r} (must be a non-negative integer)") from None
    if val < 0:
        raise CsvFormatError(f"Invalid index for '{field}' on CSV row {line_no}: {val} (indices must be >= 0)")
    return val


def parse_csv_model(text: str, name: str = "CSV_MODEL") -> dict:
    """
    Parse the type,row,col,value,rhs,sense CSV schema into the project's
    canonical dict (same shape canonicalize_via_highspy() returns), ready
    for gpu_format.export_to_gpu_format().

    Raises CsvFormatError with a clear, user-facing message for any
    structural problem (missing columns, bad numbers, missing constraint
    rows, invalid senses, ...). Never silently guesses.
    """
    if not text or not text.strip():
        raise CsvFormatError("CSV file is empty.")

    reader = _csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise CsvFormatError("CSV file is empty.") from None

    header_norm = [h.strip().lower() for h in header]
    missing = [c for c in REQUIRED_COLUMNS if c not in header_norm]
    if missing:
        raise CsvFormatError(f"CSV is missing required column(s): {', '.join(missing)}")
    col_idx = {c: header_norm.index(c) for c in REQUIRED_COLUMNS}

    constraints: dict[int, dict] = {}   # row -> {"coeffs": {col: val}, "rhs": float, "sense": str}
    objective: dict[int, float] = {}    # col -> value
    max_col = -1
    any_data_row = False

    for line_no, fields in enumerate(reader, start=2):  # header is line 1
        if not fields or all((f or "").strip() == "" for f in fields):
            continue  # skip blank lines
        if len(fields) < len(header):
            raise CsvFormatError(f"Malformed CSV row {line_no}: expected {len(header)} fields, got {len(fields)}.")
        any_data_row = True

        row_type = (fields[col_idx["type"]] or "").strip().lower()
        if row_type not in VALID_TYPES:
            raise CsvFormatError(
                f"Invalid 'type' value on CSV row {line_no}: {fields[col_idx['type']]!r} "
                f"(expected 'constraint' or 'objective')."
            )

        col = _require_index(fields[col_idx["col"]], line_no, "col")
        value = _require_float(fields[col_idx["value"]], line_no, "value")
        max_col = max(max_col, col)

        if row_type == "objective":
            objective[col] = objective.get(col, 0.0) + value
            continue

        # constraint row
        row = _require_index(fields[col_idx["row"]], line_no, "row")
        rhs = _require_float(fields[col_idx["rhs"]], line_no, "rhs")
        sense_raw = (fields[col_idx["sense"]] or "").strip().lower()
        sense = SENSE_ALIASES.get(sense_raw)
        if sense is None:
            raise CsvFormatError(
                f"Invalid constraint sense on CSV row {line_no}: {fields[col_idx['sense']]!r} "
                f"(expected one of '<=', '>=', '=')."
            )

        entry = constraints.setdefault(row, {"coeffs": {}, "rhs": rhs, "sense": sense})
        if entry["rhs"] != rhs:
            raise CsvFormatError(
                f"Inconsistent rhs for constraint row {row} on CSV row {line_no}: "
                f"{entry['rhs']!r} vs {rhs!r} (every line for the same row must agree)."
            )
        if entry["sense"] != sense:
            raise CsvFormatError(
                f"Inconsistent sense for constraint row {row} on CSV row {line_no}: "
                f"{entry['sense']!r} vs {sense!r} (every line for the same row must agree)."
            )
        entry["coeffs"][col] = entry["coeffs"].get(col, 0.0) + value

    if not any_data_row:
        raise CsvFormatError("CSV has a header but no data rows.")
    if not constraints:
        raise CsvFormatError("CSV must contain at least one constraint row (type=constraint).")

    row_ids = sorted(constraints.keys())
    expected = list(range(len(row_ids)))
    if row_ids != expected:
        found_missing = sorted(set(expected) - set(row_ids))
        raise CsvFormatError(
            f"CSV is missing constraint row(s) {found_missing} -- 'row' indices must be "
            f"a contiguous 0-based range with no gaps (found rows {row_ids})."
        )

    objective_is_absent = len(objective) == 0
    n_col = max(max_col + 1, (max(objective.keys()) + 1) if objective else 0)

    ub_rows, ub_cols, ub_vals, ub_rhs = [], [], [], []
    eq_rows, eq_cols, eq_vals, eq_rhs = [], [], [], []
    n_eq = 0
    n_ub = 0
    for row in row_ids:
        entry = constraints[row]
        sense = entry["sense"]
        if sense == "=":
            r = n_eq
            for c, v in entry["coeffs"].items():
                eq_rows.append(r); eq_cols.append(c); eq_vals.append(v)
            eq_rhs.append(entry["rhs"])
            n_eq += 1
        else:
            sign = 1.0 if sense == "<=" else -1.0
            r = n_ub
            for c, v in entry["coeffs"].items():
                ub_rows.append(r); ub_cols.append(c); ub_vals.append(v * sign)
            ub_rhs.append(entry["rhs"] * sign)
            n_ub += 1

    A_ub = sparse.csr_matrix((ub_vals, (ub_rows, ub_cols)), shape=(n_ub, n_col)) if n_ub else None
    b_ub = np.array(ub_rhs) if n_ub else None
    A_eq = sparse.csr_matrix((eq_vals, (eq_rows, eq_cols)), shape=(n_eq, n_col)) if n_eq else None
    b_eq = np.array(eq_rhs) if n_eq else None

    c = np.zeros(n_col)
    for col, val in objective.items():
        c[col] = val

    return {
        "name": name,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "c": c,
        "objective_constant": 0.0,
        "was_maximize": False,
        "objective_is_absent": objective_is_absent,
        "var_names": [f"x{j}" for j in range(n_col)],
        "var_capacity": [None] * n_col,
        "n_row": n_ub + n_eq,
        "n_col": n_col,
    }


def canonicalize_csv(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return parse_csv_model(text, name=Path(path).stem.upper())
    except CsvFormatError:
        raise
    except Exception as e:
        raise CsvFormatError(f"Unable to parse CSV file {path}: {e}") from e


def write_meta(filename, model: dict) -> None:
    with open(filename, "w") as f:
        json.dump({
            "name": model["name"],
            "objective_constant": model["objective_constant"],
            "was_maximize": model["was_maximize"],
            "objective_is_absent": model.get("objective_is_absent", False),
            "note": (
                "true_optimal_objective = "
                "-(objective_value_from_gpu_format_solve + objective_constant) "
                "if was_maximize else "
                "(objective_value_from_gpu_format_solve + objective_constant)"
            ),
            "var_names": model["var_names"],
            "var_capacity": model["var_capacity"],
            "n_row": model["n_row"],
            "n_col": model["n_col"],
        }, f, indent=2)


def convert_file(input_file, output_file) -> dict:
    print(f"\nConverting: {input_file}")
    model = canonicalize_csv(input_file)
    if model["objective_is_absent"]:
        print("WARNING: CSV contained no 'objective' rows -- objective vector is all-zero.")
    export_to_gpu_format(
        model["c"], model["A_ub"], model["b_ub"], model["A_eq"], model["b_eq"], output_file
    )
    meta_file = Path(output_file).with_suffix(".meta.json")
    write_meta(meta_file, model)

    nnz = (model["A_ub"].nnz if model["A_ub"] is not None else 0) + \
          (model["A_eq"].nnz if model["A_eq"] is not None else 0)
    size = model["n_row"] * model["n_col"]
    print(f"Output      : {output_file}")
    print(f"Metadata    : {meta_file}")
    print(f"Dimensions  : {model['n_row']} x {model['n_col']}")
    print(f"Nonzeros    : {nnz}")
    print(f"Density     : {nnz / size if size else 0:.8f}")
    print("Conversion successful.")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="input .csv file (type,row,col,value,rhs,sense schema)")
    p.add_argument("--out", help="output .txt file")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    output_path = Path(args.out) if args.out else input_path.with_suffix(".txt")
    try:
        convert_file(input_path, output_path)
    except CsvFormatError as e:
        print(f"Conversion failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
