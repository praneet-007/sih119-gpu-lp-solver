"""
MPS -> GPU-format canonicalizer, built on highspy, staying sparse throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import highspy
import numpy as np
from scipy import sparse

from gpu_format import export_to_gpu_format


def _sparse_from_highspy(a_matrix, n_row: int, n_col: int) -> sparse.csc_matrix:
    """Wrap HiGHS's own column-wise sparse arrays as a scipy CSC matrix -- O(nnz), no dense copy."""
    return sparse.csc_matrix(
        (a_matrix.value_, a_matrix.index_, a_matrix.start_), shape=(n_row, n_col)
    )


def canonicalize_mps(path: str) -> dict:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(str(path))
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"highspy failed to read {path}: status={status}")

    lp = h.getLp()
    n_col = lp.num_col_
    n_row = lp.num_row_

    A_orig = _sparse_from_highspy(lp.a_matrix_, n_row, n_col).tocsr()
    c = np.array(lp.col_cost_, dtype=float)
    offset = float(lp.offset_)
    # FIX: canonicalize_mps() negates c (and offset) to convert a
    # maximize problem into HiGHS's/scipy's canonical minimize form.
    # Without recording that here, any downstream solver (CPU or GPU)
    # that reports "objective_value_from_gpu_format_solve" has the
    # WRONG SIGN for every maximize-sense MPS file -- and this whole
    # project is a profit-MAXIMIZATION dashboard, so this would hit
    # essentially every real run. was_maximize is written to the
    # companion .meta.json (see write_meta) so consumers can correct
    # for it: true_objective = -(solved_value + objective_constant)
    # when was_maximize, else (solved_value + objective_constant).
    was_maximize = lp.sense_ == highspy.ObjSense.kMaximize
    if was_maximize:
        c = -c
        offset = -offset

    col_lower = np.array(lp.col_lower_, dtype=float)
    col_upper = np.array(lp.col_upper_, dtype=float)
    row_lower = np.array(lp.row_lower_, dtype=float)
    row_upper = np.array(lp.row_upper_, dtype=float)

    orig_names = list(lp.col_names_) if lp.col_names_ else [f"C{j}" for j in range(n_col)]

    ub_row_idx, ub_sign, ub_rhs = [], [], []
    eq_row_idx, eq_rhs = [], []

    for i in range(n_row):
        lo, hi = row_lower[i], row_upper[i]
        if lo == hi:
            eq_row_idx.append(i)
            eq_rhs.append(hi)
        elif np.isinf(hi):
            ub_row_idx.append(i); ub_sign.append(-1.0); ub_rhs.append(-lo)
        elif np.isinf(lo):
            ub_row_idx.append(i); ub_sign.append(1.0); ub_rhs.append(hi)
        else:
            ub_row_idx.append(i); ub_sign.append(1.0); ub_rhs.append(hi)
            ub_row_idx.append(i); ub_sign.append(-1.0); ub_rhs.append(-lo)

    if ub_row_idx:
        A_ub_can = A_orig[ub_row_idx, :].multiply(np.array(ub_sign)[:, None]).tocsc()
    else:
        A_ub_can = sparse.csc_matrix((0, n_col))
    b_ub_can = np.array(ub_rhs)

    if eq_row_idx:
        A_eq_can = A_orig[eq_row_idx, :].tocsc()
    else:
        A_eq_can = sparse.csc_matrix((0, n_col))
    b_eq_can = np.array(eq_rhs)

    new_names = []
    new_cols = []
    # Capacity of each canonical (post-shift) column, i.e. its own upper
    # bound in the shifted/split variable space -- None means unbounded.
    # This is exactly the rhs later pushed into extra_bound_rows for the
    # bounded case, so the two stay consistent by construction. Frontend
    # dashboards (tank-utilization gauges) use this to turn a raw solved
    # value into a % of capacity without having to re-derive it.
    new_capacity = []
    shift = np.zeros(n_col)
    extra_bound_rows = []

    for j in range(n_col):
        lo, hi = col_lower[j], col_upper[j]
        if np.isinf(lo) and np.isinf(hi):
            new_cols.append((j, 1.0)); new_names.append(orig_names[j] + "_pos"); new_capacity.append(None)
            new_cols.append((j, -1.0)); new_names.append(orig_names[j] + "_neg"); new_capacity.append(None)
        elif np.isinf(lo):
            shift[j] = hi
            new_cols.append((j, -1.0)); new_names.append(orig_names[j]); new_capacity.append(None)
        else:
            shift[j] = lo
            k = len(new_cols)
            new_cols.append((j, 1.0)); new_names.append(orig_names[j])
            if not np.isinf(hi):
                extra_bound_rows.append((k, hi - lo))
                new_capacity.append(hi - lo)
            else:
                new_capacity.append(None)

    new_n = len(new_cols)
    orig_cols = np.array([j for j, _ in new_cols])
    signs = np.array([sign for _, sign in new_cols])

    A_ub_new = A_ub_can[:, orig_cols].multiply(signs).tocsr()
    b_ub_new = b_ub_can - A_ub_can @ shift

    A_eq_new = A_eq_can[:, orig_cols].multiply(signs).tocsr()
    b_eq_new = b_eq_can - A_eq_can @ shift

    if extra_bound_rows:
        k_idx = np.array([k for k, _ in extra_bound_rows])
        rhs_vals = np.array([rhs for _, rhs in extra_bound_rows])
        n_extra = len(extra_bound_rows)
        extra_block = sparse.coo_matrix(
            (np.ones(n_extra), (np.arange(n_extra), k_idx)), shape=(n_extra, new_n)
        ).tocsr()
        A_ub_new = sparse.vstack([A_ub_new, extra_block], format="csr")
        b_ub_new = np.concatenate([b_ub_new, rhs_vals])

    c_new = np.zeros(new_n)
    for k, (j, sign) in enumerate(new_cols):
        c_new[k] = c[j] * sign
    objective_constant = offset + float(c @ shift)

    return {
        "name": lp.model_name_ or Path(path).stem.upper(),
        "A_ub": A_ub_new if A_ub_new.shape[0] > 0 else None,
        "b_ub": b_ub_new if A_ub_new.shape[0] > 0 else None,
        "A_eq": A_eq_new if A_eq_new.shape[0] > 0 else None,
        "b_eq": b_eq_new if A_eq_new.shape[0] > 0 else None,
        "c": c_new,
        "objective_constant": objective_constant,
        "was_maximize": was_maximize,
        "var_names": new_names,
        "var_capacity": new_capacity,
        "n_row": A_ub_new.shape[0] + A_eq_new.shape[0],
        "n_col": new_n,
    }


def write_meta(filename, model: dict) -> None:
    """Write the sidecar metadata (objective constant, sense, variable mapping)."""
    with open(filename, "w") as f:
        json.dump({
            "name": model["name"],
            "objective_constant": model["objective_constant"],
            "was_maximize": model["was_maximize"],
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
    model = canonicalize_mps(input_file)
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
    print(f"Obj constant: {model['objective_constant']:.6f}")
    print("Conversion successful.")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="input .mps file")
    p.add_argument("--out", help="output .txt file")
    p.add_argument("--input-dir", help="directory containing MPS files")
    p.add_argument("--output-dir", help="directory for converted TXT files")
    args = p.parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        output_path = Path(args.out) if args.out else input_path.with_suffix(".txt")
        convert_file(input_path, output_path)
        return

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        output_dir = Path(args.output_dir if args.output_dir else "converted_txt")
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob("*.mps"))
        if not files:
            print("No .mps files found.")
            return

        print(f"Found {len(files)} MPS files.")
        for input_file in files:
            output_file = output_dir / input_file.with_suffix(".txt").name
            try:
                convert_file(input_file, output_file)
            except Exception as e:
                print(f"FAILED: {input_file}")
                print(f"Reason: {e}")
        print("\nBatch conversion completed.")
        return

    p.print_help()


if __name__ == "__main__":
    main()
