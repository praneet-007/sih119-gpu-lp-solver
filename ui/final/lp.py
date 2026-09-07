"""
Shared HiGHS-backed canonicalizer used by both the MPS and LP input paths.

highspy's Highs.readModel() auto-detects the file format from its
extension/contents and already understands both classic MPS and CPLEX-style
LP files, so a single code path handles the "read a model with HiGHS, then
canonicalize it into the project's A_ub/b_ub/A_eq/b_eq/c form" logic for
both formats. This module is exactly that logic, extracted unchanged from
mps_to_txt.py's canonicalize_mps() so it can be reused by lp_to_txt.py /
convert_to_gpu_format.py without duplicating it.
"""

from __future__ import annotations

from pathlib import Path

import highspy
import numpy as np
from scipy import sparse


def _sparse_from_highspy(a_matrix, n_row: int, n_col: int) -> sparse.csc_matrix:
    """Wrap HiGHS's own column-wise sparse arrays as a scipy CSC matrix -- O(nnz), no dense copy."""
    return sparse.csc_matrix(
        (a_matrix.value_, a_matrix.index_, a_matrix.start_), shape=(n_row, n_col)
    )


def canonicalize_via_highspy(path: str) -> dict:
    """
    Read an MPS or LP file with HiGHS and canonicalize it into:

        minimize     c^T x
        subject to   A_ub x <= b_ub
                     A_eq x  = b_eq
                     x >= 0

    identical in structure and semantics to canonicalize_mps() in
    mps_to_txt.py (this function IS that logic -- mps_to_txt.py now calls
    it). Works for both .mps and .lp inputs because highspy's readModel()
    dispatches on the file itself, not on a format flag we pass in.
    """
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
    # A maximize-sense model is negated into HiGHS's/scipy's canonical
    # minimize form here; was_maximize is written to the companion
    # .meta.json (see write_meta) so consumers (cpu_solve.py, the GPU
    # solver, the frontend) can correct the sign back:
    #   true_objective = -(solved_value + objective_constant) if was_maximize
    #                     else (solved_value + objective_constant)
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
