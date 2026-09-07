"""
Shared sparse COO input/output format for the GPU solver, the CPU
baseline (cpu_solve.py), and mps_to_txt.py.

Format (plain text)
--------------------
    Line 1: M N nnz
    Line 2: c            (N values -- objective coefficients)
    Line 3: b            (M values -- b_ub rows first, then b_eq rows)
    Line 4: row_type     (M values -- 0 = "<=" row, 1 = "=" row)
    Remaining nnz lines: row col value   (0-indexed, one nonzero per line)

The problem this file always describes is exactly:

    minimize     c^T x
    subject to   A_ub x <= b_ub
                 A_eq x  = b_eq
                 x >= 0

Every row is either a pure "<=" row or a pure "=" row -- there is no
representation for variable bounds other than x >= 0. Any real-world
variable bounds (from BOUNDS in an MPS file) must be canonicalized away
before reaching this format (see mps_to_txt.py's substitution step) --
this format intentionally does not, and should not, try to express them.

Rows are written b_ub rows first, then b_eq rows, matching row_type
being sorted (all 0s before all 1s) -- this keeps the reader simple and
avoids needing a separate row-index remap.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def export_to_gpu_format(c, A_ub, b_ub, A_eq, b_eq, output_file) -> None:
    """
    Write c, A_ub, b_ub, A_eq, b_eq to the shared sparse COO text format.

    A_ub / A_eq may be None (meaning zero rows of that type) or any
    scipy.sparse matrix -- they are converted to COO internally, so no
    dense copy is ever made regardless of the input format.
    """
    c = np.asarray(c, dtype=float)
    n_col = len(c)

    n_ub = A_ub.shape[0] if A_ub is not None else 0
    n_eq = A_eq.shape[0] if A_eq is not None else 0
    m = n_ub + n_eq

    b_ub = np.asarray(b_ub, dtype=float) if b_ub is not None else np.array([])
    b_eq = np.asarray(b_eq, dtype=float) if b_eq is not None else np.array([])

    if len(b_ub) != n_ub:
        raise ValueError(f"b_ub has {len(b_ub)} values but A_ub has {n_ub} rows.")
    if len(b_eq) != n_eq:
        raise ValueError(f"b_eq has {len(b_eq)} values but A_eq has {n_eq} rows.")

    b = np.concatenate([b_ub, b_eq])
    row_type = np.concatenate([
        np.zeros(n_ub, dtype=int),
        np.ones(n_eq, dtype=int),
    ])

    triplets = []
    if A_ub is not None and n_ub > 0:
        coo = A_ub.tocoo()
        for r, col, v in zip(coo.row, coo.col, coo.data):
            if v != 0.0:
                triplets.append((int(r), int(col), float(v)))
    if A_eq is not None and n_eq > 0:
        coo = A_eq.tocoo()
        for r, col, v in zip(coo.row, coo.col, coo.data):
            if v != 0.0:
                triplets.append((n_ub + int(r), int(col), float(v)))

    with open(output_file, "w") as f:
        f.write(f"{m} {n_col} {len(triplets)}\n")
        f.write(" ".join(repr(float(x)) for x in c) + "\n")
        f.write(" ".join(repr(float(x)) for x in b) + "\n")
        f.write(" ".join(map(str, row_type)) + "\n")
        for r, col, v in triplets:
            f.write(f"{r} {col} {v!r}\n")


def read_gpu_format(input_file: str):
    """
    Read the shared sparse COO text format.

    Returns:
        c (ndarray), A_ub (csr_matrix or None), b_ub (ndarray or None),
        A_eq (csr_matrix or None), b_eq (ndarray or None)
    """
    with open(input_file) as f:
        header = f.readline().split()
        if len(header) != 3:
            raise ValueError(
                f"Malformed header line: expected 'M N nnz', got {header!r}"
            )
        m, n, nnz = map(int, header)

        c_line = f.readline().split()
        if len(c_line) != n:
            raise ValueError(f"c has {len(c_line)} values but header says N={n}")
        c = np.array(list(map(float, c_line)))

        b_line = f.readline().split()
        if len(b_line) != m:
            raise ValueError(f"b has {len(b_line)} values but header says M={m}")
        b = np.array(list(map(float, b_line)))

        rt_line = f.readline().split()
        if len(rt_line) != m:
            raise ValueError(
                f"row_type has {len(rt_line)} values but header says M={m}"
            )
        row_type = np.array(list(map(int, rt_line)))

        rows, cols, vals = [], [], []
        for line_no in range(nnz):
            parts = f.readline().split()
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed nonzero entry on data line {line_no}: {parts!r}"
                )
            r, col, v = parts
            rows.append(int(r))
            cols.append(int(col))
            vals.append(float(v))

    ub_mask = row_type == 0
    eq_mask = row_type == 1
    n_ub = int(ub_mask.sum())
    n_eq = int(eq_mask.sum())

    # Rows are written b_ub-first-then-b_eq (see module docstring), so
    # original row index -> (block, local index) is a simple offset split
    # rather than needing a lookup per row.
    ub_rows, ub_cols, ub_vals = [], [], []
    eq_rows, eq_cols, eq_vals = [], [], []
    for r, col, v in zip(rows, cols, vals):
        if row_type[r] == 0:
            ub_rows.append(r)
            ub_cols.append(col)
            ub_vals.append(v)
        else:
            eq_rows.append(r - n_ub)
            eq_cols.append(col)
            eq_vals.append(v)

    A_ub = (
        sparse.csr_matrix((ub_vals, (ub_rows, ub_cols)), shape=(n_ub, n))
        if n_ub > 0
        else None
    )
    b_ub_out = b[ub_mask] if n_ub > 0 else None

    A_eq = (
        sparse.csr_matrix((eq_vals, (eq_rows, eq_cols)), shape=(n_eq, n))
        if n_eq > 0
        else None
    )
    b_eq_out = b[eq_mask] if n_eq > 0 else None

    return c, A_ub, b_ub_out, A_eq, b_eq_out
