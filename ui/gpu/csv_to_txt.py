"""
CSV -> GPU-format canonicalizer, mirroring mps_to_txt.py's convert_file()
so run_real_solver_pipeline() in the Streamlit frontend can drive a CSV
upload through a real converter subprocess exactly the way it already
does for .mps (see PATCH_INSTRUCTIONS.md, patch #5/#6).

Input schema (company data): type,row,col,value,rhs,sense
    - "constraint" rows define A/b (rhs + sense read from the first row
      seen for that "row" index; every line for that row must belong to
      the same "row").
    - "objective" rows (optional) define c directly as MINIMIZE c^T x
      coefficients -- the same convention gpu_format.py uses. If none
      are present, c is left all-zero (never fabricated) and a warning
      is printed, matching parse_csv_text()'s behavior in the frontend.

Reuses convert_to_gpu_format.py's canonicalize_csv() so there is exactly
one CSV-parsing implementation for the file-based conversion path; the
frontend's own parse_csv_text() (patch #2) is the in-memory equivalent
used when no real converter/solver binaries are installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_to_gpu_format import canonicalize_csv
from gpu_format import export_to_gpu_format
from mps_to_txt import write_meta


def convert_file(input_file, output_file) -> dict:
    print(f"\nConverting: {input_file}")
    model = canonicalize_csv(input_file)
    # canonicalize_csv() has no bound-shift / maximize-negation step (the
    # CSV schema is x >= 0, minimize-only by construction), so these are
    # always the identity values -- present anyway so write_meta() and
    # run_real_solver_pipeline()'s .meta.json reader work unchanged for
    # every input format.
    model.setdefault("objective_constant", 0.0)
    model.setdefault("was_maximize", False)
    model.setdefault("var_capacity", [None] * model["n_col"])

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
    p.add_argument("--out", help="output .txt file (default: same name, .txt extension)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = Path(args.out) if args.out else input_path.with_suffix(".txt")
    convert_file(input_path, output_path)


if __name__ == "__main__":
    main()
