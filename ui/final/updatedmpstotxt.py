"""
MPS -> GPU-format canonicalizer, built on highspy, staying sparse throughout.

The actual HiGHS-based canonicalization now lives in lp_canonicalize.py
(shared with the .lp input path, since highspy reads both formats through
the same readModel() call) -- canonicalize_mps() below is kept as a thin,
backwards-compatible wrapper so existing callers/imports of this module
are unaffected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_format import export_to_gpu_format
from lp_canonicalize import canonicalize_via_highspy


def canonicalize_mps(path: str) -> dict:
    return canonicalize_via_highspy(path)


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