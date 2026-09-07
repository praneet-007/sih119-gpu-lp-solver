from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from scipy.optimize import linprog
from gpu_format import read_gpu_format
def read_objective_constant(input_file: str) -> float:
    meta_path = Path(input_file).with_suffix(".meta.json")
    if not meta_path.exists():
        return 0.0
    with open(meta_path) as f:
        meta = json.load(f)
    return float(meta.get("objective_constant", 0.0))
def read_was_maximize(input_file: str) -> bool:
    meta_path = Path(input_file).with_suffix(".meta.json")
    if not meta_path.exists():
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return bool(meta.get("was_maximize", False))
def solve_on_cpu(c, A_ub, b_ub, A_eq, b_eq):
    print("Starting CPU solver...")
    start_time = time.perf_counter()
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs"
    )
    end_time = time.perf_counter()
    solve_time = end_time - start_time
    return result, solve_time
def extract_dual_values(result):
    """
    Pull constraint shadow prices (dual values) out of a HiGHS linprog
    result, in the SAME row order the gpu_format file uses: b_ub rows
    first, then b_eq rows (see gpu_format.py's module docstring). The
    'highs' method returns these for free via result.ineqlin / .eqlin,
    but that only exists on success -- callers should guard accordingly.
    """
    ub = getattr(result, "ineqlin", None)
    eq = getattr(result, "eqlin", None)
    ub_marginals = (
        list(map(float, ub.marginals)) if ub is not None and ub.marginals is not None else []
    )
    eq_marginals = (
        list(map(float, eq.marginals)) if eq is not None and eq.marginals is not None else []
    )
    return ub_marginals + eq_marginals


def save_result(result, solve_time, objective_constant, was_maximize, output_file):
    if result.success:
        # FIX: for a maximize-sense original problem, canonicalize_mps()
        # negated c before writing the GPU format, so result.fun here is
        # -(true_objective - objective_constant). Un-negating the WHOLE
        # (solved + constant) term recovers the true original objective;
        # negating only one of the two terms would still be wrong.
        raw_total = float(result.fun) + objective_constant
        objective_value = -raw_total if was_maximize else raw_total
        data = {
            "success": True,
            "status": int(result.status),
            "message": result.message,
            "objective_value": objective_value,
            "objective_constant": objective_constant,
            "was_maximize": was_maximize,
            "solve_time_seconds": float(solve_time),
            "solution_vector": result.x.tolist(),
            "dual_values": extract_dual_values(result)
        }
    else:
        data = {
            "success": False,
            "status": int(result.status),
            "message": result.message,
            "objective_value": None,
            "objective_constant": objective_constant,
            "was_maximize": was_maximize,
            "solve_time_seconds": float(solve_time),
            "solution_vector": None
        }
    with open(output_file, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )
def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--input",
        type=str,
        default="matrix_input.txt",
        help="input matrix file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="cpu_result.json",
        help="output JSON result file"
    )
    args = parser.parse_args()
    print()
    print("Reading input...")
    print("----------------")
    c, A_ub, b_ub, A_eq, b_eq = read_gpu_format(args.input)
    n_col = len(c)
    n_ub = A_ub.shape[0] if A_ub is not None else 0
    n_eq = A_eq.shape[0] if A_eq is not None else 0
    objective_constant = read_objective_constant(args.input)
    was_maximize = read_was_maximize(args.input)
    print(f"Input file   : {args.input}")
    print(f"Variables    : {n_col}")
    print(f"<= rows      : {n_ub}")
    print(f"=  rows      : {n_eq}")
    if objective_constant:
        print(f"Obj constant : {objective_constant} (from companion .meta.json)")
    if was_maximize:
        print("Sense        : maximize (original MPS; solved internally as minimize)")
    result, solve_time = solve_on_cpu(
        c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
    )
    print()
    print("CPU RESULT")
    print("----------")
    if result.success:
        raw_total = result.fun + objective_constant
        reported_objective = -raw_total if was_maximize else raw_total
        print("Status       : SUCCESS")
        print(
            f"Objective    : {reported_objective:.10f}"
        )
        print(
            f"Solve time   : {solve_time:.6f} seconds"
        )
    else:
        print("Status       : FAILED")
        print("Message      :", result.message)
        print(
            f"Solve time   : {solve_time:.6f} seconds"
        )
    save_result(
        result,
        solve_time,
        objective_constant,
        was_maximize,
        args.out
    )
    print()
    print(
        f"Result saved : {args.out}"
    )
    print()
if __name__ == "__main__":
    main()