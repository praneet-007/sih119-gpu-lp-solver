"""
Run this ON THE GPU LAPTOP after build_gpu_solver.sh has produced
./gpu_solver, and after you've installed the Python deps
(numpy, scipy, pandas, highspy):

    pip install numpy scipy pandas highspy

Usage:
    python verify_gpu_vs_cpu.py sample.csv
    python verify_gpu_vs_cpu.py sample.mps

It runs the SAME input through both pipelines:
    input file -> csv_to_txt.py / mps_to_txt.py -> gpu_format.txt
        -> cpu_solve.py            (scipy/HiGHS, ground truth)
        -> ./gpu_solver             (your compiled CUDA binary)

then diffs the two objective values. This is the real integration
check -- it proves the GPU binary, the converter, and the shared
gpu_format.py text format all agree with each other end to end,
which is the one thing that could NOT be checked without a GPU.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU_BINARY = HERE / "gpu_solver"
TOL = 1e-3


def convert(input_file: Path, out_txt: Path):
    ext = input_file.suffix.lower()
    if ext == ".csv":
        script = HERE / "csv_to_txt.py"
    elif ext in (".mps", ".lp"):
        script = HERE / "mps_to_txt.py"
    else:
        raise ValueError(f"Unsupported extension: {ext} (use .csv, .mps, or .lp)")

    result = subprocess.run(
        [sys.executable, str(script), str(input_file), "--out", str(out_txt)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{script.name} failed on {input_file}")


def run_cpu(gpu_txt: Path, out_json: Path):
    result = subprocess.run(
        [sys.executable, str(HERE / "cpu_solve.py"),
         "--input", str(gpu_txt), "--out", str(out_json)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("cpu_solve.py failed")
    with open(out_json) as f:
        return json.load(f)


def run_gpu(gpu_txt: Path, out_json: Path):
    if not GPU_BINARY.exists():
        raise FileNotFoundError(
            f"{GPU_BINARY} not found. Run build_gpu_solver.sh first."
        )
    result = subprocess.run(
        [str(GPU_BINARY), str(gpu_txt), str(out_json)],
        capture_output=True, text=True,
    )
    print("--- gpu_solver stderr (progress log) ---")
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"gpu_solver exited with code {result.returncode}")
    with open(out_json) as f:
        return json.load(f)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input.csv|input.mps|input.lp>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gpu_txt = tmp / "problem.txt"
        cpu_json = tmp / "cpu_result.json"
        gpu_json = tmp / "gpu_result.json"

        print(f"=== 1/3 Converting {input_file.name} -> gpu_format ===")
        convert(input_file, gpu_txt)

        print("\n=== 2/3 Solving on CPU (scipy/HiGHS, ground truth) ===")
        cpu_result = run_cpu(gpu_txt, cpu_json)

        print("\n=== 3/3 Solving on GPU (compiled gpu_solver.cu) ===")
        gpu_result = run_gpu(gpu_txt, gpu_json)

    print("\n=== COMPARISON ===")
    cpu_obj = cpu_result.get("objective_value")
    gpu_obj = gpu_result.get("objective_value")
    gpu_status = gpu_result.get("status")
    print(f"CPU objective : {cpu_obj}")
    print(f"GPU objective : {gpu_obj}  (status={gpu_status})")

    if cpu_obj is None:
        print("CPU solve did not succeed -- cannot compare. See cpu_result above.")
        sys.exit(1)
    if gpu_status not in ("OPTIMAL",):
        print(f"FAIL: GPU did not report OPTIMAL (status={gpu_status}).")
        sys.exit(1)

    diff = abs(cpu_obj - gpu_obj) / (1.0 + abs(cpu_obj))
    print(f"Relative difference: {diff:.6e} (tolerance {TOL:.0e})")
    if diff <= TOL:
        print("PASS: GPU and CPU agree within tolerance.")
    else:
        print("FAIL: GPU and CPU objectives disagree -- do not trust the GPU path yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
