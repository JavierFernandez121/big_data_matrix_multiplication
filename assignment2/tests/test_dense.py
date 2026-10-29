"""
Correctness tests for the three dense methods (ijk, ikj, blocked) in C.

Each case writes A and B to a temporary folder, runs every method, reads the
full result C and compares it with the expected matrix using
|x - y| <= atol + rtol * |y|. NumPy is used only as the reference.

The blocked method is also tested with sizes that are not multiples of the
block size, and with blocks larger than the matrix, so the edge blocks are
checked.

Usage (from the repository root):
    python assignment2/tests/test_dense.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "c" / ("dense.exe" if sys.platform == "win32" else "dense")
RTOL, ATOL = 1e-9, 1e-12


def make_cases():
    rng = np.random.default_rng(2026)
    a2, b2 = np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])
    cases = [("known_2x2", a2, b2, np.array([[19.0, 22.0], [43.0, 50.0]])),
             ("one_by_one", np.array([[3.0]]), np.array([[-2.5]]), np.array([[-7.5]]))]
    a = rng.random((50, 50))
    cases.append(("identity", a, np.eye(50), a))
    cases.append(("zero", a, np.zeros((50, 50)), np.zeros((50, 50))))
    for n in (64, 101, 130):  # 101 and 130 are not multiples of the block size
        a, b = rng.random((n, n)), rng.random((n, n))
        cases.append((f"random_{n}", a, b, a @ b))
    return cases


def main():
    subprocess.run(["gcc", "-O2", "-o", str(EXE), str(ROOT / "c" / "dense.c")], check=True)
    configs = [("ijk", 64), ("ikj", 64), ("blocked", 64), ("blocked", 7), ("blocked", 500)]
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, a, b, expected in make_cases():
            n = a.shape[0]
            a.astype("<f8").tofile(tmp / f"A_{n}.bin")
            b.astype("<f8").tofile(tmp / f"B_{n}.bin")
            for method, bs in configs:
                out = tmp / "C.bin"
                out.unlink(missing_ok=True)
                cmd = [str(EXE), method, str(n), str(tmp), str(out), "--block", str(bs)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0 or not out.exists():
                    print(f"  FAIL  {name:<11} {method}(bs={bs}) error: {result.stderr.strip()}")
                    all_ok = False
                    continue
                c = np.fromfile(out, dtype="<f8").reshape(n, n)
                ok = np.allclose(c, expected, rtol=RTOL, atol=ATOL)
                label = f"{method}(bs={bs})" if method == "blocked" else method
                print(f"  {'PASS' if ok else 'FAIL'}  {name:<11} {label:<16} "
                      f"max_abs_error={np.max(np.abs(c - expected)):.2e}")
                all_ok = all_ok and ok
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()