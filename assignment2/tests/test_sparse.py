"""
Correctness tests for Part B (sparse.c).

For several sizes and densities (including 0% and 100%), the dense and CSR
methods dump A, B and C; the test checks that
  1. both methods generated exactly the same A and B (matched logical inputs),
  2. C equals A @ B (NumPy reference) with |x - y| <= atol + rtol * |y|,
  3. the convert method finds the same number of nonzeros as the CSR generator.

Usage (from the repository root):
    python assignment2/tests/test_sparse.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "c" / ("sparse.exe" if sys.platform == "win32" else "sparse")
RTOL, ATOL = 1e-9, 1e-12


def run(method, n, density, dump=None):
    cmd = [str(EXE), method, str(n), str(density)] + (["--dump", str(dump)] if dump else [])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return {k: v for k, v in re.findall(r"(\w+)=(\S+)", out)}


def load(folder, name, n):
    return np.fromfile(folder / name, dtype="<f8").reshape(n, n)


def main():
    subprocess.run(["gcc", "-O2", "-o", str(EXE), str(ROOT / "c" / "sparse.c")], check=True)
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        dense_dir, csr_dir = Path(tmp) / "dense", Path(tmp) / "csr"
        dense_dir.mkdir()
        csr_dir.mkdir()
        for n in (1, 7, 64, 101, 200):
            for density in (0, 0.5, 5, 50, 100):
                run("dense", n, density, dense_dir)
                info = run("csr", n, density, csr_dir)
                conv = run("convert", n, density)
                a, b = load(dense_dir, "A.bin", n), load(dense_dir, "B.bin", n)
                same_inputs = (np.array_equal(a, load(csr_dir, "A.bin", n))
                               and np.array_equal(b, load(csr_dir, "B.bin", n)))
                ref = a @ b
                ok_dense = np.allclose(load(dense_dir, "C.bin", n), ref, rtol=RTOL, atol=ATOL)
                ok_csr = np.allclose(load(csr_dir, "C.bin", n), ref, rtol=RTOL, atol=ATOL)
                ok_conv = conv["nnz_a"] == info["nnz_a"] and conv["nnz_b"] == info["nnz_b"]
                ok = same_inputs and ok_dense and ok_csr and ok_conv
                print(f"  {'PASS' if ok else 'FAIL'}  n={n:<4} density={density:<4}% "
                      f"same_inputs={same_inputs} dense={ok_dense} csr={ok_csr} "
                      f"convert={ok_conv} nnz_c={info['nnz_c']}")
                all_ok = all_ok and ok
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()