import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
JAVA_DIR = ASSIGNMENT_DIR / "java"
C_DIR = ASSIGNMENT_DIR / "c"
C_EXE = C_DIR / ("matmul.exe" if sys.platform == "win32" else "matmul")


RTOL = 1e-9
ATOL = 1e-12


def build():
    """Compile the Java and C implementations so tests use the latest code."""
    print("Compiling Java and C...")
    subprocess.run(["javac", str(JAVA_DIR / "MatMul.java")], check=True)
    subprocess.run(["gcc", "-O2", "-o", str(C_EXE), str(C_DIR / "matmul.c")],
                   check=True)


def commands(n, data_dir, out_file):
    """Command line for each implementation."""
    args = [str(n), str(data_dir), str(out_file)]
    return {
        "Python": [sys.executable, str(ASSIGNMENT_DIR / "python" / "matmul.py")] + args,
        "Java": ["java", "-cp", str(JAVA_DIR), "MatMul"] + args,
        "C": [str(C_EXE)] + args,
    }


def make_cases():
    rng = np.random.default_rng(2026)
    cases = []

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    cases.append(("known_2x2", a, b, expected))

    cases.append(("one_by_one", np.array([[3.0]]), np.array([[-2.5]]),
                  np.array([[-7.5]])))

    a = rng.random((50, 50))
    cases.append(("identity", a, np.eye(50), a))

    cases.append(("zero", rng.random((50, 50)), np.zeros((50, 50)),
                  np.zeros((50, 50))))

    for n in (64, 101):
        a, b = rng.random((n, n)), rng.random((n, n))
        cases.append((f"random_{n}", a, b, a @ b))

    return cases


def run_case(name, a, b, expected, work_dir):
    n = a.shape[0]
    a.astype("<f8").tofile(work_dir / f"A_{n}.bin")
    b.astype("<f8").tofile(work_dir / f"B_{n}.bin")

    all_ok = True
    for language, cmd in commands(n, work_dir, work_dir / "C_out.bin").items():
        out_file = work_dir / "C_out.bin"
        out_file.unlink(missing_ok=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out_file.exists():
            print(f"  FAIL  {name:<11} {language:<6} program error: {result.stderr.strip()}")
            all_ok = False
            continue

        c = np.fromfile(out_file, dtype="<f8").reshape(n, n)
        ok = np.allclose(c, expected, rtol=RTOL, atol=ATOL)
        max_err = np.max(np.abs(c - expected))
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:<11} {language:<6} max_abs_error={max_err:.2e}")
        all_ok = all_ok and ok
    return all_ok


def main():
    build()
    print(f"\nRunning tests (rtol={RTOL}, atol={ATOL})")
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, a, b, expected in make_cases():
            all_ok = run_case(name, a, b, expected, Path(tmp)) and all_ok

    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()