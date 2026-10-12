import sys
import time
from array import array
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / "data"


def load_matrix(path, n):
    """Read n*n little-endian float64 values and return a list of n rows."""
    values = array("d")
    with open(path, "rb") as f:
        values.fromfile(f, n * n)
    if sys.byteorder != "little":
        values.byteswap()
    return [values[i * n:(i + 1) * n].tolist() for i in range(n)]


def multiply(a, b, c, n):
    """C = A x B using the basic i-j-k triple loop. C must be pre-allocated."""
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            c[i][j] = s


def main():
    if len(sys.argv) != 2:
        print("Usage: python matmul.py <n>")
        sys.exit(1)
    n = int(sys.argv[1])

    # Not timed: reading inputs and allocating the result
    a = load_matrix(DATA_DIR / f"A_{n}.bin", n)
    b = load_matrix(DATA_DIR / f"B_{n}.bin", n)
    c = [[0.0] * n for _ in range(n)]

    # Timed: only the multiplication kernel
    start = time.perf_counter_ns()
    multiply(a, b, c, n)
    elapsed_ns = time.perf_counter_ns() - start

    checksum = sum(sum(row) for row in c)
    print(f"n={n} time_s={elapsed_ns / 1e9:.6f} checksum={checksum:.10e}")


if __name__ == "__main__":
    main()