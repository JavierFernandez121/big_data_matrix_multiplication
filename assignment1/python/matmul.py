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


def save_matrix(m, path):
    """Write a list of rows as little-endian float64 values (row-major)."""
    values = array("d", (v for row in m for v in row))
    if sys.byteorder != "little":
        values.byteswap()
    with open(path, "wb") as f:
        values.tofile(f)


def multiply(a, b, c, n):
    """C = A x B using the basic i-j-k triple loop. C must be pre-allocated."""
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            c[i][j] = s


def parse_args(argv):
    """Split argv into positional arguments and the --warmup / --reps options."""
    positional, warmup, reps = [], 0, 1
    i = 1
    while i < len(argv):
        if argv[i] == "--warmup":
            warmup = int(argv[i + 1])
            i += 2
        elif argv[i] == "--reps":
            reps = int(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    return positional, warmup, reps


def main():
    positional, warmup, reps = parse_args(sys.argv)
    if not 1 <= len(positional) <= 3:
        print("Usage: python matmul.py <n> [data_dir] [output_file] [--warmup W] [--reps R]")
        sys.exit(1)
    n = int(positional[0])
    data_dir = Path(positional[1]) if len(positional) >= 2 else DATA_DIR
    output_file = Path(positional[2]) if len(positional) == 3 else None

    # Not timed: reading inputs and allocating the result
    a = load_matrix(data_dir / f"A_{n}.bin", n)
    b = load_matrix(data_dir / f"B_{n}.bin", n)
    c = [[0.0] * n for _ in range(n)]

    for phase, count in (("warmup", warmup), ("measure", reps)):
        for rep in range(1, count + 1):
            # Timed: only the multiplication kernel
            start = time.perf_counter_ns()
            multiply(a, b, c, n)
            elapsed_ns = time.perf_counter_ns() - start

            checksum = sum(sum(row) for row in c)
            print(f"n={n} phase={phase} rep={rep} time_s={elapsed_ns / 1e9:.9f} "
                  f"checksum={checksum:.10e}", flush=True)

    if output_file is not None:
        save_matrix(c, output_file)


if __name__ == "__main__":
    main()