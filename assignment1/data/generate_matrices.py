import random
import sys
from array import array
from pathlib import Path

SEED = 42
DEFAULT_SIZES = [10, 50, 100, 200, 500, 1000]
OUTPUT_DIR = Path(__file__).parent


def generate_matrix(n, rng):
    """Return an n x n matrix as a flat array of float64 in row-major order."""
    return array("d", (rng.random() for _ in range(n * n)))


def save_matrix(values, path):
    if sys.byteorder != "little":
        values = array("d", values)
        values.byteswap()
    with open(path, "wb") as f:
        values.tofile(f)


def main():
    sizes = [int(arg) for arg in sys.argv[1:]] or DEFAULT_SIZES

    for n in sizes:
        # One generator per size
        rng = random.Random(f"{SEED}-{n}")
        a = generate_matrix(n, rng)
        b = generate_matrix(n, rng)
        save_matrix(a, OUTPUT_DIR / f"A_{n}.bin")
        save_matrix(b, OUTPUT_DIR / f"B_{n}.bin")
        print(f"n={n}: A_{n}.bin, B_{n}.bin ({8 * n * n:,} bytes each)")


if __name__ == "__main__":
    main()