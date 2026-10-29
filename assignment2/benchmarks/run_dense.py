"""
Benchmark driver for Assignment 2, Part A (dense methods in C).

For every method (and block size, for the blocked method) and every n, one
process runs the multiplication --warmup times (excluded) and --reps times
(measured). The driver records every run and the peak memory of the process.

Budget: a configuration stops growing n when one multiplication is predicted
(from the previous size, assuming n^3 growth) or observed to exceed --budget
seconds. Skipped sizes are recorded as skipped_budget.

Outputs in assignment2/results/raw/:
    dense_<timestamp>.csv        one row per run
    dense_env_<timestamp>.txt    environment, flags and compiler vectorisation report

Usage (from the repository root):
    python assignment2/benchmarks/run_dense.py
    python assignment2/benchmarks/run_dense.py --methods blocked --blocks 16 32 64 128 256 --sizes 2000
"""

import argparse
import csv
import datetime as dt
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = ROOT / "results" / "raw"
SRC = ROOT / "c" / "dense.c"
EXE = ROOT / "c" / ("dense.exe" if sys.platform == "win32" else "dense")
C_FLAGS = ["-O2"]

LINE_RE = re.compile(r"phase=(\w+) rep=(\d+) time_s=([\d.eE+-]+) checksum=(\S+)")
FIELDS = ["method", "block", "n", "phase", "rep", "time_s", "checksum", "peak_memory_mb", "status"]


def build():
    subprocess.run(["gcc", *C_FLAGS, "-o", str(EXE), str(SRC)], check=True)
    # Record which loops the compiler vectorised (SIMD), to interpret the results
    report = subprocess.run(["gcc", *C_FLAGS, "-fopt-info-vec-optimized", "-c", str(SRC),
                             "-o", str(EXE) + ".o"], capture_output=True, text=True).stderr
    Path(str(EXE) + ".o").unlink(missing_ok=True)
    return report.strip() or "no loops vectorised"


def ensure_inputs(sizes):
    missing = [n for n in sizes if not (DATA / f"B_{n}.bin").exists()]
    if missing:
        print(f"Generating input matrices for n = {missing}...")
        subprocess.run([sys.executable, str(DATA / "generate_matrices.py"),
                        *map(str, missing)], check=True)


def tool_version(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip().splitlines()[0]


def cpu_name():
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def peak_memory_windows(proc):
    """Peak working set (bytes) of a finished process, via GetProcessMemoryInfo."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                   [(name, ctypes.c_size_t) for name in (
                       "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                       "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                       "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ok = get_info(int(proc._handle), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize if ok else 0


def peak_memory_linux(pid):
    """Peak resident set (VmHWM, bytes) of a running process."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return 0


def run_config(method, block, n, warmup, reps, timeout_s):
    cmd = [str(EXE), method, str(n), str(DATA), "--warmup", str(warmup),
           "--reps", str(reps), "--block", str(block)]
    with tempfile.TemporaryFile(mode="w+") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL, text=True)
        status, peak, start = "ok", 0, time.monotonic()
        while proc.poll() is None:
            if sys.platform.startswith("linux"):
                try:
                    if psutil.Process(proc.pid).cmdline()[1:] == cmd[1:]:
                        peak = max(peak, peak_memory_linux(proc.pid))
                except (psutil.Error, OSError):
                    pass
            if time.monotonic() - start > timeout_s:
                proc.kill()
                proc.wait()
                status = "timeout"
                break
            time.sleep(0.005)
        if sys.platform == "win32":
            peak = max(peak, peak_memory_windows(proc))
        if status == "ok" and proc.returncode != 0:
            status = "error"
        out.seek(0)
        runs = [m.groups() for m in LINE_RE.finditer(out.read())]
    return runs, (round(peak / 1e6, 2) if peak else ""), status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["ijk", "ikj", "blocked"])
    parser.add_argument("--blocks", type=int, nargs="+", default=[64],
                        help="block sizes for the blocked method")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--budget", type=float, default=120.0)
    args = parser.parse_args()
    args.sizes.sort()

    vec_report = build()
    ensure_inputs(args.sizes)
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, env_path = RAW / f"dense_{stamp}.csv", RAW / f"dense_env_{stamp}.txt"
    env_path.write_text("\n".join([
        f"date: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"os: {platform.platform()}",
        f"cpu: {cpu_name()}",
        f"cpu_cores_physical: {psutil.cpu_count(logical=False)}",
        f"cpu_cores_logical: {psutil.cpu_count(logical=True)}",
        f"ram_gb: {psutil.virtual_memory().total / 1e9:.1f}",
        f"gcc: {tool_version(['gcc', '--version'])}",
        f"c_flags: {' '.join(C_FLAGS)}",
        f"methods: {args.methods}", f"blocks: {args.blocks}", f"sizes: {args.sizes}",
        f"warmup_runs: {args.warmup}", f"measured_runs: {args.reps}",
        f"budget_per_multiplication_s: {args.budget}",
        "timed_region: zeroing C + multiplication",
        "vectorisation_report (gcc -fopt-info-vec-optimized):", vec_report,
    ]) + "\n", encoding="utf-8")

    configs = [(m, b) for m in args.methods for b in (args.blocks if m == "blocked" else [0])]
    last, stopped = {}, set()
    timeout_s = args.budget * (args.warmup + args.reps) * 1.5 + 30

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for n in args.sizes:
            print(f"\nn = {n}")
            for method, block in configs:
                key = (method, block)
                label = f"{method}(bs={block})" if method == "blocked" else method
                row = {"method": method, "block": block, "n": n}
                if key in last:
                    prev_n, prev_t = last[key]
                    if prev_t * (n / prev_n) ** 3 > args.budget:
                        stopped.add(key)
                if key in stopped:
                    writer.writerow({**row, "status": "skipped_budget"})
                    print(f"  {label:<16} skipped (over budget)")
                    continue

                runs, peak_mb, status = run_config(method, max(block, 1), n,
                                                   args.warmup, args.reps, timeout_s)
                if status != "ok":
                    stopped.add(key)
                    writer.writerow({**row, "peak_memory_mb": peak_mb, "status": status})
                    print(f"  {label:<16} {status}")
                    continue
                for phase, rep, t, checksum in runs:
                    writer.writerow({**row, "phase": phase, "rep": rep, "time_s": t,
                                     "checksum": checksum, "peak_memory_mb": peak_mb,
                                     "status": "ok"})
                f.flush()
                times = sorted(float(t) for p, _, t, _ in runs if p == "measure")
                med = times[len(times) // 2]
                last[key] = (n, med)
                print(f"  {label:<16} median {med:10.4f} s   peak memory {peak_mb} MB")

    print(f"\nRaw results: {csv_path}\nEnvironment: {env_path}")


if __name__ == "__main__":
    main()