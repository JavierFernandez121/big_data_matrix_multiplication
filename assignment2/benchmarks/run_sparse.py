"""
Benchmark driver for Assignment 2, Part B (sparse CSR versus dense, in C).

Runs every combination of method x density x n. For each one, a process runs
--warmup excluded runs and --reps measured runs and reports time, nonzero
counts of A, B and C, and the size of the data structures; the driver also
records the peak memory of the process.

Budgets, declared before running:
  time    a (method, density) series stops when one run is predicted (n^3 model
          from the previous size) or observed to exceed --budget seconds
  memory  dense and convert are not run when their dense arrays would exceed
          --mem-budget-gb (status skipped_memory)

Outputs in assignment2/results/raw/:
    sparse_<tag>_<timestamp>.csv     one row per run
    sparse_env_<tag>_<timestamp>.txt environment and settings

Usage (from the repository root):
    # Experiment 1: fixed n, several densities
    python assignment2/benchmarks/run_sparse.py --tag density --sizes 2000 --densities 100 50 10 5 1 0.5 0.1 0.01
    # Experiment 2: fixed density, growing n
    python assignment2/benchmarks/run_sparse.py --tag scaling --sizes 1000 2000 4000 8000 16000 --densities 0.1
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
RAW = ROOT / "results" / "raw"
SRC = ROOT / "c" / "sparse.c"
EXE = ROOT / "c" / ("sparse.exe" if sys.platform == "win32" else "sparse")
C_FLAGS = ["-O2"]

LINE_RE = re.compile(r"phase=(\w+) rep=(\d+) time_s=(\S+) checksum=(\S+) "
                     r"nnz_a=(\d+) nnz_b=(\d+) nnz_c=(\d+) struct_mb=(\S+)")
FIELDS = ["method", "n", "density_pct", "phase", "rep", "time_s", "checksum", "nnz_a",
          "nnz_b", "nnz_c", "struct_mb", "peak_memory_mb", "status"]


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
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return 0


def run_config(method, n, density, warmup, reps, timeout_s):
    cmd = [str(EXE), method, str(n), str(density), "--warmup", str(warmup), "--reps", str(reps)]
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
    parser.add_argument("--tag", default="run")
    parser.add_argument("--methods", nargs="+", default=["dense", "csr", "convert"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000])
    parser.add_argument("--densities", type=float, nargs="+",
                        default=[100, 50, 10, 5, 1, 0.5, 0.1, 0.01])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--mem-budget-gb", type=float, default=4.0)
    args = parser.parse_args()
    args.sizes.sort()

    subprocess.run(["gcc", *C_FLAGS, "-o", str(EXE), str(SRC)], check=True)
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RAW / f"sparse_{args.tag}_{stamp}.csv"
    gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout
    (RAW / f"sparse_env_{args.tag}_{stamp}.txt").write_text("\n".join([
        f"date: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"os: {platform.platform()}", f"cpu: {cpu_name()}",
        f"cpu_cores_physical: {psutil.cpu_count(logical=False)}",
        f"cpu_cores_logical: {psutil.cpu_count(logical=True)}",
        f"ram_gb: {psutil.virtual_memory().total / 1e9:.1f}",
        f"gcc: {gcc.splitlines()[0]}", f"c_flags: {' '.join(C_FLAGS)}",
        f"methods: {args.methods}", f"sizes: {args.sizes}", f"densities_pct: {args.densities}",
        f"warmup_runs: {args.warmup}", f"measured_runs: {args.reps}",
        f"time_budget_per_run_s: {args.budget}", f"memory_budget_gb: {args.mem_budget_gb}",
        "inputs: generated inside sparse.c, splitmix64, seed 42, uniform positions, values in [0,1)",
    ]) + "\n", encoding="utf-8")

    last, stopped = {}, set()
    timeout_s = args.budget * (args.warmup + args.reps) * 1.5 + 60
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for n in args.sizes:
            for density in args.densities:
                print(f"\nn = {n}, density = {density}%")
                for method in args.methods:
                    key = (method, density)
                    row = {"method": method, "n": n, "density_pct": density}
                    dense_gb = (3 if method == "dense" else 2) * 8 * n * n / 1e9
                    if method != "csr" and dense_gb > args.mem_budget_gb:
                        writer.writerow({**row, "status": "skipped_memory"})
                        print(f"  {method:<8} skipped (dense arrays {dense_gb:.1f} GB > budget)")
                        continue
                    if key in last:
                        prev_n, prev_t = last[key]
                        if prev_t * (n / prev_n) ** 3 > args.budget:
                            stopped.add(key)
                    if key in stopped:
                        writer.writerow({**row, "status": "skipped_budget"})
                        print(f"  {method:<8} skipped (over time budget)")
                        continue

                    runs, peak_mb, status = run_config(method, n, density, args.warmup,
                                                       args.reps, timeout_s)
                    if status != "ok":
                        stopped.add(key)
                        writer.writerow({**row, "peak_memory_mb": peak_mb, "status": status})
                        print(f"  {method:<8} {status}")
                        continue
                    for phase, rep, t, chk, na, nb, nc, smb in runs:
                        writer.writerow({**row, "phase": phase, "rep": rep, "time_s": t,
                                         "checksum": chk, "nnz_a": na, "nnz_b": nb,
                                         "nnz_c": nc, "struct_mb": smb,
                                         "peak_memory_mb": peak_mb, "status": "ok"})
                    f.flush()
                    times = sorted(float(r[2]) for r in runs if r[0] == "measure")
                    med = times[len(times) // 2]
                    last[key] = (n, med)
                    print(f"  {method:<8} median {med:10.6f} s   nnz_c {runs[-1][6]:>10}   "
                          f"peak memory {peak_mb} MB")
    print(f"\nRaw results: {csv_path}")


if __name__ == "__main__":
    main()