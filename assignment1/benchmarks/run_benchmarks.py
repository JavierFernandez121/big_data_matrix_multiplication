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

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ASSIGNMENT_DIR / "data"
RESULTS_DIR = ASSIGNMENT_DIR / "results" / "raw"
JAVA_DIR = ASSIGNMENT_DIR / "java"
C_DIR = ASSIGNMENT_DIR / "c"
C_EXE = C_DIR / ("matmul.exe" if sys.platform == "win32" else "matmul")

C_FLAGS = ["-O2"]
JAVA_FLAGS = []  # default JVM settings

DEFAULT_SIZES = [10, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
LANGUAGES = ["C", "Java", "Python"]

LINE_RE = re.compile(r"phase=(\w+) rep=(\d+) time_s=([\d.eE+-]+) checksum=(\S+)")
CSV_FIELDS = ["language", "n", "phase", "rep", "time_s", "checksum",
              "peak_memory_mb", "status"]


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def build():
    print("Compiling Java and C...")
    subprocess.run(["javac", str(JAVA_DIR / "MatMul.java")], check=True)
    subprocess.run(["gcc", *C_FLAGS, "-o", str(C_EXE), str(C_DIR / "matmul.c")],
                   check=True)


def ensure_inputs(sizes):
    """Generate the shared input matrices that are not on disk yet."""
    missing = [n for n in sizes if not (DATA_DIR / f"A_{n}.bin").exists()
               or not (DATA_DIR / f"B_{n}.bin").exists()]
    if missing:
        print(f"Generating input matrices for n = {missing}...")
        subprocess.run([sys.executable, str(DATA_DIR / "generate_matrices.py"),
                        *map(str, missing)], check=True)


def command(language, n, warmup, reps):
    opts = [str(n), "--warmup", str(warmup), "--reps", str(reps)]
    if language == "Python":
        return [sys.executable, str(ASSIGNMENT_DIR / "python" / "matmul.py"), *opts]
    if language == "Java":
        return ["java", *JAVA_FLAGS, "-cp", str(JAVA_DIR), "MatMul", *opts]
    return [str(C_EXE), *opts]


# --------------------------------------------------------------------------
# Environment description (required by the assignment protocol)
# --------------------------------------------------------------------------

def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def tool_version(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        lines = (out.stdout + out.stderr).strip().splitlines()
        lines = [l for l in lines if not l.startswith("Picked up")]
        return " | ".join(lines[:2])
    except FileNotFoundError:
        return "not found"


def describe_environment(args):
    freq = psutil.cpu_freq()
    return "\n".join([
        f"date: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"os: {platform.platform()}",
        f"cpu: {cpu_name()}",
        f"cpu_cores_physical: {psutil.cpu_count(logical=False)}",
        f"cpu_cores_logical: {psutil.cpu_count(logical=True)}",
        f"cpu_max_freq_mhz: {freq.max if freq else 'unknown'}",
        f"ram_gb: {psutil.virtual_memory().total / 1e9:.1f}",
        f"python: {platform.python_version()} ({platform.python_implementation()})",
        f"java: {tool_version(['java', '-version'])}",
        f"javac: {tool_version(['javac', '-version'])}",
        f"gcc: {tool_version(['gcc', '--version'])}",
        f"c_flags: {' '.join(C_FLAGS)}",
        f"jvm_flags: {' '.join(JAVA_FLAGS) or '(default)'}",
        f"numeric_type: float64 (double)",
        f"algorithm: basic triple loop, i-j-k order, scalar accumulator",
        f"timer: monotonic (perf_counter_ns / System.nanoTime / "
        f"QueryPerformanceCounter or clock_gettime), kernel only",
        f"sizes: {args.sizes}",
        f"warmup_runs: {args.warmup}",
        f"measured_runs: {args.reps}",
        f"budget_per_multiplication_s: {args.budget}",
    ]) + "\n"


# --------------------------------------------------------------------------
# Running one (language, n) configuration
# --------------------------------------------------------------------------

def windows_peak_after_exit(proc):
    """Peak working set (bytes) of a finished Windows process.

    GetProcessMemoryInfo still works after the process has exited because
    subprocess keeps its handle open, so even very short runs are exact.
    """
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    try:
        ok = get_info(int(proc._handle), ctypes.byref(counters), counters.cb)
    except Exception:
        return 0
    return counters.PeakWorkingSetSize if ok else 0


def sample_peak(ps_proc, cmd):
    """Current value of the peak-memory counter of a running process (bytes).

    Linux: VmHWM (peak resident set size). Windows: peak working set.
    macOS: current RSS (the maximum over the samples is kept).
    Returns 0 while the child is still a copy of this script (before exec).
    """
    if sys.platform != "win32" and ps_proc.cmdline()[1:] != cmd[1:]:
        return 0
    if sys.platform.startswith("linux"):
        with open(f"/proc/{ps_proc.pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        return 0
    info = ps_proc.memory_info()
    return getattr(info, "peak_wset", info.rss)


def run_config(language, n, warmup, reps, timeout_s):
    """Run one process; return (list of parsed runs, peak memory MB, status)."""
    cmd = command(language, n, warmup, reps)
    with tempfile.TemporaryFile(mode="w+") as out, \
         tempfile.TemporaryFile(mode="w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
        status, peak, start = "ok", 0, time.monotonic()
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.Error:
            ps_proc = None  # already finished

        while proc.poll() is None:
            try:
                peak = max(peak, sample_peak(ps_proc, cmd))
            except (psutil.Error, OSError, ValueError, AttributeError):
                pass  # process just finished
            if time.monotonic() - start > timeout_s:
                proc.kill()
                proc.wait()
                status = "timeout"
                break
            time.sleep(0.002)

        if sys.platform == "win32":
            peak = max(peak, windows_peak_after_exit(proc))
        if status == "ok" and proc.returncode != 0:
            status = "error"
            err.seek(0)
            print(f"    {language} failed: {err.read().strip()}")

        out.seek(0)
        runs = [m.groups() for m in LINE_RE.finditer(out.read())]

    peak_mb = round(peak / 1e6, 2) if peak else ""
    return runs, peak_mb, status


def median(values):
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--budget", type=float, default=120.0,
                        help="max seconds allowed for ONE multiplication")
    args = parser.parse_args()
    args.sizes = sorted(args.sizes)

    build()
    ensure_inputs(args.sizes)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"benchmark_{stamp}.csv"
    env_path = RESULTS_DIR / f"environment_{stamp}.txt"
    env_path.write_text(describe_environment(args), encoding="utf-8")

    runs_per_process = args.warmup + args.reps
    timeout_s = args.budget * runs_per_process * 1.5 + 30  # safety margin
    last = {}         # language -> (n, median time) of the last completed size
    stopped = set()   # languages that exceeded the budget

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for n in args.sizes:
            print(f"\nn = {n}")
            checksums = {}
            for language in args.languages:
                row = {"language": language, "n": n}

                if language in stopped:
                    writer.writerow({**row, "status": "skipped_budget"})
                    print(f"  {language:<7} skipped (over budget at a smaller n)")
                    continue
                if language in last:
                    prev_n, prev_t = last[language]
                    predicted = prev_t * (n / prev_n) ** 3
                    if predicted > args.budget:
                        stopped.add(language)
                        writer.writerow({**row, "status": "skipped_budget"})
                        print(f"  {language:<7} skipped (predicted {predicted:.0f} s "
                              f"> budget {args.budget:.0f} s)")
                        continue

                runs, peak_mb, status = run_config(language, n, args.warmup,
                                                   args.reps, timeout_s)
                if status != "ok":
                    stopped.add(language)
                    writer.writerow({**row, "peak_memory_mb": peak_mb, "status": status})
                    print(f"  {language:<7} {status}")
                    continue

                for phase, rep, time_s, checksum in runs:
                    writer.writerow({**row, "phase": phase, "rep": rep,
                                     "time_s": time_s, "checksum": checksum,
                                     "peak_memory_mb": peak_mb, "status": "ok"})
                f.flush()

                measured = [float(t) for p, _, t, _ in runs if p == "measure"]
                med = median(measured)
                last[language] = (n, med)
                checksums[language] = float(runs[-1][3])
                print(f"  {language:<7} median {med:12.6f} s   "
                      f"peak memory {peak_mb} MB")
                if med > args.budget:
                    stopped.add(language)

            # Sanity check: all languages must produce the same result
            values = list(checksums.values())
            if values and max(values) - min(values) > 1e-9 * abs(values[0]):
                print(f"  WARNING: checksums differ between languages: {checksums}")

    print(f"\nRaw results:  {csv_path.relative_to(ASSIGNMENT_DIR.parent)}")
    print(f"Environment:  {env_path.relative_to(ASSIGNMENT_DIR.parent)}")


if __name__ == "__main__":
    main()