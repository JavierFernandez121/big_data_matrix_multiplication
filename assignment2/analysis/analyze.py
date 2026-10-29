"""
Analysis of Assignment 2: summary tables and figures for Part A and Part B.

Reads the newest raw CSV of each experiment in results/raw/ and writes
results/summary_*.csv and results/figures/*.png. Only measured runs are used.

Usage (from the repository root):
    python assignment2/analysis/analyze.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = RESULTS / "figures"
STYLE = {"ijk": ("#2a78d6", "o"), "ikj": ("#eb6834", "s"), "blocked": ("#1baf7a", "^"),
         "dense": ("#2a78d6", "o"), "csr": ("#eb6834", "s"), "csr + convert": ("#1baf7a", "^"),
         "convert": ("#6250d6", "D")}


def newest(pattern, exclude=None):
    files = [f for f in sorted((RESULTS / "raw").glob(pattern)) if not exclude or exclude not in f.name]
    return files[-1]


def summarise(df, keys):
    ok = df[(df["status"] == "ok") & (df["phase"] == "measure")]
    g = ok.groupby(keys)
    s = g.agg(median_s=("time_s", "median"),
              q1=("time_s", lambda x: x.quantile(0.25)),
              q3=("time_s", lambda x: x.quantile(0.75)),
              runs=("time_s", "count"),
              peak_memory_mb=("peak_memory_mb", "first")).reset_index()
    s["iqr_pct"] = 100 * (s["q3"] - s["q1"]) / s["median_s"]
    return s.drop(columns=["q1", "q3"])


def plot(series, xlabel, ylabel, title, filename, reference=None, log_x=True, log_y=True):
    """series: {label: (x, y)}"""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, (x, y) in series.items():
        color, marker = STYLE[label]
        ax.plot(x, y, color=color, marker=marker, linewidth=2, label=label)
    if reference is not None:
        ax.plot(*reference[:2], "--", color="gray", linewidth=1.2, label=reference[2])
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left")
    ax.grid(True, color="#e4e3df")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)


def part_a():
    # The block-size sweep is the file with blocks other than 64; the main run is the other
    sweep_file = [f for f in sorted((RESULTS / "raw").glob("dense_2*.csv"))
                  if set(pd.read_csv(f)["block"].unique()) - {0, 64}]
    main_file = [f for f in sorted((RESULTS / "raw").glob("dense_2*.csv")) if f not in sweep_file][-1]
    raw = pd.read_csv(main_file)
    s = summarise(raw, ["method", "n"])
    s["ns_per_op"] = 1e9 * s["median_s"] / s["n"].astype(float) ** 3
    s.to_csv(RESULTS / "summary_dense.csv", index=False, float_format="%.6g")

    series = {m: (d["n"], d["median_s"]) for m, d in s.groupby("method")}
    series = {m: series[m] for m in ("ijk", "ikj", "blocked")}
    ns = np.array([s["n"].min(), s["n"].max()], dtype=float)
    b100 = s[(s["method"] == "blocked") & (s["n"] == 1000)]["median_s"].iloc[0]
    plot(series, "Matrix size n", "Median time (s)", "Dense methods: time vs n (log-log)",
         "dense_time_vs_n.png", reference=(ns, b100 * (ns / 1000) ** 3, "n³ reference"))
    per_op = {m: (d["n"], d["ns_per_op"]) for m, d in s.groupby("method")}
    plot({m: per_op[m] for m in ("ijk", "ikj", "blocked")},
         "Matrix size n", "Time / n³ (ns)", "Dense methods: cost per multiply-add (flat = n³)",
         "dense_time_per_op.png")

    print("PART A - median time (s):")
    print(s.pivot(index="n", columns="method", values="median_s").round(4), "\n")
    print("PART A - IQR (% of median):")
    print(s.pivot(index="n", columns="method", values="iqr_pct").round(1), "\n")
    print("PART A - not run:")
    print(raw[raw["status"] != "ok"][["method", "n", "status"]].to_string(index=False), "\n")

    if sweep_file:
        sw = summarise(pd.read_csv(sweep_file[-1]), ["block", "n"])
        sw.to_csv(RESULTS / "summary_block_sweep.csv", index=False, float_format="%.6g")
        print("PART A - block size sweep:")
        print(sw[["n", "block", "median_s", "iqr_pct"]].round(3).to_string(index=False), "\n")


def part_b():
    for tag in ("density", "scaling"):
        raw = pd.read_csv(newest(f"sparse_{tag}_*.csv"))
        s = summarise(raw, ["method", "n", "density_pct"])
        first = raw[raw["status"] == "ok"].groupby(["method", "n", "density_pct"]).first()
        s = s.merge(first[["nnz_a", "nnz_b", "nnz_c", "struct_mb"]].reset_index(),
                    on=["method", "n", "density_pct"])
        s["output_density_pct"] = 100 * s["nnz_c"] / s["n"].astype(float) ** 2
        s.to_csv(RESULTS / f"summary_sparse_{tag}.csv", index=False, float_format="%.6g")

        t = s.pivot_table(index=["n", "density_pct"], columns="method", values="median_s")
        t["csr + convert"] = t["csr"] + t.get("convert", np.nan)
        if tag == "density":
            x = t.index.get_level_values("density_pct")
            plot({k: (x, t[k]) for k in ("dense", "csr", "csr + convert")},
                 "Density of A and B (% nonzero)", "Median time (s)",
                 "n = 2000: time vs density (log-log)", "sparse_time_vs_density.png")
        else:
            x = t.index.get_level_values("n")
            plot({k: (x[t[k].notna()], t[k].dropna()) for k in ("dense", "csr", "convert")},
                 "Matrix size n", "Median time (s)",
                 "Density 0.1%: time vs n (log-log)", "sparse_time_vs_n.png")

        csr = s[s["method"] == "csr"].set_index(["n", "density_pct"])
        table = pd.DataFrame({
            "dense_s": t.get("dense"), "csr_s": t["csr"], "convert_s": t.get("convert"),
            "speedup_dense_over_csr": t.get("dense") / t["csr"],
            "nnz_a": csr["nnz_a"], "nnz_c": csr["nnz_c"],
            "output_density_pct": csr["output_density_pct"],
            "csr_struct_mb": csr["struct_mb"]})
        print(f"PART B - {tag}:")
        print(table.to_string(float_format=lambda v: f"{v:.4g}"), "\n")
        not_run = raw[raw["status"] != "ok"][["method", "n", "density_pct", "status"]]
        if not not_run.empty:
            print(f"PART B - {tag} not run:\n{not_run.to_string(index=False)}\n")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    part_a()
    part_b()
    print("Written: results/summary_*.csv and results/figures/*.png")


if __name__ == "__main__":
    main()