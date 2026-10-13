from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = RESULTS / "figures"
STYLE = {"C": ("#2a78d6", "o"), "Java": ("#eb6834", "s"), "Python": ("#1baf7a", "^")}


def summarise(df):
    ok = df[df["status"] == "ok"]
    measured = ok[ok["phase"] == "measure"].groupby(["language", "n"])["time_s"]
    s = pd.DataFrame({
        "median_s": measured.median(),
        "iqr_pct": 100 * (measured.quantile(0.75) - measured.quantile(0.25)) / measured.median(),
        "peak_memory_mb": ok.groupby(["language", "n"])["peak_memory_mb"].first(),
    })
    # Time per inner-loop iteration: constant if time grows exactly as n^3
    s["ns_per_op"] = 1e9 * s["median_s"] / s.index.get_level_values("n").astype(float) ** 3
    # First warm-up run divided by the median measured run (1 = no warm-up effect)
    first_run = ok[(ok["phase"] == "warmup") & (ok["rep"] == 1)].set_index(["language", "n"])
    s["warmup_ratio"] = first_run["time_s"] / s["median_s"]
    return s.reset_index()


def growth_exponent(s, min_n=100):
    """Slope of log(time) vs log(n). The n^3 model predicts 3."""
    d = s[s["n"] >= min_n]
    return {lang: np.polyfit(np.log(g["n"]), np.log(g["median_s"]), 1)[0]
            for lang, g in d.groupby("language")}


def plot(s, column, ylabel, title, filename, reference=None, log_y=True):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for lang, (color, marker) in STYLE.items():
        d = s[s["language"] == lang]
        ax.plot(d["n"], d[column], color=color, marker=marker, linewidth=2, label=lang)
    if reference is not None:
        ax.plot(*reference[:2], "--", color="gray", linewidth=1.2, label=reference[2])
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set(xlabel="Matrix size n", ylabel=ylabel)
    ax.set_title(title, loc="left")
    ax.grid(True, color="#e4e3df")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)


def main():
    csv_path = sorted((RESULTS / "raw").glob("benchmark_*.csv"))[-1]  # newest
    print(f"Analysing {csv_path.name}\n")
    s = summarise(pd.read_csv(csv_path))

    FIGURES.mkdir(parents=True, exist_ok=True)
    s.to_csv(RESULTS / "summary.csv", index=False, float_format="%.6g")

    ns = np.array([s["n"].min(), s["n"].max()], dtype=float)
    c100 = s[(s["language"] == "C") & (s["n"] == 100)]["median_s"].iloc[0]
    plot(s, "median_s", "Median time (s)", "Multiplication time vs n (log-log)",
         "time_vs_n.png", reference=(ns, c100 * (ns / 100) ** 3, "n³ reference"))
    plot(s, "ns_per_op", "Time / n³ (ns)", "Cost per inner-loop iteration (flat = n³)",
         "time_per_op.png")
    plot(s, "peak_memory_mb", "Peak process memory (MB)", "Peak memory vs n",
         "memory_vs_n.png", reference=(ns, 24 * ns**2 / 1e6, "data: A, B, C (3·8·n² bytes)"))
    plot(s, "warmup_ratio", "First warm-up run / median", "Warm-up effect (1 = none)",
         "warmup.png", reference=(ns, [1, 1], "no effect"), log_y=False)

    for column in ["median_s", "iqr_pct", "peak_memory_mb", "warmup_ratio"]:
        print(f"{column}:\n{s.pivot(index='n', columns='language', values=column).round(4)}\n")
    for lang, k in growth_exponent(s).items():
        print(f"Growth exponent {lang}: {k:.2f}  (n^3 model: 3.00)")


if __name__ == "__main__":
    main()