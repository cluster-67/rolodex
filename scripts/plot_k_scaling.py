#!/usr/bin/env python3
"""
Plot training time vs k for serial, OpenMP, and MPI variants.

Usage:
    python3 scripts/plot_k_scaling.py [logs_dir]

If logs_dir is omitted, the latest bench-k-* directory under data/logs is used.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import pandas as pd


BUILD_TIME_RE = re.compile(r"cluster_build_time_ms=([0-9]+(?:\.[0-9]+)?)")

# Filename pattern: k{K}-{impl}.log
LOG_RE = re.compile(r"^k(\d+)-(serial|openmp-t(\d+)|mpi-N(\d+)-n(\d+))\.log$")

SERIES_ORDER = [
    "Serial",
    "OpenMP t=2",
    "OpenMP t=4",
    "OpenMP t=8",
    "MPI N=1 n=2",
    "MPI N=1 n=4",
    "MPI N=1 n=8",
    "MPI N=2 n=2",
    "MPI N=2 n=4",
    "MPI N=2 n=8",
]

SERIAL_EXTRAPOLATE_AFTER_K = 128

SERIES_STYLES = {
    "Serial": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "OpenMP t=2": {"color": "#ff7f0e", "marker": "s", "linestyle": "-"},
    "OpenMP t=4": {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "OpenMP t=8": {"color": "#d62728", "marker": "D", "linestyle": "-"},
    "MPI N=1 n=2": {"color": "#9467bd", "marker": "P", "linestyle": "-"},
    "MPI N=1 n=4": {"color": "#8c564b", "marker": "X", "linestyle": "-"},
    "MPI N=1 n=8": {"color": "#e377c2", "marker": "*", "linestyle": "-"},
    "MPI N=2 n=2": {"color": "#7f7f7f", "marker": "v", "linestyle": "--"},
    "MPI N=2 n=4": {"color": "#bcbd22", "marker": "<", "linestyle": "--"},
    "MPI N=2 n=8": {"color": "#17becf", "marker": ">", "linestyle": "--"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training time vs k.")
    parser.add_argument(
        "logs_dir",
        nargs="?",
        help="Path to bench-k-* log directory. Defaults to latest under data/logs.",
    )
    return parser.parse_args()


def resolve_logs_dir(arg: str | None) -> Path:
    root = Path(__file__).resolve().parents[1]
    logs_root = root / "data" / "logs"

    if arg:
        candidate = Path(arg).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()

    bench_dirs = sorted(
        [p for p in logs_root.glob("bench-k-*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not bench_dirs:
        raise FileNotFoundError(f"No bench-k-* directories found under {logs_root}")
    return bench_dirs[-1]


def series_label(log_name: str) -> str | None:
    m = LOG_RE.match(log_name)
    if not m:
        return None
    impl = m.group(2)
    if impl == "serial":
        return "Serial"
    if impl.startswith("openmp"):
        return f"OpenMP t={m.group(3)}"
    return f"MPI N={m.group(4)} n={m.group(5)}"


def collect_records(logs_dir: Path) -> list[dict]:
    records = []
    for log_file in sorted(logs_dir.glob("k*.log")):
        m = LOG_RE.match(log_file.name)
        if not m:
            continue
        k = int(m.group(1))
        label = series_label(log_file.name)
        if label is None:
            continue

        text = log_file.read_text(encoding="utf-8", errors="replace")
        bm = BUILD_TIME_RE.search(text)
        if bm is None:
            print(f"warning: no cluster_build_time_ms in {log_file.name}")
            continue

        build_ms = float(bm.group(1))
        records.append({"k": k, "series": label, "seconds": build_ms / 1000.0})

    return records


def add_serial_extrapolation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    serial = df[
        (df["series"] == "Serial") & (df["k"] <= SERIAL_EXTRAPOLATE_AFTER_K)
    ].sort_values("k")
    if len(serial) < 2:
        return df, pd.DataFrame()

    # Keep the solid serial line to the last intentionally measured point.
    df = df[(df["series"] != "Serial") | (df["k"] <= SERIAL_EXTRAPOLATE_AFTER_K)].copy()

    max_k = int(df["k"].max())
    extrapolate_ks = [
        int(k)
        for k in sorted(df["k"].unique())
        if SERIAL_EXTRAPOLATE_AFTER_K <= int(k) <= max_k
    ]
    if len(extrapolate_ks) < 2:
        return df, pd.DataFrame()

    last_two = serial.tail(2)
    k1, k2 = float(last_two.iloc[0]["k"]), float(last_two.iloc[1]["k"])
    t1, t2 = float(last_two.iloc[0]["time"]), float(last_two.iloc[1]["time"])
    if k1 <= 0 or k2 <= k1 or t1 <= 0 or t2 <= 0:
        return df, pd.DataFrame()

    # Power-law trend through the last two measured serial points on log-log axes.
    exponent = (math.log(t2) - math.log(t1)) / (math.log(k2) - math.log(k1))
    scale = t2 / (k2**exponent)

    extrapolated = pd.DataFrame(
        {
            "k": extrapolate_ks,
            "series": "Serial extrapolated",
            "time": [scale * (float(k) ** exponent) for k in extrapolate_ks],
        }
    )
    return df, extrapolated


def save_legend(handles, labels, output_path: Path) -> None:
    if not handles:
        return

    legend_fig = plt.figure(figsize=(8, 4.5))
    legend = legend_fig.legend(
        handles,
        labels,
        loc="center",
        title="Approach",
        ncol=2,
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    legend.get_frame().set_linewidth(0.8)
    legend_fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(legend_fig)
    print(f"Wrote legend: {output_path}")


def plot(records: list[dict], output_path: Path, legend_output_path: Path) -> None:
    df = pd.DataFrame(records)

    max_seconds = df["seconds"].max() if not df.empty else 0
    if max_seconds >= 120:
        df["time"] = df["seconds"] / 60.0
        y_label = "Training Time (minutes)"
    else:
        df["time"] = df["seconds"]
        y_label = "Training Time (seconds)"

    # Keep only series that appear in the data, in display order.
    series_present = [s for s in SERIES_ORDER if s in df["series"].values]
    df["series"] = pd.Categorical(df["series"], categories=series_present, ordered=True)
    df = df.sort_values(["series", "k"])
    df, serial_extrapolated = add_serial_extrapolation(df)

    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(14, 8))

    for series in series_present:
        series_df = df[df["series"] == series]
        if series_df.empty:
            continue
        style = SERIES_STYLES[series]
        ax.plot(
            series_df["k"],
            series_df["time"],
            label=series,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=8,
            linewidth=3.0,
            alpha=0.95,
        )

    if not serial_extrapolated.empty:
        serial_style = SERIES_STYLES["Serial"]
        ax.plot(
            serial_extrapolated["k"],
            serial_extrapolated["time"],
            label=f"Serial projected after k={SERIAL_EXTRAPOLATE_AFTER_K}",
            color=serial_style["color"],
            linestyle=":",
            marker=serial_style["marker"],
            markersize=8,
            linewidth=3.4,
            alpha=0.9,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel(y_label)
    ax.set_title("Training Time vs k  (k-means cluster build)")
    handles, labels = ax.get_legend_handles_labels()
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {output_path}")
    save_legend(handles, labels, legend_output_path)


def main() -> int:
    args = parse_args()
    logs_dir = resolve_logs_dir(args.logs_dir)

    if not logs_dir.exists() or not logs_dir.is_dir():
        print(f"error: not a directory: {logs_dir}", file=sys.stderr)
        return 1

    print(f"Using logs directory: {logs_dir}")
    records = collect_records(logs_dir)

    if not records:
        print("error: no valid log files found", file=sys.stderr)
        return 1

    output_path = logs_dir / "k_scaling_training_time.png"
    legend_output_path = logs_dir / "k_scaling_training_time_legend.png"
    plot(records, output_path, legend_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
