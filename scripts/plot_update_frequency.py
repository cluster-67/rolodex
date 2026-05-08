#!/usr/bin/env python3
"""
Plot OpenMP centroid update-frequency benchmark results.

Usage:
    python3 scripts/plot_update_frequency.py [logs_dir]

If logs_dir is omitted, the latest bench-update-frequency-* directory under
data/logs is used.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BUILD_TIME_RE = re.compile(r"cluster_build_time_ms=([0-9]+(?:\.[0-9]+)?)")
RECALL_RE = re.compile(r"aggregate:\s+mean_recall@(\d+)=([0-9]+(?:\.[0-9]+)?)")
LOG_RE = re.compile(r"^update-frequency-(\d+)-openmp-t(\d+)\.log$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cluster build time and recall vs OpenMP centroid update frequency."
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        help=(
            "Path to bench-update-frequency-* log directory. "
            "Defaults to latest under data/logs."
        ),
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
        [p for p in logs_root.glob("bench-update-frequency-*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not bench_dirs:
        raise FileNotFoundError(
            f"No bench-update-frequency-* directories found under {logs_root}"
        )
    return bench_dirs[-1]


def collect_records(logs_dir: Path) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []

    for log_file in sorted(logs_dir.glob("update-frequency-*-openmp-t*.log")):
        name_match = LOG_RE.match(log_file.name)
        if not name_match:
            continue

        update_frequency = int(name_match.group(1))
        threads = int(name_match.group(2))
        text = log_file.read_text(encoding="utf-8", errors="replace")

        build_match = BUILD_TIME_RE.search(text)
        recall_match = RECALL_RE.search(text)
        if build_match is None or recall_match is None:
            missing = []
            if build_match is None:
                missing.append("cluster_build_time_ms")
            if recall_match is None:
                missing.append("mean_recall")
            print(f"warning: missing {', '.join(missing)} in {log_file.name}")
            continue

        records.append(
            {
                "update_frequency": update_frequency,
                "threads": threads,
                "top_k": int(recall_match.group(1)),
                "build_seconds": float(build_match.group(1)) / 1000.0,
                "mean_recall": float(recall_match.group(2)),
            }
        )

    return records


def plot(records: list[dict[str, float | int]], output_path: Path) -> None:
    df = pd.DataFrame(records).sort_values(["threads", "update_frequency"])
    if df.empty:
        raise ValueError("no records to plot")

    recall_k = int(df["top_k"].iloc[0])
    threads = sorted(df["threads"].unique())
    thread_label = ", ".join(f"t={int(t)}" for t in threads)

    sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    sns.lineplot(
        data=df,
        x="update_frequency",
        y="build_seconds",
        hue="threads",
        palette="deep",
        marker="o",
        linewidth=2.5,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Cluster Build Time")
    axes[0].set_xlabel("Centroid Update Frequency")
    axes[0].set_ylabel("Seconds")

    sns.lineplot(
        data=df,
        x="update_frequency",
        y="mean_recall",
        hue="threads",
        palette="deep",
        marker="o",
        linewidth=2.5,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title(f"Mean Recall@{recall_k}")
    axes[1].set_xlabel("Centroid Update Frequency")
    axes[1].set_ylabel(f"Mean Recall@{recall_k}")
    axes[1].set_ylim(bottom=0.0, top=min(1.05, max(1.0, df["mean_recall"].max() * 1.1)))

    for ax in axes:
        ax.set_xticks(sorted(df["update_frequency"].unique()))

    fig.suptitle(f"OpenMP Centroid Update-Frequency Sweep ({thread_label})", y=1.03)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {output_path}")


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

    csv_path = logs_dir / "update_frequency_openmp.csv"
    pd.DataFrame(records).sort_values(["threads", "update_frequency"]).to_csv(
        csv_path, index=False
    )
    print(f"Wrote data: {csv_path}")

    output_path = logs_dir / "update_frequency_openmp.png"
    plot(records, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
