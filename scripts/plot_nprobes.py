#!/usr/bin/env python3
"""
Plot OpenMP recall vs query latency for nprobe sweep runs.

Usage:
    python3 scripts/plot_nprobes.py [logs_dir]

If logs_dir is omitted, the latest bench-nprobes-* directory under
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


RECALL_RE = re.compile(r"aggregate:\s+mean_recall@(\d+)=([0-9]+(?:\.[0-9]+)?)")
MEAN_MS_RE = re.compile(r"aggregate:\s+.*\bmean_ms=([0-9]+(?:\.[0-9]+)?)")
LOG_RE = re.compile(r"^nprobe-(\d+)-openmp-t(\d+)\.log$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recall vs query mean latency for OpenMP nprobe sweeps."
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        help=(
            "Path to bench-nprobes-* log directory. "
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
        [p for p in logs_root.glob("bench-nprobes-*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not bench_dirs:
        raise FileNotFoundError(
            f"No bench-nprobes-* directories found under {logs_root}"
        )
    return bench_dirs[-1]


def collect_records(logs_dir: Path) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []

    for log_file in sorted(logs_dir.rglob("nprobe-*-openmp-t*.log")):
        name_match = LOG_RE.match(log_file.name)
        if not name_match:
            continue

        nprobe = int(name_match.group(1))
        threads = int(name_match.group(2))
        dataset = (
            log_file.parent.name
            if log_file.parent != logs_dir
            else "unknown-dataset"
        )
        text = log_file.read_text(encoding="utf-8", errors="replace")

        recall_match = RECALL_RE.search(text)
        mean_ms_match = MEAN_MS_RE.search(text)
        if recall_match is None or mean_ms_match is None:
            missing = []
            if recall_match is None:
                missing.append("mean_recall")
            if mean_ms_match is None:
                missing.append("mean_ms")
            print(f"warning: missing {', '.join(missing)} in {log_file.name}")
            continue

        records.append(
            {
                "dataset": dataset,
                "nprobe": nprobe,
                "threads": threads,
                "top_k": int(recall_match.group(1)),
                "mean_recall": float(recall_match.group(2)),
                "query_mean_ms": float(mean_ms_match.group(1)),
            }
        )

    return records


def plot(records: list[dict[str, float | int]], output_path: Path) -> None:
    df = pd.DataFrame(records).sort_values(["dataset", "threads", "nprobe"])
    if df.empty:
        raise ValueError("no records to plot")

    sns.set_theme(style="ticks", context="talk", font_scale=1.05)
    fig, ax_recall = plt.subplots(figsize=(10, 7))
    ax_latency = ax_recall.twinx()

    grouped = (
        df.groupby("nprobe", as_index=False)
        .agg(
            mean_recall=("mean_recall", "mean"),
            query_mean_ms=("query_mean_ms", "mean"),
        )
        .sort_values("nprobe")
    )

    x_values = grouped["nprobe"].astype(int)
    recall_values = grouped["mean_recall"].astype(float)
    latency_values = grouped["query_mean_ms"].astype(float)

    recall_line = ax_recall.plot(
        x_values,
        recall_values,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="Mean Recall",
    )[0]
    latency_line = ax_latency.plot(
        x_values,
        latency_values,
        marker="s",
        linewidth=2.2,
        color="#d62728",
        label="Query Mean Latency (ms)",
    )[0]

    recall_k_values = sorted(df["top_k"].unique())
    recall_label = (
        f"Mean Recall@{int(recall_k_values[0])}"
        if len(recall_k_values) == 1
        else "Mean Recall"
    )
    dataset_values = sorted(df["dataset"].unique())
    thread_values = sorted(int(t) for t in df["threads"].unique())
    if len(dataset_values) == 1 and len(thread_values) == 1:
        title_suffix = f"{dataset_values[0]}, t={thread_values[0]}"
    else:
        title_suffix = "averaged over matched logs"

    recall_min = float(recall_values.min())
    recall_max = float(recall_values.max())
    recall_pad = max(0.015, (recall_max - recall_min) * 0.15)
    recall_low = max(0.0, recall_min - recall_pad)
    recall_high = min(1.02, recall_max + recall_pad)
    if recall_low >= recall_high:
        recall_low = max(0.0, recall_min - 0.02)
        recall_high = min(1.02, recall_max + 0.03)

    latency_min = float(latency_values.min())
    latency_max = float(latency_values.max())
    latency_pad = max(0.08, (latency_max - latency_min) * 0.18)
    latency_low = max(0.0, latency_min - latency_pad)
    latency_high = latency_max + latency_pad
    if latency_low >= latency_high:
        latency_low = max(0.0, latency_min - 0.05)
        latency_high = latency_max + 0.1

    ax_recall.set_title(
        f"Nprobe Sweep: Recall and Query Latency ({title_suffix})", pad=18
    )
    ax_recall.set_xlabel("nprobe")
    ax_recall.set_ylabel(recall_label, color=recall_line.get_color())
    ax_recall.tick_params(axis="y", labelcolor=recall_line.get_color())
    ax_recall.set_ylim(recall_low, recall_high)

    ax_latency.set_ylabel("Query Mean Latency (ms)", color=latency_line.get_color())
    ax_latency.tick_params(axis="y", labelcolor=latency_line.get_color())
    ax_latency.set_ylim(latency_low, latency_high)

    ax_recall.set_xticks(x_values.tolist())
    ax_recall.legend(handles=[recall_line, latency_line], loc="best")
    # Keep only subtle major y-grid lines to reduce visual clutter.
    ax_recall.minorticks_off()
    ax_recall.grid(axis="y", which="major", linewidth=0.7, alpha=0.2)
    ax_recall.grid(axis="x", visible=False)
    ax_latency.grid(False)
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

    csv_path = logs_dir / "nprobes_openmp.csv"
    pd.DataFrame(records).sort_values(["dataset", "threads", "nprobe"]).to_csv(
        csv_path, index=False
    )
    print(f"Wrote data: {csv_path}")

    output_path = logs_dir / "nprobes_recall_vs_query_latency.png"
    plot(records, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
