#!/usr/bin/env python3
"""
Plot benchmark metrics from bench.sh output logs.

Creates two PNG files in a benchmark run directory:
1) cluster_build_time_ms.png
2) query_latency_mean_ms.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


BUILD_TIME_RE = re.compile(r"cluster_build_time_ms=([0-9]+(?:\.[0-9]+)?)")
MEAN_MS_RE = re.compile(r"aggregate:\s+.*\bmean_ms=([0-9]+(?:\.[0-9]+)?)")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create benchmark plots from data/logs bench outputs."
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        help=(
            "Path to benchmark run directory (e.g. "
            "data/logs/bench-2026-04-28-20-22-17). "
            "If omitted, latest bench-* under data/logs is used."
        ),
    )
    return parser.parse_args()


def resolve_logs_dir(logs_dir_arg: str | None) -> Path:
    root = Path(__file__).resolve().parents[1]
    logs_root = root / "data" / "logs"

    if logs_dir_arg:
        candidate = Path(logs_dir_arg).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()

    bench_dirs = sorted(
        [p for p in logs_root.glob("bench-*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not bench_dirs:
        raise FileNotFoundError(f"No bench-* directories found under {logs_root}")
    return bench_dirs[-1]


def parse_series_name(log_file: Path) -> str | None:
    name = log_file.name
    if name == "serial.log":
        return "Serial"

    openmp_match = re.fullmatch(r"openmp-t(\d+)\.log", name)
    if openmp_match:
        return f"OpenMP t={openmp_match.group(1)}"

    mpi_match = re.fullmatch(r"mpi-N(\d+)-n(\d+)\.log", name)
    if mpi_match:
        return f"MPI N={mpi_match.group(1)} n={mpi_match.group(2)}"

    return None


def parse_log_metrics(log_file: Path) -> tuple[float | None, float | None]:
    text = log_file.read_text(encoding="utf-8", errors="replace")

    build_match = BUILD_TIME_RE.search(text)
    build_ms = float(build_match.group(1)) if build_match else None

    mean_match = MEAN_MS_RE.search(text)
    mean_ms = float(mean_match.group(1)) if mean_match else None

    return build_ms, mean_ms


def collect_records(logs_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    build_records: list[dict[str, object]] = []
    latency_records: list[dict[str, object]] = []

    dataset_dirs = sorted([p for p in logs_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        for log_file in sorted(dataset_dir.glob("*.log"), key=lambda p: p.name):
            series = parse_series_name(log_file)
            if not series:
                continue

            build_ms, mean_ms = parse_log_metrics(log_file)

            # Missing metrics are skipped per request.
            if build_ms is not None:
                build_records.append(
                    {
                        "dataset": dataset,
                        "series": series,
                        "milliseconds": build_ms,
                    }
                )
            if mean_ms is not None:
                latency_records.append(
                    {
                        "dataset": dataset,
                        "series": series,
                        "milliseconds": mean_ms,
                    }
                )

    return build_records, latency_records


def plot_metric(records: list[dict[str, object]], title: str, output_path: Path) -> None:
    if not records:
        print(f"warning: no data found for plot {output_path.name}")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7))

    # For now we usually have one dataset, but this supports multiple.
    datasets = sorted({str(r["dataset"]) for r in records})
    series_in_data = [s for s in SERIES_ORDER if any(r["series"] == s for r in records)]

    for series in series_in_data:
        xs: list[str] = []
        ys: list[float] = []
        for dataset in datasets:
            value = next(
                (
                    float(r["milliseconds"])
                    for r in records
                    if r["series"] == series and r["dataset"] == dataset
                ),
                None,
            )
            if value is None:
                continue
            xs.append(dataset)
            ys.append(value)

        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=series)

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Milliseconds")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    logs_dir = resolve_logs_dir(args.logs_dir)

    if not logs_dir.exists() or not logs_dir.is_dir():
        print(f"error: log directory not found or not a directory: {logs_dir}", file=sys.stderr)
        return 1

    print(f"Using logs directory: {logs_dir}")

    build_records, latency_records = collect_records(logs_dir)

    build_plot = logs_dir / "cluster_build_time_ms.png"
    latency_plot = logs_dir / "query_latency_mean_ms.png"

    plot_metric(
        build_records,
        title="Cluster Build Time by Dataset",
        output_path=build_plot,
    )
    plot_metric(
        latency_records,
        title="Query Mean Latency by Dataset",
        output_path=latency_plot,
    )

    print(f"Wrote plot: {build_plot}")
    print(f"Wrote plot: {latency_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
