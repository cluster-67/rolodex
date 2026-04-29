#!/usr/bin/env python3
"""
Plot benchmark metrics from bench.sh output logs.

Creates two PNG files in a benchmark run directory:
1) cluster_build_time_ms.png
2) query_latency_mean_ms.png
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd
import seaborn as sns


FLOAT_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
BUILD_TIME_RE = re.compile(rf"cluster_build_time_ms={FLOAT_RE}")
MEAN_MS_RE = re.compile(rf"aggregate:\s+.*\bmean_ms={FLOAT_RE}")

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
SERIES_PALETTE = {
    series: color
    for series, color in zip(SERIES_ORDER, sns.color_palette("tab10", n_colors=len(SERIES_ORDER)))
}


def choose_time_unit(max_ms: float) -> tuple[float, str, str]:
    if max_ms < 1_000:
        return 1.0, "Milliseconds (ms)", "ms"
    if max_ms < 120_000:
        return 1_000.0, "Seconds (s)", "s"
    return 60_000.0, "Minutes (min)", "min"


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
    df = pd.DataFrame(records)
    datasets = sorted(df["dataset"].unique())
    series_in_data = [s for s in SERIES_ORDER if any(df["series"] == s)]

    ncols = 2
    nrows = max(1, math.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        df_ds = df[df["dataset"] == dataset]
        scale, y_label, unit = choose_time_unit(float(df_ds["milliseconds"].max()))
        df_ds = df_ds.copy()
        df_ds["value"] = df_ds["milliseconds"] / scale

        sns.barplot(
            data=df_ds,
            x="series",
            y="value",
            hue="series",
            order=series_in_data,
            hue_order=series_in_data,
            palette=SERIES_PALETTE,
            estimator="mean",
            errorbar=None,
            dodge=False,
            ax=ax,
        )

        group_bands = [
            ("OpenMP", "#4C72B0"),
            ("MPI", "#55A868"),
        ]
        band_regions: list[tuple[float, float, str]] = []
        for group_prefix, band_color in group_bands:
            group_positions = [
                pos for pos, series_name in enumerate(series_in_data) if series_name.startswith(group_prefix)
            ]
            if group_positions:
                band_start = min(group_positions) - 0.5
                band_end = max(group_positions) + 0.5
                ax.axvspan(
                    band_start,
                    band_end,
                    color=band_color,
                    alpha=0.10,
                    zorder=0,
                )
                band_regions.append((band_start, band_end, group_prefix))

        for patch in ax.patches:
            height = patch.get_height()
            if not math.isfinite(height):
                continue
            x_center = patch.get_x() + (patch.get_width() / 2.0)
            label = f"{height:.0f}" if unit == "ms" else f"{height:.1f}"
            ax.annotate(
                label,
                (x_center, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        ax.set_title(str(dataset))
        ax.set_xlabel("Series")
        ax.set_ylabel(y_label)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, _pos: f"{val:.0f}" if unit == "ms" else f"{val:.1f}")
        )
        ymax = float(df_ds["value"].max())
        ax.set_ylim(0, ymax * 1.10 if ymax > 0 else 1.0)
        ax.set_xlim(-0.5, len(series_in_data) - 0.5)
        ax.margins(x=0)
        y_top = ax.get_ylim()[1]
        group_label_y = y_top * 0.97
        for band_start, band_end, group_name in band_regions:
            ax.text(
                (band_start + band_end) / 2.0,
                group_label_y,
                group_name,
                ha="center",
                va="top",
                fontsize=9,
                color="#444444",
                fontweight="semibold",
            )
        ax.tick_params(axis="x", labelrotation=40)
        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("right")
            tick_label.set_rotation_mode("anchor")

    for idx in range(len(datasets), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=18, y=0.98)
    fig.subplots_adjust(top=0.90)
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
