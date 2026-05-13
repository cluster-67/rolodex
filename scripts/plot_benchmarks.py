#!/usr/bin/env python3
"""
Plot benchmark metrics from bench.sh output logs.

Creates PNG files in a benchmark run directory:
1) cluster_build_time_ms.png
2) query_latency_mean_ms.png
3) cluster_build_stacked.png
4) query_stacked.png
5) series_legend.png
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd
import seaborn as sns


FLOAT_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
BUILD_TIME_RE = re.compile(rf"cluster_build_time_ms={FLOAT_RE}")
MEAN_MS_RE = re.compile(rf"aggregate:\s+.*\bmean_ms={FLOAT_RE}")

# Stage timing keys (tweak these lists to add/remove stages)
CLUSTER_BUILD_STAGES = [
    "cluster_build_membership_ms",
    "cluster_build_centroid_update_ms",
    "cluster_build_mpi_membership_comm_ms",
    "cluster_build_mpi_centroid_comm_ms",
]

CLUSTER_BUILD_STAGE_LABELS = [
    "Membership\nAssignment",
    "Centroid\nUpdate",
    "MPI Membership\nComm",
    "MPI Centroid\nComm",
]

QUERY_STAGES = [
    "query_centroid_dist_ms",
    "query_scan_ms",
    "query_openmp_merge_ms",
    "query_mpi_bcast_ms",
    "query_mpi_gather_ms",
    "query_mpi_merge_ms",
    "query_result_assemble_ms",
]

QUERY_STAGE_LABELS = [
    "Centroid\nDistances",
    "Scan",
    "OpenMP\nMerge",
    "MPI\nBroadcast",
    "MPI\nGather",
    "MPI\nMerge",
    "Result\nAssembly",
]

OPENMP_SERIES_RE = re.compile(r"^OpenMP t=(\d+)$")
MPI_SERIES_RE = re.compile(r"^MPI N=(\d+) n=(\d+)$")


def choose_time_unit(max_ms: float) -> tuple[float, str, str]:
    if max_ms < 10.0:
        return 0.001, "Microseconds (μs)", "μs"
    if max_ms < 10_000:
        return 1.0, "Milliseconds (ms)", "ms"
    if max_ms < 120_000:
        return 1_000.0, "Seconds (s)", "s"
    return 60_000.0, "Minutes (min)", "min"


def series_sort_key(series: str) -> tuple[int, int, int]:
    if series == "Serial":
        return (0, 0, 0)

    openmp_match = OPENMP_SERIES_RE.fullmatch(series)
    if openmp_match:
        return (1, int(openmp_match.group(1)), 0)

    mpi_match = MPI_SERIES_RE.fullmatch(series)
    if mpi_match:
        return (2, int(mpi_match.group(1)), int(mpi_match.group(2)))

    return (3, 0, 0)


def get_series_order(series_values: list[str]) -> list[str]:
    return sorted(series_values, key=series_sort_key)


def build_series_palette(series_order: list[str]) -> dict[str, tuple[float, float, float]]:
    colors = sns.color_palette("tab20", n_colors=max(1, len(series_order)))
    return {series: color for series, color in zip(series_order, colors)}


def get_x_tick_fontsize(series_count: int) -> float:
    return max(6.0, min(11.0, 12.0 - (series_count * 0.3)))


def write_legend_image(
    series_order: list[str],
    series_palette: dict[str, tuple[float, float, float]],
    output_path: Path,
) -> None:
    if not series_order:
        print(f"warning: no series found for legend {output_path.name}")
        return

    fig_width = max(12.0, 1.7 * len(series_order))
    fig_height = 1.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.5, len(series_order) - 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")

    swatch_width = 0.90
    swatch_height = 0.36
    swatch_y = 0.60
    label_y = 0.38

    for idx, series in enumerate(series_order):
        color = series_palette[series]
        ax.add_patch(
            Rectangle(
                (idx - swatch_width / 2.0, swatch_y),
                swatch_width,
                swatch_height,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            idx,
            label_y,
            series,
            ha="right",
            va="top",
            fontsize=18,
            rotation=40,
            rotation_mode="anchor",
            fontweight="bold",
        )

    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
        key=lambda p: p.stat().st_mtime,
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


def parse_stage_value(text: str, key: str, is_mpi: bool = False) -> float | None:
    if is_mpi:
        m = re.search(rf"^rank=0\s+{key}={FLOAT_RE}", text, re.MULTILINE)
        if m:
            return float(m.group(1))
        return None
    # For non-MPI: match key=value at line start, or after a newline (not prefixed by rank=)
    for m in re.finditer(rf"^{key}={FLOAT_RE}", text, re.MULTILINE):
        return float(m.group(1))
    return None


def parse_stage_timings(log_file: Path, stage_keys: list[str]) -> dict[str, float]:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    is_mpi = "mpi" in log_file.name
    result: dict[str, float] = {}
    for key in stage_keys:
        val = parse_stage_value(text, key, is_mpi)
        if val is not None:
            result[key] = val
    return result


def collect_records(
    logs_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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


def collect_stage_records(
    logs_dir: Path, stage_keys: list[str]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    dataset_dirs = sorted([p for p in logs_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        for log_file in sorted(dataset_dir.glob("*.log"), key=lambda p: p.name):
            series = parse_series_name(log_file)
            if not series:
                continue

            timings = parse_stage_timings(log_file, stage_keys)
            if timings:
                record: dict[str, object] = {
                    "dataset": dataset,
                    "series": series,
                }
                record.update(timings)
                records.append(record)

    return records


def plot_metric(
    records: list[dict[str, object]],
    title: str,
    output_path: Path,
    series_order: list[str],
    series_palette: dict[str, tuple[float, float, float]],
) -> None:
    if not records:
        print(f"warning: no data found for plot {output_path.name}")
        return

    sns.set_theme(style="whitegrid", context="talk")
    df = pd.DataFrame(records)
    datasets = sorted(df["dataset"].unique())
    series_in_data = [s for s in series_order if any(df["series"] == s)]

    ncols = 2
    nrows = max(1, math.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.2 * nrows))
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
            palette=series_palette,
            estimator="mean",
            errorbar=None,
            dodge=False,
            ax=ax,
        )
        bar_patches = list(ax.patches)

        group_bands = [
            ("OpenMP", "#4C72B0"),
            ("MPI", "#55A868"),
        ]
        band_regions: list[tuple[float, float, str]] = []
        for group_prefix, band_color in group_bands:
            group_positions = [
                pos
                for pos, series_name in enumerate(series_in_data)
                if series_name.startswith(group_prefix)
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

        for patch in bar_patches:
            height = patch.get_height()
            if not math.isfinite(height):
                continue
            x_center = patch.get_x() + (patch.get_width() / 2.0)
            label = f"{height:.0f}" if unit in ("ms", "μs") else f"{height:.1f}"
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
        ax.set_xlabel("")
        ax.set_ylabel(y_label)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, _pos: f"{val:.0f}" if unit in ("ms", "μs") else f"{val:.1f}")
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
        x_tick_fontsize = get_x_tick_fontsize(len(series_in_data))
        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("right")
            tick_label.set_rotation_mode("anchor")
            tick_label.set_fontsize(x_tick_fontsize)

    for idx in range(len(datasets), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=18, y=0.98)
    fig.subplots_adjust(top=0.90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_stages(
    records: list[dict[str, object]],
    stage_keys: list[str],
    stage_labels: list[str],
    title: str,
    output_path: Path,
    series_order: list[str],
) -> None:
    if not records:
        print(f"warning: no data found for plot {output_path.name}")
        return

    sns.set_theme(style="whitegrid", context="talk")
    df = pd.DataFrame(records)
    for k in stage_keys:
        if k not in df.columns:
            df[k] = 0.0
    datasets = sorted(df["dataset"].unique())
    series_in_data = [s for s in series_order if any(df["series"] == s)]

    ncols = 2
    nrows = max(1, math.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.2 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    # Build a palette for stages
    n_stages = len(stage_keys)
    stage_colors = sns.color_palette("Set2", n_colors=n_stages)

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        df_ds = df[df["dataset"] == dataset]

        pivot = df_ds.pivot_table(index="series", values=stage_keys, aggfunc="first")
        # Reindex to match series_order; fill missing with 0
        pivot = pivot.reindex(series_in_data, fill_value=0.0)

        totals = pivot.sum(axis=1)
        if totals.empty or totals.max() == 0:
            ax.set_title(f"{dataset} (no data)")
            continue

        max_val = totals.max()
        scale, y_label, unit = choose_time_unit(max_val)

        x_pos = list(range(len(series_in_data)))
        cum_bottom = [0.0] * len(series_in_data)

        for si, key in enumerate(stage_keys):
            values = (pivot[key] / scale).values
            bars = ax.bar(
                x_pos,
                values,
                bottom=cum_bottom,
                width=0.65,
                color=stage_colors[si],
                edgecolor="white",
                linewidth=0.4,
                label=stage_labels[si] if idx == 0 else None,
            )
            # Annotate segments that are large enough to show
            for xi, v in enumerate(values):
                if v > 0 and v > max_val / scale * 0.04:
                    ax.annotate(
                        f"{v:.1f}",
                        (xi, cum_bottom[xi] + v / 2),
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="black",
                        fontweight="bold",
                    )
            for xi in range(len(series_in_data)):
                cum_bottom[xi] += values[xi]

        # Add total annotation on top of each bar
        for xi, series_name in enumerate(series_in_data):
            total_val = totals.iloc[xi] / scale
            if total_val > 0:
                ax.annotate(
                    f"{total_val:.0f}" if unit in ("ms", "μs") else f"{total_val:.1f}",
                    (xi, totals.iloc[xi] / scale),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

        ax.set_title(str(dataset))
        ax.set_xlabel("")
        ax.set_ylabel(y_label)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(series_in_data, rotation=40, ha="right", fontsize=get_x_tick_fontsize(len(series_in_data)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, _pos: f"{val:.0f}" if unit in ("ms", "μs") else f"{val:.1f}")
        )
        ymax_val = float(totals.max() / scale)
        ax.set_ylim(0, ymax_val * 1.15 if ymax_val > 0 else 1.0)

        # Group bands for OpenMP/MPI
        group_bands = [
            ("OpenMP", "#4C72B0"),
            ("MPI", "#55A868"),
        ]
        group_labels_shown = []
        for group_prefix, band_color in group_bands:
            group_positions = [
                pos
                for pos, series_name in enumerate(series_in_data)
                if series_name.startswith(group_prefix)
            ]
            if group_positions:
                gs = min(group_positions) - 0.5
                ge = max(group_positions) + 0.5
                ax.axvspan(gs, ge, color=band_color, alpha=0.08, zorder=0)
                group_labels_shown.append(
                    ((gs + ge) / 2.0, ymax_val * 1.12, group_prefix)
                )

        for gx, gy, gname in group_labels_shown:
            ax.text(gx, gy, gname, ha="center", va="bottom", fontsize=8, color="#444444", fontweight="semibold")

    for idx in range(len(datasets), len(axes_flat)):
        axes_flat[idx].axis("off")

    # Add legend for stages
    if n_stages > 0 and len(datasets) > 0:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=stage_colors[si], edgecolor="white")
            for si in range(n_stages)
        ]
        # Simplify labels: remove newlines for legend
        legend_labels = [lb.replace("\n", " ") for lb in stage_labels]
        axes_flat[0].legend(
            handles,
            legend_labels,
            loc="upper left",
            fontsize=7,
            framealpha=0.9,
        )

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
    all_series_values = [str(r["series"]) for r in (build_records + latency_records)]
    series_order = get_series_order(sorted(set(all_series_values)))
    series_palette = build_series_palette(series_order)

    build_plot = logs_dir / "cluster_build_time_ms.png"
    latency_plot = logs_dir / "query_latency_mean_ms.png"
    legend_plot = logs_dir / "series_legend.png"

    cluster_stage_records = collect_stage_records(logs_dir, CLUSTER_BUILD_STAGES)
    query_stage_records = collect_stage_records(logs_dir, QUERY_STAGES)

    # Existing plots
    plot_metric(
        build_records,
        title="Cluster Build Time by Dataset",
        output_path=build_plot,
        series_order=series_order,
        series_palette=series_palette,
    )
    plot_metric(
        latency_records,
        title="Query Mean Latency by Dataset",
        output_path=latency_plot,
        series_order=series_order,
        series_palette=series_palette,
    )
    write_legend_image(series_order=series_order, series_palette=series_palette, output_path=legend_plot)

    print(f"Wrote plot: {build_plot}")
    print(f"Wrote plot: {latency_plot}")
    print(f"Wrote plot: {legend_plot}")

    # Stacked stage breakdown plots
    cluster_stacked_plot = logs_dir / "cluster_build_stacked.png"
    query_stacked_plot = logs_dir / "query_stacked.png"

    plot_stacked_stages(
        cluster_stage_records,
        CLUSTER_BUILD_STAGES,
        CLUSTER_BUILD_STAGE_LABELS,
        title="Cluster Build Breakdown by Dataset",
        output_path=cluster_stacked_plot,
        series_order=series_order,
    )
    plot_stacked_stages(
        query_stage_records,
        QUERY_STAGES,
        QUERY_STAGE_LABELS,
        title="Query Stage Breakdown by Dataset",
        output_path=query_stacked_plot,
        series_order=series_order,
    )

    print(f"Wrote plot: {cluster_stacked_plot}")
    print(f"Wrote plot: {query_stacked_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
