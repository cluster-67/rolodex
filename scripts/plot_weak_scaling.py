#!/usr/bin/env python3
"""
Plot weak scaling curves from bench_weak_scaling.sh logs.

bench_weak_scaling.sh uses one subdirectory per deep1b size; each folder is one
(dataset, P) pair (e.g. deep1b-1K with P=2). Within that folder, runs differ by
Slurm node count N and backend (OpenMP vs MPI).

Layout: **one figure.** Series are **OpenMP** (single curve) and **MPI N={slurm_nodes}**
(one curve per node count). Points from all dataset subdirs merge so each line is vs
**P** (threads or MPI ranks) on **log₂**; **Y** is **log₁₀** of the metric.

Reads:
  <dataset>/openmp-N<N>-t<P>.log
  <dataset>/mpi-partition-N<N>-n<P>.log

Writes in the run directory (default: latest data/logs/bench-weak-*):
  weak_scaling_cluster_build.png
  weak_scaling_query_mean_ms.png
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns

FLOAT_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
BUILD_TIME_RE = re.compile(rf"cluster_build_time_ms={FLOAT_RE}")
MEAN_MS_RE = re.compile(rf"aggregate:\s+.*\bmean_ms={FLOAT_RE}")

OPENMP_WEAK_RE = re.compile(r"^openmp-N(\d+)-t(\d+)\.log$")
MPI_PARTITION_RE = re.compile(r"^mpi-partition-N(\d+)-n(\d+)\.log$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot weak scaling curves from bench-weak-* log directories."
    )
    p.add_argument(
        "logs_dir",
        nargs="?",
        default=None,
        help=(
            "Path to a bench-weak run directory (e.g. data/logs/bench-weak-2026-05-13-12-00-00). "
            "If omitted, the latest bench-weak-* under data/logs is used."
        ),
    )
    return p.parse_args()


def resolve_logs_dir(logs_dir_arg: Optional[str]) -> Path:
    root = Path(__file__).resolve().parents[1]
    logs_root = root / "data" / "logs"

    if logs_dir_arg:
        candidate = Path(logs_dir_arg).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()

    bench_dirs = sorted(
        [p for p in logs_root.glob("bench-weak-*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not bench_dirs:
        raise FileNotFoundError(f"No bench-weak-* directories found under {logs_root}")
    return bench_dirs[-1]


def parse_log_filename(path: Path) -> Optional[Tuple[str, int, int]]:
    m = OPENMP_WEAK_RE.fullmatch(path.name)
    if m:
        return ("OpenMP", int(m.group(1)), int(m.group(2)))
    m = MPI_PARTITION_RE.fullmatch(path.name)
    if m:
        return ("MPI", int(m.group(1)), int(m.group(2)))
    return None


def parse_log_metrics(text: str) -> Tuple[Optional[float], Optional[float]]:
    build_m = BUILD_TIME_RE.search(text)
    build_ms = float(build_m.group(1)) if build_m else None
    mean_m = MEAN_MS_RE.search(text)
    mean_ms = float(mean_m.group(1)) if mean_m else None
    return build_ms, mean_ms


def collect_records(logs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dataset_dirs = sorted([p for p in logs_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        for log_file in sorted(dataset_dir.glob("*.log"), key=lambda p: p.name):
            parsed = parse_log_filename(log_file)
            if not parsed:
                continue
            backend, n_nodes, p = parsed
            if backend == "OpenMP":
                series = "OpenMP"
            else:
                series = f"MPI N={n_nodes}"
            text = log_file.read_text(encoding="utf-8", errors="replace")
            build_ms, mean_ms = parse_log_metrics(text)
            rows.append(
                {
                    "dataset": dataset,
                    "backend": backend,
                    "N_nodes": n_nodes,
                    "P": p,
                    "series": series,
                    "cluster_build_time_ms": build_ms,
                    "query_mean_ms": mean_ms,
                }
            )
    return pd.DataFrame(rows)


def _series_order(sub: pd.DataFrame) -> list[str]:
    """Legend / draw order: OpenMP first, then MPI by increasing Slurm N."""
    keys: dict[str, tuple] = {}
    for _, r in sub.iterrows():
        s = str(r["series"])
        if r["backend"] == "OpenMP":
            keys[s] = (0, 0, s)
        else:
            keys[s] = (1, int(r["N_nodes"]), s)
    return sorted(keys.keys(), key=lambda k: keys[k])


def plot_scaling_figure(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    if df.empty or value_col not in df.columns:
        print(f"warning: no data for {output_path.name}")
        return

    sub = df[df[value_col].notna()].copy()
    sub = sub[sub[value_col] > 0]
    if sub.empty:
        print(f"warning: no positive {value_col} for log y-axis in {output_path.name}")
        return

    series_names = _series_order(sub)
    if not series_names:
        print(f"warning: no series for {output_path.name}")
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7.5))

    palette = sns.color_palette("tab20", n_colors=max(20, len(series_names)))

    for i, sname in enumerate(series_names):
        g = sub[sub["series"] == sname].sort_values("P")
        g = g.drop_duplicates(subset=["P"], keep="last")
        if g.empty:
            continue
        x = g["P"].astype(float).values
        y = g[value_col].astype(float).values
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.85,
            markersize=6,
            label=sname,
            color=palette[i % len(palette)],
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Parallelism P (threads or MPI ranks, log₂ scale)")
    ax.set_ylabel(y_label)
    ax.set_yscale("log", base=10)
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _pos: f"{int(round(x))}" if abs(x - round(x)) < 1e-6 else f"{x:g}"
        )
    )
    p_pos = sub["P"].astype(float)
    ax.set_xlim(float(p_pos.min()) * 0.88, float(p_pos.max()) * 1.12)
    y_pos = sub[value_col].astype(float)
    ymin = float(y_pos.min())
    ymax = float(y_pos.max())
    ax.set_ylim(ymin * 0.85, ymax * 1.15)
    ax.legend(
        loc="upper left",
        ncol=1,
        fontsize=7.5,
        framealpha=0.95,
    )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        logs_dir = resolve_logs_dir(args.logs_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not logs_dir.is_dir():
        print(f"error: not a directory: {logs_dir}", file=sys.stderr)
        return 1

    print(f"Using logs directory: {logs_dir}")
    df = collect_records(logs_dir)
    if df.empty:
        print(
            "error: no matching weak-scaling logs found "
            "(expected openmp-N*-t*.log or mpi-partition-*.log under dataset subdirs)",
            file=sys.stderr,
        )
        return 1

    out_build = logs_dir / "weak_scaling_cluster_build.png"
    out_query = logs_dir / "weak_scaling_query_mean_ms.png"

    plot_scaling_figure(
        df,
        "cluster_build_time_ms",
        title="Weak Scaling (Cluster Build Time vs Parallelism P)",
        y_label="Cluster Build Time (ms, log10 scale)",
        output_path=out_build,
    )
    plot_scaling_figure(
        df,
        "query_mean_ms",
        title="Weak scaling: validation query mean latency vs parallelism P",
        y_label="Mean Query Latency (ms, log10 scale)",
        output_path=out_query,
    )

    print(f"Wrote {out_build}")
    print(f"Wrote {out_query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
