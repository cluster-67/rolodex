#!/usr/bin/env python3
"""
Render OpenMP k-means cluster timeline frames from debug snapshots.

Frames produced:
  - t=0 synthetic frame (all points gray)
  - one frame per saved debug snapshot in data/debug
"""

import argparse
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


PALETTE = [
    "#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad",
    "#f39c12", "#16a085", "#6c3483", "#1a5276", "#196f3d",
]
np.random.shuffle(PALETTE)
CLUSTER_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
T0_COLOR = "#777777"
BG = "#ffffff"
ACCENT = "#1a1a1a"
CARD_EDGE = "#e0e0e0"

DEFAULT_DEBUG_DIR = Path("data/debug")
DEFAULT_OUTPUT_DIR = Path("scripts/plots/cluster_timeline")
DEFAULT_TSNE_POINTS = 5000
DEFAULT_SEED = 42


def dataset_path_from_name(name: str) -> str:
    scratch_dir = os.getenv("SCRATCH")
    if not scratch_dir:
        raise RuntimeError("SCRATCH environment variable is not set")

    mapping = {
        "fashion-mnist": "fashion-mnist-784-euclidean.hdf5",
        "gist": "gist-960-euclidean.hdf5",
        "mnist": "mnist-784-euclidean.hdf5",
        "sift": "sift-128-euclidean.hdf5",
    }
    if name not in mapping:
        raise ValueError(f"Invalid dataset '{name}'. Expected one of: {', '.join(sorted(mapping))}")
    return os.path.join(scratch_dir, "data", mapping[name])


def parse_iter_and_final(snapshot_path: Path):
    with h5py.File(snapshot_path, "r") as f:
        iteration = int(f.attrs.get("iteration", -1))
        is_final = bool(int(f.attrs.get("is_final", 0)))
    return iteration, is_final


def sorted_snapshots(debug_dir: Path):
    snapshots = sorted(debug_dir.glob("openmp_dataset_*_iter_*.h5"))
    if not snapshots:
        return []
    return sorted(snapshots, key=lambda p: parse_iter_and_final(p))


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.empty(labels.shape[0], dtype=object)
    for c in range(len(PALETTE)):
        colors[labels == c] = PALETTE[c]
    # Any out-of-range labels are rendered in black to make unexpected values obvious.
    colors[~np.isin(labels, np.arange(len(PALETTE), dtype=labels.dtype))] = "#000000"
    return colors


def save_frame(points_2d: np.ndarray, colors: np.ndarray, labels: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(CARD_EDGE)

    if np.all(labels < 0):
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c=colors, s=50, alpha=0.65, linewidths=0, zorder=2)
    else:
        for c in range(len(PALETTE)):
            mask = labels == c
            if not np.any(mask):
                continue
            marker = CLUSTER_MARKERS[c % len(CLUSTER_MARKERS)]
            ax.scatter(points_2d[mask, 0], points_2d[mask, 1], c=PALETTE[c], marker=marker,
                       s=50, alpha=0.65, linewidths=0, zorder=2)

        outlier_mask = ~np.isin(labels, np.arange(len(PALETTE), dtype=labels.dtype))
        if np.any(outlier_mask):
            ax.scatter(points_2d[outlier_mask, 0], points_2d[outlier_mask, 1], c="#000000",
                       marker="x", s=50, alpha=0.8, linewidths=0.6, zorder=3)

        for c in range(len(PALETTE)):
            mask = labels == c
            if not np.any(mask):
                continue
            cx, cy = points_2d[mask, 0].mean(), points_2d[mask, 1].mean()
            ax.text(cx, cy, str(c), fontsize=15, fontweight="bold", color=PALETTE[c],
                    ha="center", va="center", zorder=4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE dimension 1", color=ACCENT, fontsize=12, labelpad=6)
    ax.set_ylabel("t-SNE dimension 2", color=ACCENT, fontsize=12, labelpad=6)
    ax.tick_params(colors=ACCENT)
    ax.set_title(title, color=ACCENT, fontsize=13, pad=12, loc="center")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize OpenMP cluster timeline from debug snapshots.")
    parser.add_argument("--dataset", default="fashion-mnist",
                        choices=["fashion-mnist", "gist", "mnist", "sift"])
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-tsne", type=int, default=DEFAULT_TSNE_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    snapshots = sorted_snapshots(args.debug_dir)
    if not snapshots:
        raise RuntimeError(f"No snapshot files found in: {args.debug_dir}")

    print(f"Found {len(snapshots)} snapshot files")
    data_file = dataset_path_from_name(args.dataset)
    print(f"Loading dataset from {data_file}")
    with h5py.File(data_file, "r") as f:
        train = f["train"][:]
    X = (train / 255.0).astype(np.float32)

    if args.n_tsne <= 0 or args.n_tsne > X.shape[0]:
        raise ValueError(f"--n-tsne must be in [1, {X.shape[0]}]")

    rng = np.random.RandomState(args.seed)
    ts_idx = rng.choice(X.shape[0], args.n_tsne, replace=False)
    X_ts = X[ts_idx]

    print(f"Running fixed t-SNE embedding on {args.n_tsne} points")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    )
    embedding = tsne.fit_transform(X_ts.astype(np.float64))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0_colors = np.full(args.n_tsne, T0_COLOR, dtype=object)
    t0_labels = np.full(args.n_tsne, -1, dtype=np.int32)
    t0_file = args.output_dir / "snapshot_t000000.png"
    save_frame(embedding, t0_colors, t0_labels, t0_file,
               f"{args.dataset}: t=0 (synthetic, all-gray)")
    print(f"Saved {t0_file}")

    for snapshot in snapshots:
        iteration, is_final = parse_iter_and_final(snapshot)
        with h5py.File(snapshot, "r") as f:
            centroids = f["centroids"][:]
            membership = f["membership"][:]
        if centroids.ndim != 2:
            raise RuntimeError(f"Invalid centroids shape in {snapshot}: {centroids.shape}")
        labels_ts = membership[ts_idx]
        colors = labels_to_colors(labels_ts)

        suffix = "_final" if is_final else ""
        out_name = f"snapshot_t{iteration:06d}{suffix}.png"
        out_path = args.output_dir / out_name
        title_suffix = " (final)" if is_final else ""
        save_frame(embedding, colors, labels_ts, out_path,
                   f"{args.dataset}: iteration={iteration}, k={centroids.shape[0]}{title_suffix}")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
