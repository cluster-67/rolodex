#!/usr/bin/env python3
"""
Render MPI k-means cluster timeline frames from per-rank debug snapshots.

For each saved iteration, produces one frame per MPI rank showing only
that rank's locally-owned points colored by their cluster membership.
All frames share a fixed t-SNE embedding and axis limits for spatial
consistency across ranks and iterations.

Snapshots are written by the MPI C++ code every 2 iterations to data/debug/mpi/.

Frame naming (flat output directory):
  snapshot_t000000_rank{r}.png              -- synthetic t=0 (all gray)
  snapshot_t{iter:06d}_rank{r}.png          -- per-rank membership at iteration
  snapshot_t{iter:06d}_final_rank{r}.png    -- final convergence frame
"""

import argparse
import os
import re
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

DEFAULT_DEBUG_DIR = Path("data/debug/mpi")
DEFAULT_OUTPUT_DIR = Path("scripts/plots/mpi_cluster_timeline")
DEFAULT_TSNE_POINTS = 5000
DEFAULT_SEED = 42

_SNAP_RE = re.compile(r"mpi_rank_(\d+)_iter_(\d+)(_final)?\.h5$")


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


def sorted_snapshots(debug_dir: Path):
    def sort_key(p):
        m = _SNAP_RE.search(p.name)
        if not m:
            return (99999999, False, 99999)
        return (int(m.group(2)), m.group(3) is not None, int(m.group(1)))

    snapshots = list(debug_dir.glob("mpi_rank_*_iter_*.h5"))
    return sorted(snapshots, key=sort_key)


def load_snapshot(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {
            "rank": int(f.attrs["rank"]),
            "iteration": int(f.attrs["iteration"]),
            "is_final": bool(int(f.attrs["is_final"])),
            "global_offset": int(f.attrs["global_offset"]),
            "local_n": int(f.attrs["local_n"]),
            "global_n": int(f.attrs["global_n"]),
            "num_ranks": int(f.attrs["num_ranks"]),
            "centroids": f["centroids"][:],
            "membership": f["membership"][:],
        }


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.empty(labels.shape[0], dtype=object)
    for c in range(len(PALETTE)):
        colors[labels == c] = PALETTE[c]
    colors[~np.isin(labels, np.arange(len(PALETTE), dtype=labels.dtype))] = "#000000"
    return colors


def draw_rank_panel(ax, full_embedding, rank_mask, labels, subtitle, xlim, ylim):
    """Draw one rank's panel onto an existing Axes."""
    fg = full_embedding[rank_mask]
    bg = full_embedding[~rank_mask]

    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(CARD_EDGE)

    # Non-owned points: faint ghost background for spatial context.
    if len(bg) > 0:
        ax.scatter(bg[:, 0], bg[:, 1], c="#cccccc", s=6, alpha=0.15,
                   linewidths=0, zorder=1)

    # Owned points: colored by cluster (or all gray at t=0).
    if np.all(labels < 0):
        ax.scatter(fg[:, 0], fg[:, 1], c=T0_COLOR, s=35, alpha=0.75,
                   linewidths=0, zorder=2)
    else:
        for c in range(len(PALETTE)):
            mask = labels == c
            if not np.any(mask):
                continue
            marker = CLUSTER_MARKERS[c % len(CLUSTER_MARKERS)]
            ax.scatter(fg[mask, 0], fg[mask, 1], c=PALETTE[c], marker=marker,
                       s=35, alpha=0.80, linewidths=0, zorder=2)

        outlier_mask = ~np.isin(labels, np.arange(len(PALETTE), dtype=labels.dtype))
        if np.any(outlier_mask):
            ax.scatter(fg[outlier_mask, 0], fg[outlier_mask, 1], c="#000000",
                       marker="x", s=35, alpha=0.8, linewidths=0.6, zorder=3)

        for c in range(len(PALETTE)):
            mask = labels == c
            if not np.any(mask):
                continue
            cx, cy = fg[mask, 0].mean(), fg[mask, 1].mean()
            ax.text(cx, cy, str(c), fontsize=11, fontweight="bold", color=PALETTE[c],
                    ha="center", va="center", zorder=4)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(subtitle, color=ACCENT, fontsize=11, pad=6)


def save_grid(full_embedding, rank_masks, rank_labels, out_path, title, xlim, ylim, num_ranks):
    """Save a grid of rank panels — 2 columns, ceil(num_ranks/2) rows."""
    ncols = 2
    nrows = (num_ranks + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13 * ncols, 10 * nrows), facecolor=BG)
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten()

    for r in range(num_ranks):
        draw_rank_panel(axes_flat[r], full_embedding, rank_masks[r], rank_labels[r],
                        f"rank {r}", xlim, ylim)

    # Hide unused panels (when num_ranks is odd).
    for r in range(num_ranks, len(axes_flat)):
        axes_flat[r].set_visible(False)

    fig.suptitle(title, color=ACCENT, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize MPI k-means cluster timeline.")
    parser.add_argument("--dataset", default="fashion-mnist",
                        choices=["fashion-mnist", "gist", "mnist", "sift"])
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-tsne", type=int, default=DEFAULT_TSNE_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    snapshot_paths = sorted_snapshots(args.debug_dir)
    if not snapshot_paths:
        raise RuntimeError(f"No MPI snapshot files found in: {args.debug_dir}")

    print(f"Found {len(snapshot_paths)} snapshot files")
    snapshots = [load_snapshot(p) for p in snapshot_paths]

    num_ranks = snapshots[0]["num_ranks"]
    global_n = snapshots[0]["global_n"]

    # Collect static per-rank slice info (global_offset, local_n) from metadata.
    # These are identical across iterations for a given rank.
    rank_info = {}
    for snap in snapshots:
        r = snap["rank"]
        if r not in rank_info:
            rank_info[r] = (snap["global_offset"], snap["local_n"])

    print(f"Detected {num_ranks} MPI rank(s), {global_n} total points")
    for r in sorted(rank_info):
        off, n = rank_info[r]
        print(f"  rank {r}: global indices [{off}, {off + n})")

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

    # Fixed axis limits so all frames share the same spatial coordinate system.
    pad = 0.05
    x_range = embedding[:, 0].max() - embedding[:, 0].min()
    y_range = embedding[:, 1].max() - embedding[:, 1].min()
    xlim = (embedding[:, 0].min() - pad * x_range, embedding[:, 0].max() + pad * x_range)
    ylim = (embedding[:, 1].min() - pad * y_range, embedding[:, 1].max() + pad * y_range)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def make_rank_mask_and_labels(rank, membership):
        global_offset, local_n = rank_info[rank]
        mask = (ts_idx >= global_offset) & (ts_idx < global_offset + local_n)
        local_indices = ts_idx[mask] - global_offset
        return mask, membership[local_indices]

    # Synthetic t=0 grid — all owned points gray, non-owned faint background.
    t0_masks = []
    t0_labels = []
    for r in sorted(rank_info):
        global_offset, local_n = rank_info[r]
        mask = (ts_idx >= global_offset) & (ts_idx < global_offset + local_n)
        t0_masks.append(mask)
        t0_labels.append(np.full(mask.sum(), -1, dtype=np.int32))
    out_path = args.output_dir / "snapshot_t000000.png"
    save_grid(embedding, t0_masks, t0_labels, out_path,
              f"{args.dataset} | t=0 (synthetic, all-gray) | {num_ranks} ranks",
              xlim, ylim, num_ranks)
    print(f"Saved {out_path}")

    # Group snapshots by (iteration, is_final) and emit one grid per group.
    from itertools import groupby
    key = lambda s: (s["iteration"], s["is_final"])
    for (iteration, is_final), group in groupby(snapshots, key=key):
        group = sorted(group, key=lambda s: s["rank"])
        masks, labels_list, k = [], [], None
        for snap in group:
            m, lbl = make_rank_mask_and_labels(snap["rank"], snap["membership"])
            masks.append(m)
            labels_list.append(lbl)
            k = snap["centroids"].shape[0]

        suffix = "_final" if is_final else ""
        out_path = args.output_dir / f"snapshot_t{iteration:06d}{suffix}.png"
        title_suffix = " (final)" if is_final else ""
        save_grid(embedding, masks, labels_list, out_path,
                  f"{args.dataset} | iteration={iteration}, k={k}{title_suffix} | {num_ranks} ranks",
                  xlim, ylim, num_ranks)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
