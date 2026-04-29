#!/usr/bin/env python3
"""
Fashion MNIST Cluster Visualization — poster edition
Pipeline: load 60k train images → k-means(10) on 20k subsample → t-SNE(5k) → plot
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import h5py
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Config ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR = os.getenv("SCRATCH")
if not SCRATCH_DIR:
    raise RuntimeError("SCRATCH environment variable is not set")

DATA_FILE  = os.path.join(SCRATCH_DIR, "data", "fashion-mnist-784-euclidean.hdf5")
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SCATTER = str(PLOTS_DIR / "fashion_mnist_tsne.png")
OUTPUT_GALLERY = str(PLOTS_DIR / "fashion_mnist_gallery.png")
N_CLUSTERS = 10
N_TSNE     = 5000   # points fed to t-SNE
N_KMEANS   = 20000  # points used to fit k-means
N_GALLERY  = 4      # sample images shown per cluster
SEED       = 42

BG        = "#ffffff"
ACCENT    = "#1a1a1a"
CARD_BG   = "#f7f7f7"
CARD_EDGE = "#e0e0e0"
RED       = "#c0392b"

# 10 visually distinct colours that read well on white
PALETTE = [
    "#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad",
    "#d35400", "#16a085", "#c0392b", "#2471a3", "#1e8449",
]
# ensure all 10 are unique
PALETTE = [
    "#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad",
    "#f39c12", "#16a085", "#6c3483", "#1a5276", "#196f3d",
]

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data …")
with h5py.File(DATA_FILE, "r") as f:
    train = f["train"][:]          # (60000, 784) float32, range [0,255]
X = (train / 255.0).astype(np.float32)
print(f"  {X.shape[0]:,} images × {X.shape[1]} dims")

# ── K-Means ────────────────────────────────────────────────────────────────────
rng = np.random.RandomState(SEED)
km_idx = rng.choice(len(X), N_KMEANS, replace=False)
print(f"Fitting k-means on {N_KMEANS:,} points …")
km = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=SEED)
km.fit(X[km_idx])
centroids = km.cluster_centers_          # (10, 784)

print("Predicting labels for full dataset …")
all_labels = km.predict(X)               # (60000,)

# ── t-SNE subsample ────────────────────────────────────────────────────────────
ts_idx    = rng.choice(len(X), N_TSNE, replace=False)
X_ts      = X[ts_idx]
labels_ts = all_labels[ts_idx]

print(f"Running t-SNE on {N_TSNE:,} points (this takes ~3–5 min) …")
tsne   = TSNE(n_components=2, perplexity=40, max_iter=1000,
              learning_rate="auto", init="pca", random_state=SEED)
E      = tsne.fit_transform(X_ts.astype(np.float64))  # (N_TSNE, 2)
print("  t-SNE done.")

# ── Representative images per cluster ─────────────────────────────────────────
gallery = {}
for c in range(N_CLUSTERS):
    pool   = np.where(all_labels == c)[0]
    chosen = rng.choice(pool, N_GALLERY, replace=False)
    gallery[c] = X[chosen]          # (N_GALLERY, 784)

# Sort clusters by their centroid's dominant-pixel position (just for visual order)
cluster_order = list(range(N_CLUSTERS))

THUMB_SIZE = 28

# ── Figure 1: t-SNE scatter ────────────────────────────────────────────────────
fig1, ax_sc = plt.subplots(figsize=(12, 10), facecolor=BG)
fig1.patch.set_facecolor(BG)
ax_sc.set_facecolor(BG)
for spine in ax_sc.spines.values():
    spine.set_edgecolor(CARD_EDGE)

for c in cluster_order:
    mask = labels_ts == c
    ax_sc.scatter(E[mask, 0], E[mask, 1],
                  c=PALETTE[c], s=50, alpha=0.65, linewidths=0,
                  label=f"Cluster {c}", zorder=2)

for c in cluster_order:
    mask = labels_ts == c
    cx, cy = E[mask, 0].mean(), E[mask, 1].mean()
    ax_sc.text(cx, cy, str(c),
               fontsize=15, fontweight="bold", color=PALETTE[c], ha="center", va="center",
               path_effects=[pe.withStroke(linewidth=3, foreground=BG)], zorder=4)

ax_sc.set_xticks([]); ax_sc.set_yticks([])
ax_sc.set_xlabel("t-SNE dimension 1", color=ACCENT, fontsize=12, labelpad=6)
ax_sc.set_ylabel("t-SNE dimension 2", color=ACCENT, fontsize=12, labelpad=6)
ax_sc.tick_params(colors=ACCENT)
ax_sc.set_title(
    f"Fashion MNIST  ·  t-SNE Embedding coloured by k-means cluster\n"
    f"K-Means (k={N_CLUSTERS}) fitted on {N_KMEANS:,} samples  ·  "
    f"t-SNE projection of {N_TSNE:,} points",
    color=ACCENT, fontsize=13, pad=12, loc="center")

print(f"Saving → {OUTPUT_SCATTER}")
fig1.savefig(OUTPUT_SCATTER, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig1)

# ── Figure 2: cluster gallery ──────────────────────────────────────────────────
fig2 = plt.figure(figsize=(16, 8), facecolor=BG)
fig2.patch.set_facecolor(BG)

fig2.text(0.5, 0.97,
          "Fashion MNIST  ·  Cluster Centroids & Representative Images",
          ha="center", va="top", fontsize=18, fontweight="bold", color=ACCENT)
fig2.text(0.5, 0.945,
          f"Each card: cluster centroid (top) + {N_GALLERY} random members (bottom)",
          ha="center", va="top", fontsize=11, color="#8888aa")

gallery_gs = gridspec.GridSpec(2, 5, figure=fig2,
                                hspace=0.40, wspace=0.22,
                                left=0.03, right=0.97, top=0.90, bottom=0.04)

for row in range(2):
    for col in range(5):
        c = row * 5 + col
        card_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gallery_gs[row, col], hspace=0.12, height_ratios=[1.2, 1]
        )

        ax_cen = fig2.add_subplot(card_gs[0])
        ax_cen.set_facecolor(CARD_BG)
        for sp in ax_cen.spines.values():
            sp.set_edgecolor(PALETTE[c])
            sp.set_linewidth(2.5)
        ax_cen.imshow(centroids[c].reshape(THUMB_SIZE, THUMB_SIZE),
                      cmap="gray", vmin=0, vmax=1, aspect="equal")
        ax_cen.set_xticks([]); ax_cen.set_yticks([])
        ax_cen.set_title(f"Cluster {c}",
                         color=PALETTE[c], fontsize=11, fontweight="bold", pad=3)

        samp_gs = gridspec.GridSpecFromSubplotSpec(
            2, N_GALLERY // 2, subplot_spec=card_gs[1], wspace=0.05, hspace=0.05
        )
        for i in range(N_GALLERY):
            row_i, col_i = divmod(i, N_GALLERY // 2)
            ax_s = fig2.add_subplot(samp_gs[row_i, col_i])
            ax_s.set_facecolor(CARD_BG)
            for sp in ax_s.spines.values():
                sp.set_edgecolor(CARD_EDGE)
                sp.set_linewidth(0.8)
            ax_s.imshow(gallery[c][i].reshape(THUMB_SIZE, THUMB_SIZE),
                        cmap="gray", vmin=0, vmax=1, aspect="equal")
            ax_s.set_xticks([]); ax_s.set_yticks([])

print(f"Saving → {OUTPUT_GALLERY}")
fig2.savefig(OUTPUT_GALLERY, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print("Done.")
