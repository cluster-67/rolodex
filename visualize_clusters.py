#!/usr/bin/env python3
"""
Fashion MNIST Cluster Visualization — poster edition
Pipeline: load 60k train images → k-means(10) on 20k subsample → t-SNE(5k) → plot
"""

import warnings
warnings.filterwarnings("ignore")

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
DATA_FILE  = "/pscratch/sd/a/asv48/data/fashion-mnist-784-euclidean.hdf5"
OUTPUT     = "/global/homes/a/asv48/rolodex/fashion_mnist_clusters.png"
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

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 14), facecolor=BG)
fig.patch.set_facecolor(BG)

# Outer grid: [scatter | gallery]
outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1],
                          left=0.04, right=0.98, top=0.92, bottom=0.06, wspace=0.07)

# ── Left: t-SNE scatter ────────────────────────────────────────────────────────
ax_sc = fig.add_subplot(outer[0])
ax_sc.set_facecolor(BG)
for spine in ax_sc.spines.values():
    spine.set_edgecolor(CARD_EDGE)

for c in cluster_order:
    mask = labels_ts == c
    ax_sc.scatter(E[mask, 0], E[mask, 1],
                  c=PALETTE[c], s=50, alpha=0.65, linewidths=0,
                  label=f"Cluster {c}", zorder=2)

# Annotate cluster centroids on the t-SNE plot
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
ax_sc.set_title("t-SNE Embedding  ·  coloured by k-means cluster",
                color=ACCENT, fontsize=14, pad=10, loc="left")

# ── Right: gallery ─────────────────────────────────────────────────────────────
# 10 clusters arranged in 2 rows × 5 cols
# Each "card" shows: centroid (large) + N_GALLERY sample images (small, in a row)
inner = gridspec.GridSpecFromSubplotSpec(
    2, 5, subplot_spec=outer[1], hspace=0.35, wspace=0.20
)

THUMB_SIZE = 28    # all images are 28×28

for row in range(2):
    for col in range(5):
        c = row * 5 + col
        card_spec = inner[row, col]

        # Each card: 2 sub-rows — [centroid | samples strip]
        card_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=card_spec, hspace=0.12, height_ratios=[1.2, 1]
        )

        # ── Centroid ──
        ax_cen = fig.add_subplot(card_gs[0])
        ax_cen.set_facecolor(CARD_BG)
        for sp in ax_cen.spines.values():
            sp.set_edgecolor(PALETTE[c])
            sp.set_linewidth(2.5)
        cen_img = centroids[c].reshape(THUMB_SIZE, THUMB_SIZE)
        ax_cen.imshow(cen_img, cmap="gray", vmin=0, vmax=1, aspect="equal")
        ax_cen.set_xticks([]); ax_cen.set_yticks([])
        ax_cen.set_title(f"Cluster {c}",
                         color=PALETTE[c], fontsize=11, fontweight="bold", pad=3)

        # ── Sample strip (2 × 2 grid) ──
        samp_gs = gridspec.GridSpecFromSubplotSpec(
            2, N_GALLERY // 2, subplot_spec=card_gs[1], wspace=0.05, hspace=0.05
        )
        for i in range(N_GALLERY):
            row_i, col_i = divmod(i, N_GALLERY // 2)
            ax_s = fig.add_subplot(samp_gs[row_i, col_i])
            ax_s.set_facecolor(CARD_BG)
            for sp in ax_s.spines.values():
                sp.set_edgecolor(CARD_EDGE)
                sp.set_linewidth(0.8)
            img = gallery[c][i].reshape(THUMB_SIZE, THUMB_SIZE)
            ax_s.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal")
            ax_s.set_xticks([]); ax_s.set_yticks([])

# ── Titles ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.965,
         "Fashion MNIST  ·  K-Means Clustering Visualization",
         ha="center", va="top", fontsize=22, fontweight="bold", color=ACCENT)
fig.text(0.5, 0.937,
         f"K-Means (k={N_CLUSTERS}) fitted on {N_KMEANS:,} samples  ·  "
         f"t-SNE projection of {N_TSNE:,} points  ·  "
         "Each card: cluster centroid (top) + 6 random members (bottom)",
         ha="center", va="top", fontsize=11, color="#8888aa")

# ── Right-panel header ─────────────────────────────────────────────────────────
# Compute bounding box of the right subplot in figure coords
r_pos = outer[1].get_position(fig)
fig.text((r_pos.x0 + r_pos.x1) / 2, r_pos.y1 + 0.014,
         "Cluster Centroids & Representative Images",
         ha="center", va="bottom", fontsize=13, color=ACCENT, fontstyle="italic")

print(f"Saving → {OUTPUT}")
plt.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("Done.")
