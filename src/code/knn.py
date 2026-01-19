import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap

# -----------------------------
# 1) Toy dataset (2 classes, 2 features)
# -----------------------------
rng = np.random.default_rng(0)
n = 80

X0 = rng.normal(loc=[-1.2, -0.8], scale=1.0, size=(n // 2, 2))
X1 = rng.normal(loc=[+1.2, +0.8], scale=1.0, size=(n // 2, 2))

# optional noisy points near boundary
m = 10
X_noise = rng.normal(loc=[0.0, 0.0], scale=0.45, size=(m, 2))
y_noise = rng.integers(0, 2, size=m)

X = np.vstack([X0, X1, X_noise])
y = np.hstack([np.zeros(len(X0), dtype=int), np.ones(len(X1), dtype=int), y_noise])

df = pd.DataFrame(X, columns=["x1", "x2"])
df["y"] = y

# -----------------------------
# 2) k-NN distance-weighted probability
# -----------------------------
lam = 3.0  # fixed across plots

def knn_proba_weighted(X_train, y_train, X_query, k=5, lam=3.0):
    D = cdist(X_query, X_train, metric="euclidean")
    nn_idx = np.argpartition(D, kth=k-1, axis=1)[:, :k]
    Dk = np.take_along_axis(D, nn_idx, axis=1)
    W = np.exp(-lam * (Dk ** 2))
    votes = y_train[nn_idx].astype(float)
    p1 = (W * votes).sum(axis=1) / (W.sum(axis=1) + 1e-12)
    return p1

# -----------------------------
# 3) Grid for plotting
# -----------------------------
pad = 0.8
x1_min, x1_max = df["x1"].min() - pad, df["x1"].max() + pad
x2_min, x2_max = df["x2"].min() - pad, df["x2"].max() + pad

gx1 = np.linspace(x1_min, x1_max, 320)
gx2 = np.linspace(x2_min, x2_max, 260)
G1, G2 = np.meshgrid(gx1, gx2)
G = np.c_[G1.ravel(), G2.ravel()]

region_cmap = ListedColormap(["#cfe3f5", "#fde0c2"])

# -----------------------------
# 4) Three vertical plots with increasing k
# -----------------------------
Ks = [1, 6, 16]  # low -> high k
sigma = 2.4      # plot smoothing for clean boundary (same for all)

fig, axes = plt.subplots(3, 1, figsize=(8.5, 13.5), dpi=130)

for ax, k in zip(axes, Ks):
    P = knn_proba_weighted(X, y, G, k=k, lam=lam).reshape(G1.shape)
    P_smooth = gaussian_filter(P, sigma=sigma)

    # regions + boundary from smoothed probability
    ax.contourf(G1, G2, (P_smooth >= 0.5).astype(int),
                levels=[-0.5, 0.5, 1.5], cmap=region_cmap, alpha=0.35)

    # thinner boundary
    ax.contour(G1, G2, P_smooth, levels=[0.5], colors="black", linewidths=1.0)

    ax.scatter(df.loc[df.y == 0, "x1"], df.loc[df.y == 0, "x2"],
               s=34, alpha=0.95, edgecolors="none", color="#1f77b4", label="Class 0")
    ax.scatter(df.loc[df.y == 1, "x1"], df.loc[df.y == 1, "x2"],
               s=34, alpha=0.95, edgecolors="none", color="#ff7f0e", label="Class 1")

    ax.set_title(f"k-NN (Distance-weighted): (k={k})", pad=10)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.grid(True, alpha=0.25, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True,
              loc="lower right",
              bbox_to_anchor=(0.98, 0.02),
              edgecolor="black",
              facecolor="white")

    ax.text(0.98, 0.95,
            rf"$k={k},\ \lambda={lam}$" + "\n" + rf"smooth $\sigma={sigma}$",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white",
                      edgecolor="black",
                      linewidth=1.0))

plt.tight_layout()
plt.show()
