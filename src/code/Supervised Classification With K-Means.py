import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# -----------------------------
# 1) Toy dataset (no labels used for training)
# -----------------------------
rng = np.random.default_rng(0)
n_per = 80

C0 = rng.normal(loc=[-2.0, -1.0], scale=0.7, size=(n_per, 2))
C1 = rng.normal(loc=[+2.0, +1.2], scale=0.7, size=(n_per, 2))
C2 = rng.normal(loc=[-0.2, +2.4], scale=0.7, size=(n_per, 2))
X = np.vstack([C0, C1, C2])

df = pd.DataFrame(X, columns=["x1", "x2"])

# -----------------------------
# 2) K-means (from scratch, NumPy only)
# -----------------------------
def kmeans(X, k=3, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()

    for _ in range(iters):
        # assign: nearest center
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n, k)
        labels = d2.argmin(axis=1)

        # update: mean of assigned points (handle empty clusters)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = X[mask].mean(axis=0)
            else:
                new_centers[j] = X[rng.integers(0, len(X))]  # re-seed empty cluster

        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    inertia = ((X - centers[labels]) ** 2).sum()  # sum of squared distances to assigned center
    return labels, centers, inertia

labels, centers, inertia = kmeans(X, k=3, iters=50, seed=0)

# -----------------------------
# 3) Clean plot (clusters + centers)
# -----------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=130)

for j in range(3):
    pts = X[labels == j]
    ax.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.85, edgecolors="none", label=f"Cluster {j}")

ax.scatter(centers[:, 0], centers[:, 1], s=180, marker="X", linewidths=1.6, 
    label="Centers", color="gold", edgecolor="#333")

ax.set_title("Clustering with K-means (Unsupervised)", pad=10)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

ax.grid(True, alpha=0.25, linewidth=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True,
          loc="lower left",
          bbox_to_anchor=(0.82, 0.02),
          edgecolor="black",
          facecolor="white")
ax.text(0.82, 0.95,
        rf"$k=3$" + "\n" + rf"inertia = {inertia:.1f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white",
                  edgecolor="black",
                  linewidth=0.8))

plt.tight_layout()
plt.show()
