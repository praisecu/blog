import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------
# 1) Toy supervised dataset (2 classes, 2 features)
# -----------------------------
rng = np.random.default_rng(0)
n = 160

X0 = rng.normal(loc=[-1.6, -1.0], scale=0.7, size=(n // 2, 2))
X1 = rng.normal(loc=[+1.6, +1.0], scale=0.7, size=(n // 2, 2))
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(int)

df = pd.DataFrame(X, columns=["x1", "x2"])
df["y"] = y

# -----------------------------
# 2) Logistic regression trained with SciPy (BFGS)
# -----------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

Xmat = df[["x1", "x2"]].to_numpy()
yvec = df["y"].to_numpy()
Xb = np.c_[np.ones(len(Xmat)), Xmat]  # add bias term

def loss_and_grad(w):
    z = Xb @ w
    # Stable negative log-likelihood: sum(log(1+exp(z)) - y*z)
    loss = np.sum(np.logaddexp(0.0, z) - yvec * z)
    p = sigmoid(z)
    grad = Xb.T @ (p - yvec)
    return loss, grad

w0 = np.zeros(Xb.shape[1])
res = minimize(lambda w: loss_and_grad(w)[0], w0, jac=lambda w: loss_and_grad(w)[1], method="BFGS")
w = res.x

proba = sigmoid(Xb @ w)
yhat = (proba >= 0.5).astype(int)
acc = (yhat == yvec).mean()

# -----------------------------
# 3) Clean plot (points + decision boundary)
# -----------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=130)

ax.scatter(df.loc[df.y == 0, "x1"], df.loc[df.y == 0, "x2"], s=28, alpha=0.85,
           edgecolors="none", label="Class 0")
ax.scatter(df.loc[df.y == 1, "x1"], df.loc[df.y == 1, "x2"], s=28, alpha=0.85,
           edgecolors="none", label="Class 1", color="#380")

# Decision boundary: w0 + w1*x1 + w2*x2 = 0
x1_line = np.linspace(df["x1"].min() - 0.8, df["x1"].max() + 0.8, 250)
if abs(w[2]) > 1e-12:
    x2_line = -(w[0] + w[1] * x1_line) / w[2]
    ax.plot(x1_line, x2_line, linewidth=3, label="Decision boundary", color="goldenrod")
else:
    ax.axvline(-w[0] / w[1], linewidth=3, label="Decision boundary")

ax.set_title("Supervised Classification with Logistic Regression", pad=10)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

ax.grid(True, alpha=0.25, linewidth=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02))

ax.text(0.62, 0.95,
        rf"$\hat{{p}}(y=1|x)=\sigma(w_0+w_1x_1+w_2x_2)$" + "\n" +
        rf"$w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}],\ \mathrm{{acc}}={acc:.3f}$",
        transform=ax.transAxes, va="top")

plt.tight_layout()
plt.show()
