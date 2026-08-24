#!/usr/bin/env python3
"""Regenerate all figures + widget data for the panorama-stitching tutorial.

Run from anywhere:  python3 public/code/gen_pano.py
Needs:              pip install numpy opencv-python gdown

Downloads the input photographs on first run (three sequences from the
course dataset on Google Drive; the tutorial uses images 3 and 4 of the
Flatirons sequence, University of Colorado Boulder campus), then runs the
pipeline:

  Harris corners -> ANMS -> patch descriptors -> ratio-test matching
  -> RANSAC homography -> warp -> feather blend

Outputs (repo-relative):
  public/images/pano/*.jpg    step-by-step figures
  src/data/pano-data.json     data consumed by the interactive widgets
Deterministic: fixed RNG seed, so figures and widget data stay in sync.
"""
import json, os
import numpy as np
import cv2

RNG = np.random.default_rng(7)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(REPO, "public/images/pano")
DATA_PATH = os.path.join(REPO, "src/data/pano-data.json")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# Input photos come from the course dataset (Google Drive).  The tutorial
# itself uses images 3 and 4 of the Flatirons sequence; the closing section
# runs the same pipeline on the CULogo and CUBoulderSatView sequences.
SRC_DIR = os.path.join(IMG_DIR, "_src")
DATASETS = {
    "Flatirons":        "1cwLOG_4keUaQHmp5RVwe_ZiAHeULpjOd",
    "CULogo":           "1Pv7eI-LOOUZxHffz1D5pmMKOk-Y7C-kx",
    "CUBoulderSatView": "10WcuZFuCxbe1FQSwpdy_9piyMM45eOI_",
}
missing = [fid for name, fid in DATASETS.items()
           if not os.path.isdir(os.path.join(SRC_DIR, name))]
if missing:
    print("downloading input photos (course dataset, Google Drive)...")
    import gdown  # pip install gdown
    os.makedirs(SRC_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(SRC_DIR)
    try:
        for fid in missing:
            gdown.download_folder(id=fid, quiet=True, use_cookies=False)
    finally:
        os.chdir(cwd)
LEFT_SRC = os.path.join(SRC_DIR, "Flatirons", "3.jpg")
RIGHT_SRC = os.path.join(SRC_DIR, "Flatirons", "4.jpg")

JQ = [int(cv2.IMWRITE_JPEG_QUALITY), 84]

def save(name, img, q=JQ):
    path = os.path.join(IMG_DIR, name)
    ok = cv2.imwrite(path, img, q)
    assert ok, name
    print(f"  wrote {name}  {img.shape[1]}x{img.shape[0]}  {os.path.getsize(path)//1024} KB")

# ----------------------------------------------------------------------------
# 0. Build the two views
# ----------------------------------------------------------------------------
print("== building views ==")
VIEW_H = 800               # working height; width follows the aspect ratio

def load_view(path):
    im = cv2.imread(path)
    assert im is not None, path
    h, w = im.shape[:2]
    return cv2.resize(im, (int(round(w * VIEW_H / h)), VIEW_H),
                      interpolation=cv2.INTER_AREA)

im1 = load_view(LEFT_SRC)      # left view  (photo 3)
im2 = load_view(RIGHT_SRC)     # right view (photo 4)
assert im1.shape == im2.shape, "expected same-size photos"
VH, VW = im1.shape[:2]
corners_out = np.array([[0, 0], [VW, 0], [VW, VH], [0, VH]], np.float64)

save("left.jpg", im1)
save("right.jpg", im2)

g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.float32)

# --- fig: inputs side by side -----------------------------------------------
def label(img, text):
    out = img.copy()
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.rectangle(out, (14, 14), (14 + tw + 20, 14 + th + bl + 14), (30, 30, 30), -1)
    cv2.putText(out, text, (24, 14 + th + 7), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out

gap = np.full((VH, 16, 3), 255, np.uint8)
save("01-inputs.jpg", np.hstack([label(im1, "LEFT photo"), gap, label(im2, "RIGHT photo")]))

# (the naive slide-and-average figure is generated after matching, where the
#  best pure translation can be estimated from the verified matches)

# ----------------------------------------------------------------------------
# 1. Harris corners + local maxima
# ----------------------------------------------------------------------------
print("== harris ==")
def harris_corners(gray):
    R = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    Rd = cv2.dilate(R, None)               # 3x3 neighbourhood max
    is_max = (R == Rd) & (R > 1e-4 * R.max())
    ys, xs = np.nonzero(is_max)
    scores = R[ys, xs]
    order = np.argsort(-scores)
    return R, np.column_stack([xs[order], ys[order]]).astype(np.float64), scores[order]

R1, pts1, sc1 = harris_corners(g1)
R2, pts2, sc2 = harris_corners(g2)
print(f"  local maxima: left {len(pts1)}, right {len(pts2)}")

# --- fig: response heatmap ---------------------------------------------------
Rlog = np.log10(np.maximum(R1, 1e-9 * R1.max()))
Rn = (Rlog - Rlog.min()) / (Rlog.max() - Rlog.min())
heat = cv2.applyColorMap((Rn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
grayb = cv2.cvtColor(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
save("03-harris-heat.jpg", cv2.addWeighted(heat, 0.72, grayb, 0.38, 0))

# --- fig: thresholded corner dots -------------------------------------------
def draw_pts(img, pts, color=(60, 220, 60), r=3):
    out = img.copy()
    for x, y in pts.astype(int):
        cv2.circle(out, (x, y), r, color, -1, cv2.LINE_AA)
    return out

strong1 = pts1[sc1 > 0.01 * sc1[0]]
save("04-corners.jpg", label(draw_pts(im1, strong1), f"{len(strong1)} corners"))

# ----------------------------------------------------------------------------
# 2. ANMS
# ----------------------------------------------------------------------------
print("== anms ==")
def anms(pts, scores, cap=1200, c_robust=0.9):
    p = pts[:cap]; s = scores[:cap]
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    stronger = s[None, :] * c_robust > s[:, None]     # j meaningfully stronger than i
    d2 = np.where(stronger, d2, np.inf)
    radii = np.sqrt(d2.min(axis=1))
    radii[0] = np.inf                                  # strongest corner: infinite radius
    order = np.argsort(-radii)
    return order, radii

ord1, rad1 = anms(pts1, sc1)
ord2, rad2 = anms(pts2, sc2)

N_KEEP = 500
keep1 = pts1[ord1[:N_KEEP]]
keep2 = pts2[ord2[:N_KEEP]]

# --- fig: top-N by score vs ANMS --------------------------------------------
top300 = pts1[:300]
anms300 = pts1[ord1[:300]]
panelA = label(draw_pts(im1, top300, (80, 80, 245)), "Top 300 by strength")
panelB = label(draw_pts(im1, anms300, (60, 220, 60)), "300 via ANMS")
save("05-anms.jpg", np.hstack([panelA, gap, panelB]))

# ----------------------------------------------------------------------------
# 3. Descriptors
# ----------------------------------------------------------------------------
print("== descriptors ==")
PATCH, DESC = 40, 8
def make_descriptors(gray, pts):
    blur = cv2.GaussianBlur(gray, (0, 0), 2.5)
    h, w = gray.shape
    descs, kept = [], []
    for x, y in pts.astype(int):
        if x < PATCH // 2 or y < PATCH // 2 or x >= w - PATCH // 2 or y >= h - PATCH // 2:
            continue
        patch = blur[y - PATCH // 2:y + PATCH // 2, x - PATCH // 2:x + PATCH // 2]
        small = cv2.resize(patch, (DESC, DESC), interpolation=cv2.INTER_AREA).ravel()
        small = (small - small.mean()) / (small.std() + 1e-8)
        descs.append(small); kept.append((x, y))
    return np.array(descs, np.float32), np.array(kept, np.float64)

D1, kp1 = make_descriptors(g1, keep1)
D2, kp2 = make_descriptors(g2, keep2)
print(f"  descriptors: left {len(D1)}, right {len(D2)}")

# --- fig: patch -> descriptor for 3 sample corners ---------------------------
blur1 = cv2.GaussianBlur(g1, (0, 0), 2.5)
# pick 3 well-separated, photogenic corners from the ANMS list
samples = [kp1[i] for i in (0, 12, 40)]
tiles = []
for x, y in np.array(samples).astype(int):
    patch = blur1[y - 20:y + 20, x - 20:x + 20]
    small = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
    norm = (small - small.mean()) / (small.std() + 1e-8)
    vis_p = cv2.resize(patch, (160, 160), interpolation=cv2.INTER_NEAREST)
    vis_p = cv2.normalize(vis_p, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_d = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_d = cv2.resize(vis_d, (160, 160), interpolation=cv2.INTER_NEAREST)
    ctx = im1[max(y - 80, 0):y + 80, max(x - 80, 0):x + 80].copy()
    ctx = cv2.resize(ctx, (160, 160), interpolation=cv2.INTER_AREA)
    cv2.rectangle(ctx, (60, 60), (100, 100), (60, 220, 60), 2)
    row = np.hstack([ctx,
                     np.full((160, 12, 3), 255, np.uint8),
                     cv2.cvtColor(vis_p, cv2.COLOR_GRAY2BGR),
                     np.full((160, 12, 3), 255, np.uint8),
                     cv2.cvtColor(vis_d, cv2.COLOR_GRAY2BGR)])
    tiles.append(row)
    tiles.append(np.full((12, row.shape[1], 3), 255, np.uint8))
fig = np.vstack(tiles[:-1])
hdr = np.full((36, fig.shape[1], 3), 255, np.uint8)
for txt, xpos in [("corner (zoomed out)", 6), ("40x40 patch", 178), ("8x8 descriptor", 350)]:
    cv2.putText(hdr, txt, (xpos, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
save("06-descriptors.jpg", np.vstack([hdr, fig]))

# ----------------------------------------------------------------------------
# 4. Matching (SSD + ratio test)
# ----------------------------------------------------------------------------
print("== matching ==")
d2mat = ((D1 ** 2).sum(1)[:, None] + (D2 ** 2).sum(1)[None, :] - 2 * D1 @ D2.T)
nn = np.argsort(d2mat, axis=1)[:, :2]
best = d2mat[np.arange(len(D1)), nn[:, 0]]
second = d2mat[np.arange(len(D1)), nn[:, 1]]
ratio = np.sqrt(np.maximum(best, 0) / np.maximum(second, 1e-12))

LOOSE = 0.93
loose_idx = np.nonzero(ratio < LOOSE)[0]
m_p1 = kp1[loose_idx]
m_p2 = kp2[nn[loose_idx, 0]]
m_ratio = ratio[loose_idx]
strict_mask = m_ratio < 0.80
print(f"  loose matches (<{LOOSE}): {len(loose_idx)}, strict (<0.80): {strict_mask.sum()}")

def draw_matches(imA, imB, pA, pB, colors, r=3, thick=1):
    canvas = np.hstack([imA, imB])
    off = imA.shape[1]
    for (xa, ya), (xb, yb), col in zip(pA.astype(int), pB.astype(int), colors):
        cv2.circle(canvas, (xa, ya), r, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (xb + off, yb), r, col, -1, cv2.LINE_AA)
        cv2.line(canvas, (xa, ya), (xb + off, yb), col, thick, cv2.LINE_AA)
    return canvas

sp1, sp2 = m_p1[strict_mask], m_p2[strict_mask]
cols = [(0, 200, 255)] * len(sp1)
save("07-matches.jpg", draw_matches(im1, im2, sp1, sp2, cols))

# ----------------------------------------------------------------------------
# 5. RANSAC homography (DLT, from scratch)
# ----------------------------------------------------------------------------
print("== ransac ==")
def dlt_homography(src, dst):
    """H such that dst ~ H @ src (Nx2 arrays, N>=4), with normalization."""
    def norm_T(p):
        m = p.mean(0); s = np.sqrt(2) / (np.linalg.norm(p - m, axis=1).mean() + 1e-12)
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    Ts, Td = norm_T(src), norm_T(dst)
    s = (Ts @ np.column_stack([src, np.ones(len(src))]).T).T
    d = (Td @ np.column_stack([dst, np.ones(len(dst))]).T).T
    A = []
    for (x, y, _), (u, v, _) in zip(s, d):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]

def project(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]

src = m_p2[strict_mask]      # right-view points
dst = m_p1[strict_mask]      # left-view points
NMATCH = len(src)
THR = 4.0
best_count, best_mask, best_H = -1, None, None
iters_log = []
for it in range(600):
    pick = RNG.choice(NMATCH, 4, replace=False)
    try:
        H = dlt_homography(src[pick], dst[pick])
    except np.linalg.LinAlgError:
        continue
    err = np.linalg.norm(project(H, src) - dst, axis=1)
    mask = err < THR
    cnt = int(mask.sum())
    if it < 48:
        iters_log.append({"pick": pick.tolist(), "count": cnt,
                          "mask": "".join("1" if m else "0" for m in mask)})
    if cnt > best_count:
        best_count, best_mask, best_H = cnt, mask, H
H_final = dlt_homography(src[best_mask], dst[best_mask])
err = np.linalg.norm(project(H_final, src) - dst, axis=1)
inl = err < THR
print(f"  matches {NMATCH}, inliers {int(inl.sum())} ({inl.mean():.0%})")

# sanity: how tightly does the final model fit its inliers?
print(f"  inlier reprojection error: mean {err[inl].mean():.2f} px, "
      f"median {np.median(err[inl]):.2f} px, max {err[inl].max():.2f} px")

# inlier flag for ALL loose matches (for the widgets)
err_all = np.linalg.norm(project(H_final, m_p2) - m_p1, axis=1)
inlier_all = err_all < THR

# --- fig: the naive attempt (best pure translation + average) ----------------
# The best a "just slide it over" plan can possibly do: translate the right
# view by the median displacement of the verified matches, then average.
print("== naive overlay ==")
dx, dy = np.median(m_p1[inlier_all] - m_p2[inlier_all], axis=0).round().astype(int)
gx, gy = max(0, -dx), max(0, -dy)
naive = np.zeros((VH + abs(dy), VW + abs(dx), 3), np.float32)
wgt = np.zeros(naive.shape[:2] + (1,), np.float32)
naive[gy:gy + VH, gx:gx + VW] += im1
wgt[gy:gy + VH, gx:gx + VW] += 1
naive[gy + dy:gy + dy + VH, gx + dx:gx + dx + VW] += im2
wgt[gy + dy:gy + dy + VH, gx + dx:gx + dx + VW] += 1
naive = (naive / np.maximum(wgt, 1)).astype(np.uint8)
save("02-naive.jpg", naive)

cols = [(70, 210, 70) if i else (70, 70, 235) for i in inl]
save("08-ransac.jpg", draw_matches(im1, im2, src, dst, cols))

# ----------------------------------------------------------------------------
# 6. Warp + composite
# ----------------------------------------------------------------------------
print("== warp & blend ==")
corners2 = project(H_final, corners_out)
allc = np.vstack([corners2, [[0, 0], [VW, 0], [VW, VH], [0, VH]]])
x0, y0 = np.floor(allc.min(0)).astype(int)
x1, y1 = np.ceil(allc.max(0)).astype(int)
T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
CW, CH = x1 - x0, y1 - y0
warped2 = cv2.warpPerspective(im2, T @ H_final, (CW, CH))
mask2 = cv2.warpPerspective(np.full((VH, VW), 255, np.uint8), T @ H_final, (CW, CH))
canvas1 = np.zeros_like(warped2); canvas1[-y0:-y0 + VH, -x0:-x0 + VW] = im1
mask1 = np.zeros((CH, CW), np.uint8); mask1[-y0:-y0 + VH, -x0:-x0 + VW] = 255

# hard paste (right on top) -> visible seam thanks to exposure difference
hard = canvas1.copy()
hard[mask2 > 0] = warped2[mask2 > 0]
outline = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
hard_marked = hard.copy(); hard_marked[outline > 0] = (60, 60, 235)
save("09-warped.jpg", hard_marked)

# feather blend
w1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
w2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
wsum = w1 + w2
w1n = np.where(wsum > 0, w1 / np.maximum(wsum, 1e-6), 0)[..., None]
w2n = np.where(wsum > 0, w2 / np.maximum(wsum, 1e-6), 0)[..., None]
blend = (canvas1.astype(np.float32) * w1n + warped2.astype(np.float32) * w2n)
blend = np.clip(blend, 0, 255).astype(np.uint8)

# crop to the largest axis-aligned rectangle fully covered by content
# (trims the black wedges left by the warp)
def largest_interior_rect(mask):
    """(r0, c0, r1, c1) of the largest all-True rectangle, histogram method."""
    h, w = mask.shape
    heights = np.zeros(w + 1, np.int32)
    best = (0, 0, 0, 0, 0)                       # area, r0, c0, r1, c1
    for r in range(h):
        heights[:w] = np.where(mask[r], heights[:w] + 1, 0)
        stack = []                               # (start_col, height)
        for c in range(w + 1):
            start = c
            while stack and stack[-1][1] >= heights[c]:
                s, sh = stack.pop()
                if sh * (c - s) > best[0]:
                    best = (sh * (c - s), r - sh + 1, s, r + 1, c)
                start = s
            stack.append((start, heights[c]))
    return best[1:]

r0, c0, r1, c1 = largest_interior_rect(wsum > 0)
crop = (slice(r0, r1), slice(c0, c1))
final = blend[crop]
hardc = hard[crop]
save("10-final.jpg", final)
save("hard-crop.jpg", hardc)

# seam close-up: hard vs feathered around the seam line (left image's right
# border), clamped to the canvas
seam_x = int(np.clip(-x0 + VW - 40, 220, CW - 220))
seam_y = int(np.clip(CH // 2 - 200, 0, max(CH - 400, 0)))
sl = (slice(seam_y, seam_y + 400), slice(seam_x - 220, seam_x + 220))
z = np.hstack([label(cv2.resize(hard[sl], None, fx=1.6, fy=1.6), "hard cut"),
               np.full((640, 16, 3), 255, np.uint8),
               label(cv2.resize(blend[sl], None, fx=1.6, fy=1.6), "feathered")])
save("11-seam.jpg", z)

# ----------------------------------------------------------------------------
# 7. Widget data
# ----------------------------------------------------------------------------
print("== widget data ==")
CAP = 2600
wpts = pts1[:CAP]; wsc = sc1[:CAP]
logs = np.log10(np.maximum(wsc / wsc[0], 1e-7))       # 0 .. -7
data = {
    "vw": VW, "vh": VH,
    "corners": [[int(x), int(y), round(float(l), 2)] for (x, y), l in zip(wpts, logs)],
    "anmsOrder": [int(i) for i in ord1[:800]],
    "matches": {
        "p1": m_p1.astype(int).tolist(),
        "p2": m_p2.astype(int).tolist(),
        "ratio": [round(float(r), 3) for r in m_ratio],
        "inlier": [int(i) for i in inlier_all],
    },
    "strictIdx": [int(i) for i in np.nonzero(strict_mask)[0]],
    "ransac": {"thr": THR, "iters": iters_log, "bestPossible": int(inl.sum())},
    "H": [[round(float(v), 6) for v in row] for row in H_final],
}
with open(DATA_PATH, "w") as f:
    json.dump(data, f, separators=(",", ":"))
print(f"  pano-data.json {os.path.getsize(DATA_PATH)//1024} KB")

# cover image for the post listing (public/images/panorama.jpg)
cover = cv2.resize(final, (1200, int(final.shape[0] * 1200 / final.shape[1])),
                   interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(REPO, "public/images/panorama.jpg"), cover,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85])
print("  panorama.jpg (cover)")

# ----------------------------------------------------------------------------
# 8. The same pipeline on the other sequences (closing-section figures)
# ----------------------------------------------------------------------------
def pair_homography(imA, imB, iters=20000, ratio=0.85, n_keep=800):
    """H mapping imB coords -> imA coords, using the full pipeline above.

    Slightly looser settings than the tutorial pair: repetitive texture
    (e.g. stone walls) drives ratios up, so we admit more candidates and
    let RANSAC sort them out."""
    gA = cv2.cvtColor(imA, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gB = cv2.cvtColor(imB, cv2.COLOR_BGR2GRAY).astype(np.float32)
    _, pA, sA = harris_corners(gA)
    _, pB, sB = harris_corners(gB)
    oA, _ = anms(pA, sA, cap=2000)
    oB, _ = anms(pB, sB, cap=2000)
    DA, kA = make_descriptors(gA, pA[oA[:n_keep]])
    DB, kB = make_descriptors(gB, pB[oB[:n_keep]])
    dd = (DA ** 2).sum(1)[:, None] + (DB ** 2).sum(1)[None, :] - 2 * DA @ DB.T
    nn2 = np.argsort(dd, axis=1)[:, :2]
    b = dd[np.arange(len(DA)), nn2[:, 0]]
    s2 = dd[np.arange(len(DA)), nn2[:, 1]]
    rat = np.sqrt(np.maximum(b, 0) / np.maximum(s2, 1e-12))
    keep = rat < ratio
    srcB, dstA = kB[nn2[keep, 0]], kA[keep]
    bestc, bestm = -1, None
    for _ in range(iters):
        pick = RNG.choice(len(srcB), 4, replace=False)
        try:
            Hh = dlt_homography(srcB[pick], dstA[pick])
        except np.linalg.LinAlgError:
            continue
        e = np.linalg.norm(project(Hh, srcB) - dstA, axis=1)
        m = e < THR
        if m.sum() > bestc:
            bestc, bestm = int(m.sum()), m
    print(f"    pair: {len(srcB)} matches, {bestc} inliers")
    return dlt_homography(srcB[bestm], dstA[bestm])

def stitch_sequence(name, slug, use=None):
    """Stitch the images of a dataset folder into one feathered panorama.

    use: optional list of filenames to restrict to (e.g. when one photo of a
    sequence overlaps its neighbour too little for the fixed-patch
    descriptors, as CULogo photo 3 does at ~40 px)."""
    print(f"== bonus: {name} ==")
    folder = os.path.join(SRC_DIR, name)
    files = use or sorted(f for f in os.listdir(folder)
                          if f.split(".")[0].isdigit() and f.lower().endswith(".jpg"))
    ims = [cv2.imread(os.path.join(folder, f)) for f in files]
    assert all(im is not None for im in ims), files
    ref = (len(ims) - 1) // 2                     # middle image = reference

    # chain pairwise homographies to the reference frame
    Hs = {ref: np.eye(3)}
    for i in range(ref + 1, len(ims)):
        Hs[i] = Hs[i - 1] @ pair_homography(ims[i - 1], ims[i])
    for i in range(ref - 1, -1, -1):
        Hs[i] = Hs[i + 1] @ pair_homography(ims[i + 1], ims[i])

    # common canvas
    allc = np.vstack([project(Hs[i], np.array(
        [[0, 0], [im.shape[1], 0], [im.shape[1], im.shape[0]], [0, im.shape[0]]],
        np.float64)) for i, im in enumerate(ims)])
    bx0, by0 = np.floor(allc.min(0)).astype(int)
    bx1, by1 = np.ceil(allc.max(0)).astype(int)
    Tb = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], np.float64)
    size = (bx1 - bx0, by1 - by0)

    # warp everything and feather-blend
    acc = np.zeros((size[1], size[0], 3), np.float32)
    wacc = np.zeros((size[1], size[0], 1), np.float32)
    for i, im in enumerate(ims):
        warped = cv2.warpPerspective(im, Tb @ Hs[i], size)
        m = cv2.warpPerspective(np.full(im.shape[:2], 255, np.uint8), Tb @ Hs[i], size)
        wgt = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5)[..., None]
        acc += warped.astype(np.float32) * wgt
        wacc += wgt
    pano = np.clip(acc / np.maximum(wacc, 1e-6), 0, 255).astype(np.uint8)

    r0, c0, r1, c1 = largest_interior_rect(wacc[..., 0] > 0)
    pano = pano[r0:r1, c0:c1]

    gap_v = np.full((ims[0].shape[0], 12, 3), 255, np.uint8)
    row = []
    for im in ims:
        row += [im, gap_v]
    save(f"bonus-{slug}-inputs.jpg", np.hstack(row[:-1]))
    save(f"bonus-{slug}-result.jpg", pano)

stitch_sequence("CULogo", "culogo", use=["1.jpg", "2.jpg"])
stitch_sequence("CUBoulderSatView", "satview")
print("DONE")
