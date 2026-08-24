#!/usr/bin/env python3
"""Stitch two overlapping photos into a panorama, from scratch.

Companion code for the "Stitch Happens" tutorial:
Harris corners -> ANMS -> patch descriptors -> ratio-test matching
-> RANSAC + DLT homography -> inverse warp -> feather blend.

Usage:  python3 stitch.py left.jpg right.jpg panorama.jpg
Needs:  pip install numpy opencv-python
"""
import sys
import numpy as np
import cv2


def harris_corners(gray):
    """All local maxima of the Harris response, strongest first."""
    R = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    is_peak = (R == cv2.dilate(R, None)) & (R > 1e-4 * R.max())
    ys, xs = np.nonzero(is_peak)
    order = np.argsort(-R[ys, xs])
    return np.column_stack([xs, ys])[order].astype(np.float64), R[ys, xs][order]


def anms(pts, scores, num_keep=500, c_robust=0.9, cap=1200):
    """Adaptive Non-Maximal Suppression: keep strong AND well-spread corners."""
    pts, scores = pts[:cap], scores[:cap]
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    stronger = scores[None, :] * c_robust > scores[:, None]
    radius = np.sqrt(np.where(stronger, d2, np.inf).min(axis=1))
    radius[0] = np.inf
    return pts[np.argsort(-radius)[:num_keep]]


def make_descriptors(gray, pts, patch=40, out=8):
    """Blurred, subsampled, normalized patches: simple exposure-proof fingerprints."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.5)
    h, w = gray.shape
    descs, kept = [], []
    for x, y in pts.astype(int):
        if not (patch // 2 <= x < w - patch // 2 and patch // 2 <= y < h - patch // 2):
            continue
        p = blurred[y - patch // 2:y + patch // 2, x - patch // 2:x + patch // 2]
        p = cv2.resize(p, (out, out), interpolation=cv2.INTER_AREA).ravel()
        descs.append((p - p.mean()) / (p.std() + 1e-8))
        kept.append((x, y))
    return np.float32(descs), np.float64(kept)


def match(D1, D2, kp1, kp2, ratio_thr=0.8):
    """Nearest-neighbour matching with Lowe's ratio test."""
    d2 = (D1 ** 2).sum(1)[:, None] + (D2 ** 2).sum(1)[None, :] - 2 * D1 @ D2.T
    nn = np.argsort(d2, axis=1)[:, :2]
    idx = np.arange(len(D1))
    ratio = np.sqrt(np.maximum(d2[idx, nn[:, 0]], 0) /
                    np.maximum(d2[idx, nn[:, 1]], 1e-12))
    keep = ratio < ratio_thr
    return kp1[keep], kp2[nn[keep, 0]]


def apply_h(H, pts):
    """Apply homography H to Nx2 points (homogeneous divide included)."""
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def dlt_homography(src, dst):
    """Direct Linear Transform with Hartley normalization: dst ~ H @ src."""
    def norm_T(p):
        m = p.mean(0)
        s = np.sqrt(2) / (np.linalg.norm(p - m, axis=1).mean() + 1e-12)
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    Ts, Td = norm_T(src), norm_T(dst)
    s = apply_h(Ts, src)
    d = apply_h(Td, dst)
    A = []
    for (x, y), (u, v) in zip(s, d):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(A))
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def ransac_homography(src, dst, iters=600, thr=4.0, rng=None):
    """Robustly fit dst ~ H @ src by random 4-point consensus voting."""
    rng = rng or np.random.default_rng(7)
    best_mask = None
    for _ in range(iters):
        pick = rng.choice(len(src), 4, replace=False)
        try:
            H = dlt_homography(src[pick], dst[pick])
        except np.linalg.LinAlgError:
            continue
        # a degenerate sample (e.g. 3 collinear points) can send points to
        # infinity — ignore the numpy warnings and drop non-finite errors
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.linalg.norm(apply_h(H, src) - dst, axis=1)
        mask = np.isfinite(err) & (err < thr)
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask
    return dlt_homography(src[best_mask], dst[best_mask]), best_mask


def stitch(im1, im2):
    """Warp im2 into im1's frame with a RANSAC homography, feather-blend both."""
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    D1, kp1 = make_descriptors(g1, anms(*harris_corners(g1)))
    D2, kp2 = make_descriptors(g2, anms(*harris_corners(g2)))
    p1, p2 = match(D1, D2, kp1, kp2)
    print(f"matches: {len(p1)}")
    H, inliers = ransac_homography(p2, p1)
    print(f"RANSAC inliers: {inliers.sum()} / {len(p1)}")

    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    corners2 = apply_h(H, np.float64([[0, 0], [w2, 0], [w2, h2], [0, h2]]))
    allc = np.vstack([corners2, [[0, 0], [w1, 0], [w1, h1], [0, h1]]])
    x0, y0 = np.floor(allc.min(0)).astype(int)
    x1, y1 = np.ceil(allc.max(0)).astype(int)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])

    size = (x1 - x0, y1 - y0)
    warped2 = cv2.warpPerspective(im2, T @ H, size)
    mask2 = cv2.warpPerspective(np.full((h2, w2), 255, np.uint8), T @ H, size)
    canvas1 = np.zeros_like(warped2)
    canvas1[-y0:-y0 + h1, -x0:-x0 + w1] = im1
    mask1 = np.zeros(size[::-1], np.uint8)
    mask1[-y0:-y0 + h1, -x0:-x0 + w1] = 255

    w1f = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    w2f = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    total = np.maximum(w1f + w2f, 1e-6)
    pano = (canvas1 * (w1f / total)[..., None] +
            warped2 * (w2f / total)[..., None])
    return np.clip(pano, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    a, b = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
    assert a is not None and b is not None, "could not read the input images"
    cv2.imwrite(sys.argv[3], stitch(a, b))
    print(f"wrote {sys.argv[3]} — stitch happened.")
