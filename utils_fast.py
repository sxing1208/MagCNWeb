"""
utils_fast.py  –  vectorised + process-pool helpers
Now safe on Windows: pool is created lazily (no fork-bomb) and
np.tensordot uses explicit axes tuple for NumPy ≤ 1.25.
"""

from __future__ import annotations
import os, multiprocessing as _mp
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.ndimage import generic_filter

# ------------------------------------------------------------------
_mp.set_start_method("spawn", force=True)        # Windows-safe
_POOL: ProcessPoolExecutor | None = None         # lazy created


def _worker_init():
    import cv2  # heavy import only once per worker


def _get_pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=_worker_init,
        )
    return _POOL


# ---------- grid cache ----------
@lru_cache(maxsize=None)
def _xy_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    return x, y


def gaussian_2d(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    x, y = _xy_grid(h, w)
    return np.exp(-(((x - cx) ** 2) + ((y - cy) ** 2)) / (2.0 * sigma**2))


# ---------- loss ----------
def _combined_gaussians(params: NDArray[np.floating], u_obs: np.ndarray) -> float:
    cx = params[:8].reshape(4, 2)
    alpha = params[8:12]
    sigma = params[12:16]

    h, w = u_obs.shape
    x, y = _xy_grid(h, w)

    g = np.exp(
        -(((x - cx[:, 0, None, None]) ** 2) + ((y - cx[:, 1, None, None]) ** 2))
        / (2.0 * sigma[:, None, None] ** 2)
    )  # (4,h,w)

    u_fit = np.tensordot(alpha, g, axes=([0], [0]))  # explicit tuple
    diff = u_fit - u_obs
    return float(np.dot(diff.ravel(), diff.ravel()))


# ---------- peak picker ----------
def k_gauss_peaks(img: NDArray[np.floating], k: int = 4, sigma: float = 5.0):
    h, w = img.shape
    x, y = _xy_grid(h, w)
    work = img.copy()
    peaks = np.empty((k, 2), dtype=np.intp)

    for i in range(k):
        idx = int(work.argmax())
        cy, cx = divmod(idx, w)
        peaks[i] = (cx, cy)
        work *= 1.0 - np.exp(
            -(((x - cx) ** 2) + ((y - cy) ** 2)) / (2.0 * sigma**2)
        )
    return peaks  # (k,2)


# ---------- per-frame optimiser ----------
def _optimise_single(frame: np.ndarray, upsample: int = 20):
    import cv2

    u = cv2.resize(frame, (upsample, upsample), interpolation=cv2.INTER_CUBIC).astype(
        np.float32
    )
    u /= u.max() + 1e-9

    pts = k_gauss_peaks(u, 4, 5.0)
    guess = np.concatenate([pts.ravel(), np.ones(4, np.float32), 0.25 * np.ones(4)])

    h, w = u.shape
    bounds = [(0, w - 0.1), (0, h - 0.1)] * 4 + [(0, None)] * 8

    res = minimize(
        _combined_gaussians,
        guess,
        args=(u,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 150, "ftol": 1e-6},
    )

    pos = res.x[:8].reshape(4, 2).astype(np.float32)
    height = np.sqrt(1.0 / res.x[8:12]).astype(np.float32)
    return pos, height


# ---------- public api ----------
def batch_prediction(frames: np.ndarray) -> np.ndarray:
    pool = _get_pool()
    results = list(pool.map(_optimise_single, frames))

    n = len(frames)
    pos = np.empty((4, n, 2), dtype=np.float32)
    h_val = np.empty((4, n), dtype=np.float32)

    for i, (p, h) in enumerate(results):
        pos[:, i, :] = p
        h_val[:, i] = h

    return np.concatenate([pos, h_val[:, :, None]], axis=-1)


def get_volume(predicted: np.ndarray) -> np.ndarray:
    xy = predicted[:, :, :2]
    dia = 0.5 * (
        (xy[..., 0].max(0) - xy[..., 0].min(0))
        + (xy[..., 1].max(0) - xy[..., 1].min(0))
    )
    return (dia / dia.min()) ** 3 - 1


# ---------- nan helper ----------
def _nanmean(vals):
    v = vals[~np.isnan(vals)]
    return np.mean(v) if v.size else np.nan


def replace_nan_with_neighbors(arr: np.ndarray):
    from scipy.ndimage import generic_filter

    filled = generic_filter(arr, _nanmean, size=3, mode="constant", cval=np.nan)
    arr[np.isnan(arr)] = filled[np.isnan(arr)]
    return arr


__all__ = ["gaussian_2d", "batch_prediction", "get_volume", "replace_nan_with_neighbors"]
