"""Core DIC engine: normalized cross-correlation (NCC) subset tracking.

Uses multiprocessing to distribute subset rows across CPU cores.
"""

import os

import cv2
import numpy as np
from multiprocessing import Pool
from scipy.signal import fftconvolve


def _norm_cross_corr(template, search_area):
    """Compute NCC between a template and a (larger) search area.

    Returns
    -------
    ncc_map : 2-D ndarray
        Correlation coefficient at each valid lag position.
    """
    template = template.astype(np.float64)
    search_area = search_area.astype(np.float64)

    t_mean = template.mean()
    t_std = template.std()
    if t_std < 1e-12:
        return np.zeros(
            (search_area.shape[0] - template.shape[0] + 1,
             search_area.shape[1] - template.shape[1] + 1)
        )

    t_norm = template - t_mean
    th, tw = template.shape

    # Local sums via convolution for efficiency.
    ones = np.ones_like(template)
    local_sum = fftconvolve(search_area, ones[::-1, ::-1], mode="valid")
    local_sum_sq = fftconvolve(search_area ** 2, ones[::-1, ::-1], mode="valid")
    n = th * tw
    local_mean = local_sum / n
    local_var = local_sum_sq / n - local_mean ** 2
    local_std = np.sqrt(np.clip(local_var, 0, None))

    cross = fftconvolve(search_area, t_norm[::-1, ::-1], mode="valid")
    denom = local_std * t_std * n
    denom[denom < 1e-12] = 1e-12
    return cross / denom


# ── multiprocessing helpers ─────────────────────────────────────────────

_w = {}  # populated once per worker by _init_worker


def _init_worker(ref_gray, def_gray, roi_contour, half, search_margin,
                 ncc_threshold, h_img, w_img):
    """Store shared read-only data in each worker process."""
    _w["ref"] = ref_gray
    _w["def"] = def_gray
    _w["roi"] = roi_contour
    _w["half"] = half
    _w["margin"] = search_margin
    _w["thresh"] = ncc_threshold
    _w["h"] = h_img
    _w["w"] = w_img


def _track_chunk(row_range):
    """Track all subsets in a contiguous block of grid rows."""
    iy_start, iy_end, gx_chunk, gy_chunk = row_range
    ref = _w["ref"]
    defm = _w["def"]
    roi_c = _w["roi"]
    half = _w["half"]
    margin = _w["margin"]
    thresh = _w["thresh"]
    h_img = _w["h"]
    w_img = _w["w"]

    n_rows, nx = gx_chunk.shape
    u_out = np.full((n_rows, nx), np.nan)
    v_out = np.full((n_rows, nx), np.nan)
    c_out = np.full((n_rows, nx), np.nan)

    for r in range(n_rows):
        for ix in range(nx):
            cx = int(gx_chunk[r, ix])
            cy = int(gy_chunk[r, ix])

            if roi_c is not None:
                if cv2.pointPolygonTest(roi_c, (float(cx), float(cy)), False) < 0:
                    continue

            t_y0 = cy - half
            t_y1 = cy + half + 1
            t_x0 = cx - half
            t_x1 = cx + half + 1
            if t_y0 < 0 or t_x0 < 0 or t_y1 > h_img or t_x1 > w_img:
                continue
            template = ref[t_y0:t_y1, t_x0:t_x1]

            s_y0 = max(cy - half - margin, 0)
            s_y1 = min(cy + half + 1 + margin, h_img)
            s_x0 = max(cx - half - margin, 0)
            s_x1 = min(cx + half + 1 + margin, w_img)
            search = defm[s_y0:s_y1, s_x0:s_x1]

            if (search.shape[0] < template.shape[0] or
                    search.shape[1] < template.shape[1]):
                continue

            ncc_map = _norm_cross_corr(template, search)
            if ncc_map.size == 0:
                continue

            peak_val = ncc_map.max()
            c_out[r, ix] = peak_val

            if peak_val < thresh:
                continue

            peak_idx = np.unravel_index(ncc_map.argmax(), ncc_map.shape)
            match_y = s_y0 + peak_idx[0]
            match_x = s_x0 + peak_idx[1]
            u_out[r, ix] = (match_x + half) - cx
            v_out[r, ix] = (match_y + half) - cy

    return iy_start, u_out, v_out, c_out


# ── public API ──────────────────────────────────────────────────────────

def track_subsets(ref_gray, def_gray, roi, subset_size, step_size,
                  ncc_threshold, search_margin=None):
    """Track subsets from *ref_gray* to *def_gray* using NCC.

    Distributes grid rows across CPU cores via multiprocessing.

    Parameters
    ----------
    ref_gray, def_gray : 2-D uint8 arrays
    roi : ndarray (N,2) polygon, or None  — analysis region
    subset_size : int (odd, e.g. 29)
    step_size : int (e.g. 10)
    ncc_threshold : float  — minimum acceptable peak NCC
    search_margin : int or None
        Extra pixels around subset centre to search in the deformed image.
        Defaults to ``subset_size``.

    Returns
    -------
    grid_x, grid_y : 2-D arrays of subset-centre pixel coordinates
    u, v : 2-D float arrays of displacements (pixels); NaN where masked
    corr : 2-D float array of peak NCC values
    """
    if search_margin is None:
        search_margin = subset_size

    half = subset_size // 2
    h_img, w_img = ref_gray.shape

    # Determine analysis bounds from ROI polygon (bounding box).
    if roi is not None:
        rx, ry, rw, rh = cv2.boundingRect(roi.reshape(-1, 1, 2))
        roi_contour = roi.reshape(-1, 1, 2).astype(np.float32)
    else:
        rx, ry, rw, rh = 0, 0, w_img, h_img
        roi_contour = None

    # Build grid of subset centres (pixel coords).
    xs = np.arange(rx + half, rx + rw - half, step_size)
    ys = np.arange(ry + half, ry + rh - half, step_size)
    grid_x, grid_y = np.meshgrid(xs, ys)

    ny, nx = grid_x.shape
    u = np.full((ny, nx), np.nan)
    v = np.full((ny, nx), np.nan)
    corr = np.full((ny, nx), np.nan)

    if ny == 0 or nx == 0:
        return grid_x, grid_y, u, v, corr

    # Split grid rows into chunks, one per worker.
    n_workers = max(1, (os.cpu_count() or 1) - 1)
    chunk_size = max(1, -(-ny // n_workers))  # ceil division
    tasks = []
    for start in range(0, ny, chunk_size):
        end = min(start + chunk_size, ny)
        tasks.append((start, end, grid_x[start:end], grid_y[start:end]))

    init_args = (ref_gray, def_gray, roi_contour, half, search_margin,
                 ncc_threshold, h_img, w_img)

    try:
        with Pool(processes=n_workers, initializer=_init_worker,
                  initargs=init_args) as pool:
            for iy_start, u_ch, v_ch, c_ch in pool.map(_track_chunk, tasks):
                n = u_ch.shape[0]
                u[iy_start:iy_start + n] = u_ch
                v[iy_start:iy_start + n] = v_ch
                corr[iy_start:iy_start + n] = c_ch
    except Exception as exc:
        # Fallback: single-process if multiprocessing fails.
        print(f"\n  WARNING: multiprocessing failed ({exc}), "
              "falling back to single-process mode.", flush=True)
        _init_worker(*init_args)
        for task in tasks:
            iy_start, u_ch, v_ch, c_ch = _track_chunk(task)
            n = u_ch.shape[0]
            u[iy_start:iy_start + n] = u_ch
            v[iy_start:iy_start + n] = v_ch
            corr[iy_start:iy_start + n] = c_ch

    return grid_x, grid_y, u, v, corr
