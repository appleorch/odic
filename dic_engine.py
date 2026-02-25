"""Core DIC engine: normalized cross-correlation (NCC) subset tracking."""

import cv2
import numpy as np
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


def track_subsets(ref_gray, def_gray, roi, subset_size, step_size,
                  ncc_threshold, search_margin=None):
    """Track subsets from *ref_gray* to *def_gray* using NCC.

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

    for iy in range(ny):
        for ix in range(nx):
            cx = int(grid_x[iy, ix])
            cy = int(grid_y[iy, ix])

            # Skip grid points outside the polygon ROI.
            if roi_contour is not None:
                if cv2.pointPolygonTest(roi_contour, (float(cx), float(cy)), False) < 0:
                    continue

            # Extract template from reference.
            t_y0 = cy - half
            t_y1 = cy + half + 1
            t_x0 = cx - half
            t_x1 = cx + half + 1
            if t_y0 < 0 or t_x0 < 0 or t_y1 > h_img or t_x1 > w_img:
                continue
            template = ref_gray[t_y0:t_y1, t_x0:t_x1]

            # Search area in deformed image.
            s_y0 = max(cy - half - search_margin, 0)
            s_y1 = min(cy + half + 1 + search_margin, h_img)
            s_x0 = max(cx - half - search_margin, 0)
            s_x1 = min(cx + half + 1 + search_margin, w_img)
            search = def_gray[s_y0:s_y1, s_x0:s_x1]

            if (search.shape[0] < template.shape[0] or
                    search.shape[1] < template.shape[1]):
                continue

            ncc_map = _norm_cross_corr(template, search)
            if ncc_map.size == 0:
                continue

            peak_val = ncc_map.max()
            corr[iy, ix] = peak_val

            if peak_val < ncc_threshold:
                # Mark as failed — leave u, v as NaN.
                continue

            peak_idx = np.unravel_index(ncc_map.argmax(), ncc_map.shape)
            # The match position in the deformed image (top-left of template).
            match_y = s_y0 + peak_idx[0]
            match_x = s_x0 + peak_idx[1]
            # Displacement = matched centre − original centre.
            u[iy, ix] = (match_x + half) - cx
            v[iy, ix] = (match_y + half) - cy

    return grid_x, grid_y, u, v, corr
