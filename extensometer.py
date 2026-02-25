"""Virtual extensometer: two-point displacement tracking and strain."""

import cv2
import numpy as np


def pick_extensometer_points(reference_image):
    """Let the user click two points for the virtual extensometer.

    Returns
    -------
    (p1, p2) : tuple of (x, y) pixel coordinates, or (None, None) if skipped.
    """
    answer = input("Define a virtual extensometer? (y/n): ").strip().lower()
    if answer != "y":
        return None, None

    points = []
    display = reference_image.copy()
    window = "Virtual Extensometer - click two points | Ctrl+Z: undo | then press any key"

    def _redraw():
        display[:] = reference_image
        for pt in points:
            cv2.circle(display, pt, 5, (255, 0, 0), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (255, 0, 0), 2)
        cv2.imshow(window, display)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            _redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, display)
    cv2.setMouseCallback(window, _on_mouse)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 26 and points:            # Ctrl+Z — undo
            points.pop()
            _redraw()
        elif len(points) == 2 and key != 255:
            break
    cv2.destroyWindow(window)

    if len(points) != 2:
        print("Extensometer requires exactly two points — skipping.", flush=True)
        return None, None
    return tuple(points[0]), tuple(points[1])


def nearest_grid_indices(px, py, grid_x, grid_y):
    """Return (iy, ix) of the grid point nearest to pixel (px, py)."""
    dist = (grid_x - px) ** 2 + (grid_y - py) ** 2
    idx = np.unravel_index(np.nanargmin(dist), dist.shape)
    return idx


def compute_extensometer_strain(ext_p1, ext_p2, grid_x, grid_y, u_all, v_all):
    """Compute engineering strain from extensometer point tracking.

    Parameters
    ----------
    ext_p1, ext_p2 : (x, y) pixel coordinates of extensometer ends
    grid_x, grid_y : 2-D grid of subset centres
    u_all, v_all : lists of 2-D displacement arrays (one per frame, pixels)

    Returns
    -------
    strains : list of float (one per frame), NaN if either point is masked.
    """
    iy1, ix1 = nearest_grid_indices(ext_p1[0], ext_p1[1], grid_x, grid_y)
    iy2, ix2 = nearest_grid_indices(ext_p2[0], ext_p2[1], grid_x, grid_y)

    # Initial gauge length (pixels).
    x1_0 = grid_x[iy1, ix1]
    y1_0 = grid_y[iy1, ix1]
    x2_0 = grid_x[iy2, ix2]
    y2_0 = grid_y[iy2, ix2]
    L0 = np.hypot(x2_0 - x1_0, y2_0 - y1_0)
    if L0 < 1e-6:
        return [np.nan] * len(u_all)

    strains = []
    for u, v_field in zip(u_all, v_all):
        u1 = u[iy1, ix1]
        v1 = v_field[iy1, ix1]
        u2 = u[iy2, ix2]
        v2 = v_field[iy2, ix2]
        if np.isnan(u1) or np.isnan(u2):
            strains.append(np.nan)
            continue
        x1_def = x1_0 + u1
        y1_def = y1_0 + v1
        x2_def = x2_0 + u2
        y2_def = y2_0 + v2
        L = np.hypot(x2_def - x1_def, y2_def - y1_def)
        strains.append((L - L0) / L0)
    return strains
