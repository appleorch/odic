"""ROI and optional repair-zone selection using OpenCV."""

import cv2


def select_roi(reference_image, title="Select ROI (press ENTER/SPACE to confirm, C to cancel)"):
    """Prompt the user to draw a rectangle.

    Returns
    -------
    roi : tuple (x, y, w, h) or None
        Pixel coordinates, or *None* if the user cancels / draws nothing.
    """
    r = cv2.selectROI(title, reference_image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    if r == (0, 0, 0, 0):
        return None
    return r  # (x, y, w, h)


def select_repair_zone(reference_image):
    """Optionally select a repair zone inside the ROI.

    Returns
    -------
    repair : tuple (x, y, w, h) or None
    """
    answer = input("Define a repair zone? (y/n): ").strip().lower()
    if answer != "y":
        return None
    return select_roi(
        reference_image,
        title="Select Repair Zone (ENTER/SPACE to confirm, C to cancel)",
    )


def roi_to_slice(roi, image_shape):
    """Convert (x, y, w, h) to numpy-friendly row/col slices."""
    if roi is None:
        h, w = image_shape[:2]
        return slice(0, h), slice(0, w)
    x, y, w, h = roi
    return slice(y, y + h), slice(x, x + w)


def point_in_rect(px, py, rect):
    """Return True if pixel (px, py) lies inside *rect* (x, y, w, h)."""
    if rect is None:
        return False
    x, y, w, h = rect
    return x <= px < x + w and y <= py < y + h
