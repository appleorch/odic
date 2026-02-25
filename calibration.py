"""Interactive scale calibration via line drawing on the reference image."""

import cv2
import numpy as np


def _draw_calibration_line(image):
    """Let the user click two points on the image; return ((x1,y1),(x2,y2))."""
    points = []
    display = image.copy()
    window = "Calibration - click two points | Ctrl+Z: undo | then press any key"

    def _redraw():
        display[:] = image
        for pt in points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow(window, display)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
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
        raise RuntimeError("Calibration requires exactly two points.")
    return tuple(points[0]), tuple(points[1])


def run_calibration(reference_image):
    """Interactive calibration loop.

    Returns
    -------
    scale : float
        Physical units per pixel.
    unit : str
        Unit label chosen by the user.
    cal_points : tuple of two (x, y) tuples
        The pixel coordinates used for calibration.
    """
    while True:
        p1, p2 = _draw_calibration_line(reference_image)
        pixel_dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        print(f"Pixel distance between points: {pixel_dist:.1f} px", flush=True)

        phys_length = float(input("Enter the physical length this line represents: "))

        unit = ""
        while unit not in ("mm", "cm", "inches"):
            unit = input("Choose unit system (mm / cm / inches): ").strip().lower()

        scale = phys_length / pixel_dist
        print(
            f"Scale factor: {scale:.6f} {unit}/px  "
            f"({1.0 / scale:.2f} px/{unit})",
            flush=True,
        )

        confirm = input("Accept calibration? (y/n): ").strip().lower()
        if confirm == "y":
            return scale, unit, (p1, p2)
        print("Redoing calibration...", flush=True)
