"""ROI and optional repair-zone selection with polygon tool, zoom & pan."""

import cv2
import numpy as np

_CLOSE_RADIUS = 12   # screen-px: click within this to close the polygon
_ZOOM_FACTOR = 1.2   # per scroll tick
_MAX_WIN = (1280, 900)


# ── helpers ───────────────────────────────────────────────────────────────

def polygon_bbox(polygon):
    """Return *(x, y, w, h)* bounding box of a polygon (``None`` ➜ ``None``)."""
    if polygon is None:
        return None
    return cv2.boundingRect(polygon.reshape(-1, 1, 2))


def point_in_polygon(px, py, polygon):
    """Return True if pixel *(px, py)* lies inside *polygon*."""
    if polygon is None:
        return False
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0


def make_grid_mask(grid_x, grid_y, polygon):
    """Boolean array: *True* where the grid point is inside *polygon*."""
    if polygon is None:
        return np.ones(grid_x.shape, dtype=bool)
    pts = polygon.reshape(-1, 1, 2).astype(np.int32)
    # Build a raster mask and do a vectorized lookup instead of per-point tests.
    max_y = max(int(np.max(grid_y)), int(np.max(polygon[:, 1]))) + 1
    max_x = max(int(np.max(grid_x)), int(np.max(polygon[:, 0]))) + 1
    raster = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
    cv2.fillPoly(raster, [pts], 255)
    gy = np.clip(grid_y.astype(np.intp), 0, max_y)
    gx = np.clip(grid_x.astype(np.intp), 0, max_x)
    return raster[gy, gx] > 0


def roi_to_slice(roi, image_shape):
    """Convert polygon (or *None*) to numpy-friendly row / col slices."""
    if roi is None:
        h, w = image_shape[:2]
        return slice(0, h), slice(0, w)
    x, y, w, h = polygon_bbox(roi)
    return slice(y, y + h), slice(x, x + w)


# ── interactive polygon selector ─────────────────────────────────────────

class _PolygonSelector:
    """OpenCV-based polygon drawing with scroll-zoom and right-drag pan."""

    def __init__(self, image, title):
        self.image = image
        self.title = title
        self.points = []          # image-space vertices [(x, y), ...]
        self.closed = False
        self.confirmed = False

        ih, iw = image.shape[:2]
        z = min(_MAX_WIN[0] / iw, _MAX_WIN[1] / ih, 1.0)
        self.zoom = z
        self.win_w = int(iw * z)
        self.win_h = int(ih * z)
        self.offset = [0.0, 0.0]  # top-left of view in image coords

        self._panning = False
        self._pan_start = None
        self._pan_offset0 = None

    # ── coordinate helpers ──

    def _scr2img(self, sx, sy):
        return sx / self.zoom + self.offset[0], sy / self.zoom + self.offset[1]

    def _img2scr(self, ix, iy):
        return int((ix - self.offset[0]) * self.zoom), int((iy - self.offset[1]) * self.zoom)

    def _clamp_offset(self):
        ih, iw = self.image.shape[:2]
        vw, vh = self.win_w / self.zoom, self.win_h / self.zoom
        self.offset[0] = max(0, min(self.offset[0], max(iw - vw, 0)))
        self.offset[1] = max(0, min(self.offset[1], max(ih - vh, 0)))

    # ── rendering ──

    def _render(self):
        ih, iw = self.image.shape[:2]
        vw, vh = self.win_w / self.zoom, self.win_h / self.zoom
        x0, y0 = self.offset

        canvas = self.image.copy()

        if self.points:
            pts = np.array(self.points, dtype=np.int32)

            if self.closed and len(self.points) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], (0, 200, 0))
                cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
                cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
            else:
                cv2.polylines(canvas, [pts], False, (0, 255, 0), 2)

            for i, (px, py) in enumerate(self.points):
                color = (0, 0, 255) if i == 0 else (0, 255, 0)
                cv2.circle(canvas, (px, py), 4, color, -1)
                if i == 0 and not self.closed and len(self.points) >= 3:
                    r_img = max(1, int(_CLOSE_RADIUS / self.zoom))
                    cv2.circle(canvas, (px, py), r_img, (0, 0, 255), 1)

        # Crop to current view and resize.
        x0i, y0i = max(int(x0), 0), max(int(y0), 0)
        x1i = min(int(x0 + vw) + 1, iw)
        y1i = min(int(y0 + vh) + 1, ih)
        crop = canvas[y0i:y1i, x0i:x1i]
        if crop.size == 0:
            return np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        view = cv2.resize(crop, (self.win_w, self.win_h), interpolation=cv2.INTER_LINEAR)

        # HUD
        if not self.closed:
            msg = "L-click: add point | Close: click 1st pt | Ctrl+Z: undo | Scroll/+/-: zoom | R-drag: pan"
        else:
            msg = "ENTER: confirm | ESC: cancel | R: reset | Ctrl+Z: undo"
        cv2.putText(view, msg, (10, self.win_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    # ── mouse callback ──

    def _on_mouse(self, event, sx, sy, flags, param):
        # ---- pan (right-drag) ----
        if event == cv2.EVENT_RBUTTONDOWN:
            self._panning = True
            self._pan_start = (sx, sy)
            self._pan_offset0 = list(self.offset)
        elif event == cv2.EVENT_MOUSEMOVE and self._panning:
            dx = (sx - self._pan_start[0]) / self.zoom
            dy = (sy - self._pan_start[1]) / self.zoom
            self.offset[0] = self._pan_offset0[0] - dx
            self.offset[1] = self._pan_offset0[1] - dy
            self._clamp_offset()
            cv2.imshow(self.title, self._render())
        elif event == cv2.EVENT_RBUTTONUP:
            self._panning = False

        # ---- add vertex (left-click) ----
        elif event == cv2.EVENT_LBUTTONDOWN and not self._panning:
            if self.closed:
                return
            ix, iy = self._scr2img(sx, sy)
            ix, iy = int(round(ix)), int(round(iy))

            # Check if closing the polygon.
            if len(self.points) >= 3:
                fsx, fsy = self._img2scr(*self.points[0])
                dist = ((sx - fsx) ** 2 + (sy - fsy) ** 2) ** 0.5
                if dist < _CLOSE_RADIUS:
                    self.closed = True
                    cv2.imshow(self.title, self._render())
                    return

            self.points.append((ix, iy))
            cv2.imshow(self.title, self._render())

        # ---- zoom (scroll wheel) ----
        elif event == cv2.EVENT_MOUSEWHEEL:
            ix0, iy0 = self._scr2img(sx, sy)
            self.zoom *= _ZOOM_FACTOR if flags > 0 else (1.0 / _ZOOM_FACTOR)
            self.zoom = max(0.05, min(self.zoom, 20.0))
            self.offset[0] = ix0 - sx / self.zoom
            self.offset[1] = iy0 - sy / self.zoom
            self._clamp_offset()
            cv2.imshow(self.title, self._render())

    # ── main loop ──

    def run(self):
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.title, self._on_mouse)
        cv2.imshow(self.title, self._render())

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13 or key == 32:          # ENTER / SPACE
                if self.closed and len(self.points) >= 3:
                    self.confirmed = True
                    break
            elif key == 27:                     # ESC
                break
            elif key in (ord('r'), ord('R')):   # reset
                self.points = []
                self.closed = False
                cv2.imshow(self.title, self._render())
            elif key == 26:                     # Ctrl+Z — undo
                if self.closed:
                    self.closed = False
                elif self.points:
                    self.points.pop()
                cv2.imshow(self.title, self._render())
            elif key in (ord('+'), ord('=')):   # zoom in via keyboard
                cx, cy = self.win_w // 2, self.win_h // 2
                ix0, iy0 = self._scr2img(cx, cy)
                self.zoom = min(self.zoom * _ZOOM_FACTOR, 20.0)
                self.offset[0] = ix0 - cx / self.zoom
                self.offset[1] = iy0 - cy / self.zoom
                self._clamp_offset()
                cv2.imshow(self.title, self._render())
            elif key in (ord('-'), ord('_')):   # zoom out via keyboard
                cx, cy = self.win_w // 2, self.win_h // 2
                ix0, iy0 = self._scr2img(cx, cy)
                self.zoom = max(self.zoom / _ZOOM_FACTOR, 0.05)
                self.offset[0] = ix0 - cx / self.zoom
                self.offset[1] = iy0 - cy / self.zoom
                self._clamp_offset()
                cv2.imshow(self.title, self._render())

        cv2.destroyWindow(self.title)
        if self.confirmed:
            return np.array(self.points, dtype=np.int32)
        return None


# ── public API ────────────────────────────────────────────────────────────

def select_roi(reference_image, title="Select ROI"):
    """Prompt the user to draw a polygon ROI with zoom & pan.

    Returns
    -------
    roi : ndarray of shape (N, 2) with dtype int32, or *None*.
    """
    selector = _PolygonSelector(reference_image, title)
    return selector.run()


def select_repair_zone(reference_image):
    """Optionally select a repair zone polygon inside the ROI.

    Returns
    -------
    repair : ndarray (N, 2) int32 or *None*
    """
    answer = input("Define a repair zone? (y/n): ").strip().lower()
    if answer != "y":
        return None
    return select_roi(reference_image, title="Select Repair Zone")
