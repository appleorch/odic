"""All file writing, video generation, and plot export."""

import os
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for PyInstaller).
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ── helpers ──────────────────────────────────────────────────────────────

def _ensure_dirs(base):
    for sub in ("frames", "videos", "plots"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)


def _phys_extent(grid_x, grid_y, scale):
    """Return (left, right, bottom, top) in physical units for imshow extent."""
    return (
        grid_x.min() * scale,
        grid_x.max() * scale,
        grid_y.max() * scale,  # bottom (imshow convention)
        grid_y.min() * scale,  # top
    )


def _add_overlays(ax, roi, repair_zone, scale, ext_pts=None):
    """Draw ROI and repair zone rectangles and extensometer line."""
    if roi is not None:
        x, y, w, h = roi
        ax.add_patch(Rectangle(
            (x * scale, y * scale), w * scale, h * scale,
            linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--",
        ))
    if repair_zone is not None:
        x, y, w, h = repair_zone
        ax.add_patch(Rectangle(
            (x * scale, y * scale), w * scale, h * scale,
            linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle=":",
        ))
    if ext_pts is not None:
        p1, p2 = ext_pts
        ax.plot(
            [p1[0] * scale, p2[0] * scale],
            [p1[1] * scale, p2[1] * scale],
            "m-o", linewidth=2, markersize=5,
        )


# ── per-frame colormapped images ────────────────────────────────────────

def save_frame_image(field, name, frame_idx, grid_x, grid_y, scale, unit,
                     roi, repair_zone, out_dir):
    """Save a single colormapped .png for one field / one frame."""
    ext = _phys_extent(grid_x, grid_y, scale)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        field, extent=ext, origin="upper", cmap="jet",
        aspect="equal", interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(name)
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(f"{name}  — frame {frame_idx}")
    _add_overlays(ax, roi, repair_zone, scale)
    path = os.path.join(out_dir, "frames", f"{name}_frame{frame_idx:04d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── videos ───────────────────────────────────────────────────────────────

def init_video_writers(field_names, frame_shape, out_dir, fps=10):
    """Create a dict of cv2.VideoWriter objects keyed by field name."""
    writers = {}
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for name in field_names:
        path = os.path.join(out_dir, "videos", f"{name}.mp4")
        writers[name] = cv2.VideoWriter(path, fourcc, fps, (w, h))
    return writers


def write_video_frame(writer, field, grid_x, grid_y, scale, unit, name,
                      frame_idx, roi, repair_zone):
    """Render a colormapped frame and feed it to the VideoWriter."""
    ext = _phys_extent(grid_x, grid_y, scale)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        field, extent=ext, origin="upper", cmap="jet",
        aspect="equal", interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(f"{name}  — frame {frame_idx}")
    _add_overlays(ax, roi, repair_zone, scale)
    fig.canvas.draw()

    # Convert matplotlib figure to BGR image for OpenCV.
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    # Resize to match writer dimensions if needed.
    writer.write(bgr)
    plt.close(fig)


def close_video_writers(writers):
    for w in writers.values():
        w.release()


# ── final-frame contour plots (thesis quality) ──────────────────────────

def save_contour_plot(field, name, grid_x, grid_y, scale, unit,
                      roi, repair_zone, ext_pts, out_dir):
    """Save a high-resolution contour plot for the final frame."""
    phys_x = grid_x * scale
    phys_y = grid_y * scale

    fig, ax = plt.subplots(figsize=(10, 7))

    # Mask NaN for contouring.
    masked = np.ma.array(field, mask=np.isnan(field))
    levels = 20
    cs = ax.contourf(phys_x, phys_y, masked, levels=levels, cmap="jet")
    ax.contour(phys_x, phys_y, masked, levels=levels, colors="k",
               linewidths=0.3)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(name)
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(f"{name}  — final frame")
    ax.invert_yaxis()
    _add_overlays(ax, roi, repair_zone, scale, ext_pts)

    path = os.path.join(out_dir, "plots", f"{name}_contour.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ── CSV outputs ──────────────────────────────────────────────────────────

def write_summary_csv(rows, out_dir):
    path = os.path.join(out_dir, "summary.csv")
    if not rows:
        return
    keys = rows[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV written to {path}", flush=True)


def write_extensometer_csv(frame_indices, strains, out_dir):
    path = os.path.join(out_dir, "extensometer_strain.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "extensometer_strain"])
        for i, s in zip(frame_indices, strains):
            writer.writerow([i, s])
    print(f"Extensometer CSV written to {path}", flush=True)


def write_stress_strain_csv(strains, stresses, out_dir):
    path = os.path.join(out_dir, "stress_strain.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dic_strain", "stress"])
        for s, sigma in zip(strains, stresses):
            writer.writerow([s, sigma])
    print(f"Stress-strain CSV written to {path}", flush=True)
