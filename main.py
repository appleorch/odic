#!/usr/bin/env python3
"""2-D Digital Image Correlation (DIC) analysis tool — CLI entry point."""

import argparse
import multiprocessing
import os
import sys

import cv2
import numpy as np
from natsort import natsorted

from calibration import run_calibration
from roi import (
    select_roi, select_repair_zone, roi_to_slice,
    point_in_polygon, make_grid_mask,
)
from extensometer import (
    pick_extensometer_points,
    compute_extensometer_strain,
)
from dic_engine import create_engine, track_frame, shutdown_engine, update_reference
from strain import compute_strain_fields, mask_fields
from sync import parse_daq, match_frames_to_daq
from output import (
    _ensure_dirs,
    save_frame_image,
    init_video_writers,
    write_video_frame,
    close_video_writers,
    save_contour_plot,
    write_summary_csv,
    write_extensometer_csv,
    write_stress_strain_csv,
)


# ── CLI ──────────────────────────────────────────────────────────────────

def _ask_folder():
    """Open a folder-picker dialog and return the selected path (or *None*)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select folder of .tif image pairs")
        root.destroy()
        return folder if folder else None
    except Exception:
        return None


def build_parser():
    p = argparse.ArgumentParser(
        description="2-D DIC analysis tool for tensile test specimens.",
    )
    p.add_argument("--folder", required=False, default=None,
                   help="Path to folder of .tif stereo image pairs.")
    p.add_argument("--camera", type=int, choices=[0, 1], default=0,
                   help="Camera index to analyse (0=left, 1=right).")
    p.add_argument("--subset", type=int, default=29,
                   help="Subset (template) size in pixels (default 29).")
    p.add_argument("--step", type=int, default=10,
                   help="Step size in pixels (default 10).")
    p.add_argument("--ncc-threshold", type=float, default=0.7,
                   help="Minimum NCC score to accept a match (default 0.7).")
    p.add_argument("--strain-threshold", type=float, default=0.02,
                   help="Von Mises strain threshold for localization (default 0.02).")
    p.add_argument("--area", type=float, default=None,
                   help="Cross-sectional area (physical units²) for stress calc.")
    p.add_argument("--daq", default=None,
                   help="Path to DAQ .txt file (tab-separated).")
    return p


# ── image loading ────────────────────────────────────────────────────────

def load_images(folder, camera):
    """Return a naturally-sorted list of image paths for the chosen camera."""
    suffix = f"_{camera}.tif"
    all_files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    if not all_files:
        sys.exit(f"No *{suffix} images found in {folder}")
    sorted_files = natsorted(all_files)
    paths = [os.path.join(folder, f) for f in sorted_files]
    print(f"Found {len(paths)} images for camera {camera}.", flush=True)
    return paths


# ── statistics helpers ───────────────────────────────────────────────────

def _zone_stats(field, grid_x, grid_y, polygon):
    """Mean and max of *field* for grid points inside *polygon*."""
    if polygon is None:
        return np.nan, np.nan
    mask = make_grid_mask(grid_x, grid_y, polygon)
    vals = field[mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(vals)), float(np.nanmax(vals))


# ── main pipeline ────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    if args.folder is None:
        args.folder = _ask_folder()
    if not args.folder:
        sys.exit("Error: --folder is required. Usage:  dic_tool.exe --folder PATH")

    out_dir = os.path.join(os.path.dirname(args.folder.rstrip(os.sep)), "output")
    _ensure_dirs(out_dir)

    # 1. Load images.
    image_paths = load_images(args.folder, args.camera)
    ref_bgr = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if ref_bgr is None:
        sys.exit(f"Cannot read reference image: {image_paths[0]}")
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    # 2. DAQ (optional).
    daq = None
    daq_indices = None
    if args.daq:
        print("Parsing DAQ file...", flush=True)
        daq = parse_daq(args.daq)
        daq_indices = match_frames_to_daq(len(image_paths), daq)
        print(f"DAQ: {len(daq['time'])} rows, matched to {len(image_paths)} frames.",
              flush=True)

    # 3. Calibration.
    scale, unit, cal_pts = run_calibration(ref_bgr)
    step_phys = args.step * scale

    # 4. ROI.
    print("Select the primary ROI (polygon). See window for controls.", flush=True)
    roi = select_roi(ref_bgr)
    if roi is None:
        print("No ROI selected — using full image.", flush=True)
    else:
        print(f"ROI defined with {len(roi)} vertices.", flush=True)

    # 5. Repair zone.
    repair_zone = select_repair_zone(ref_bgr)
    if repair_zone is not None:
        print(f"Repair zone defined with {len(repair_zone)} vertices.", flush=True)

    # 6. Virtual extensometer.
    ext_p1, ext_p2 = pick_extensometer_points(ref_bgr)
    has_ext = ext_p1 is not None

    # 7. Run DIC frame by frame.
    n_frames = len(image_paths)
    field_names = ["U", "V", "exx", "eyy", "exy", "von_mises"]

    # Collect per-frame data for extensometer & summary.
    u_all_px = []
    v_all_px = []
    summary_rows = []
    ext_strains = []
    localization_frame = None

    # Pre-compute video writer dimensions from a test render.
    # Use a dummy figure to get the frame size.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_tmp, ax_tmp = plt.subplots(figsize=(8, 6))
    ax_tmp.imshow(np.zeros((10, 10)), cmap="jet")
    fig_tmp.canvas.draw()
    vid_h, vid_w = fig_tmp.canvas.get_width_height()[::-1]
    plt.close(fig_tmp)
    video_writers = init_video_writers(field_names, (vid_h, vid_w), out_dir, fps=10)

    grid_x, grid_y = create_engine(
        ref_gray, roi, args.subset, args.step, args.ncc_threshold,
    )

    # Accumulators for incremental tracking (frame-to-frame).
    ny, nx = grid_x.shape
    u_accum = np.zeros((ny, nx), dtype=np.float64)
    v_accum = np.zeros((ny, nx), dtype=np.float64)
    last_fields = None

    for fi in range(1, n_frames):
        pct = fi / (n_frames - 1) * 100 if n_frames > 1 else 100
        print(f"\rProcessing frame {fi}/{n_frames - 1}  [{pct:5.1f}%]",
              end="", flush=True)

        def_bgr = cv2.imread(image_paths[fi], cv2.IMREAD_COLOR)
        if def_bgr is None:
            print(f"\n  WARNING: cannot read {image_paths[fi]}, skipping.", flush=True)
            u_all_px.append(None)
            v_all_px.append(None)
            continue
        def_gray = cv2.cvtColor(def_bgr, cv2.COLOR_BGR2GRAY)

        # ---- NCC tracking (incremental: previous frame → current frame) ----
        _, _, u_inc, v_inc, corr = track_frame(def_gray)

        # Accumulate incremental displacements into totals from frame 0.
        good = ~np.isnan(u_inc)
        u_accum[good] += u_inc[good]
        v_accum[good] += v_inc[good]

        # For this frame's output, use accumulated totals; NaN where this
        # frame's tracking failed (so strain/mask reflect current quality).
        u_px = u_accum.copy()
        v_px = v_accum.copy()
        u_px[~good] = np.nan
        v_px[~good] = np.nan

        # Physical displacements.
        u_phys = u_px * scale
        v_phys = v_px * scale

        # ---- Strain ----
        exx, eyy, exy, vm = compute_strain_fields(u_phys, v_phys, step_phys)

        fields = {
            "U": u_phys, "V": v_phys,
            "exx": exx, "eyy": eyy, "exy": exy, "von_mises": vm,
        }
        fields = mask_fields(fields, corr, args.ncc_threshold)

        # Store pixel displacements for extensometer.
        u_all_px.append(u_px)
        v_all_px.append(v_px)

        # ---- Localization detection ----
        vm_field = fields["von_mises"]
        vm_valid = vm_field[~np.isnan(vm_field)] if vm_field.size else np.array([])
        localization_flag = False
        if localization_frame is None and vm_valid.size > 0:
            if np.nanmax(vm_valid) >= args.strain_threshold:
                localization_frame = fi
                localization_flag = True
                peak_idx = np.unravel_index(
                    np.nanargmax(vm_field), vm_field.shape
                )
                peak_px = (int(grid_x[peak_idx]), int(grid_y[peak_idx]))
                peak_phys = (peak_px[0] * scale, peak_px[1] * scale)
                in_repair = point_in_polygon(peak_px[0], peak_px[1], repair_zone)
                zone_label = "repair zone" if in_repair else "base material"
                msg = (
                    f"\n  ** Strain localization at frame {fi}: "
                    f"von Mises = {np.nanmax(vm_field):.4f} at "
                    f"({peak_phys[0]:.2f}, {peak_phys[1]:.2f}) {unit} "
                    f"[{zone_label}]"
                )
                if daq is not None and daq_indices is not None:
                    t_val = daq["time"][daq_indices[fi]]
                    msg += f", DAQ time = {t_val:.3f} s"
                print(msg, flush=True)

        # ---- ROI / repair zone statistics ----
        roi_mean, roi_max = _zone_stats(vm_field, grid_x, grid_y, roi)
        rz_mean, rz_max = _zone_stats(vm_field, grid_x, grid_y, repair_zone)

        # Base material = ROI minus repair zone.
        if roi is not None and repair_zone is not None:
            repair_mask = make_grid_mask(grid_x, grid_y, repair_zone)
            base_mask = ~repair_mask
            base_vals = vm_field[base_mask]
            base_vals = base_vals[~np.isnan(base_vals)]
            base_mean = float(np.nanmean(base_vals)) if base_vals.size else np.nan
            base_max = float(np.nanmax(base_vals)) if base_vals.size else np.nan
        else:
            base_mean, base_max = roi_mean, roi_max

        # ---- summary row ----
        row = {
            "frame": fi,
            "roi_vm_mean": roi_mean,
            "roi_vm_max": roi_max,
            "repair_vm_mean": rz_mean,
            "repair_vm_max": rz_max,
            "base_vm_mean": base_mean,
            "base_vm_max": base_max,
            "extensometer_strain": "",
            "load": "",
            "stress": "",
            "localization_event": localization_flag,
        }
        if daq is not None and daq_indices is not None:
            d_idx = daq_indices[fi]
            row["load"] = daq["load"][d_idx]
            if args.area:
                row["stress"] = daq["load"][d_idx] / args.area
        summary_rows.append(row)

        # ---- per-frame outputs ----
        for fname in field_names:
            save_frame_image(
                fields[fname], fname, fi, grid_x, grid_y,
                scale, unit, roi, repair_zone, out_dir,
            )
            write_video_frame(
                video_writers[fname], fields[fname],
                grid_x, grid_y, scale, unit, fname, fi, roi, repair_zone,
            )

        # Advance reference image so next frame is tracked incrementally.
        update_reference(def_gray)
        last_fields = fields

    print("\n", flush=True)
    close_video_writers(video_writers)

    # ---- Extensometer post-processing ----
    if has_ext and grid_x is not None:
        # Replace None entries with NaN arrays for safety.
        ny, nx = grid_x.shape
        u_safe = [
            u if u is not None else np.full((ny, nx), np.nan)
            for u in u_all_px
        ]
        v_safe = [
            v if v is not None else np.full((ny, nx), np.nan)
            for v in v_all_px
        ]
        ext_strains = compute_extensometer_strain(
            ext_p1, ext_p2, grid_x, grid_y, u_safe, v_safe,
        )
        frame_indices = list(range(1, n_frames))
        write_extensometer_csv(frame_indices, ext_strains, out_dir)

        # Backfill extensometer strain into summary rows.
        for row, es in zip(summary_rows, ext_strains):
            row["extensometer_strain"] = es

        if daq is not None and args.area:
            stresses = []
            for fi_idx in range(1, n_frames):
                d_idx = daq_indices[fi_idx]
                stresses.append(daq["load"][d_idx] / args.area)
            write_stress_strain_csv(ext_strains, stresses, out_dir)

    # ---- Summary CSV ----
    write_summary_csv(summary_rows, out_dir)

    # ---- Final-frame contour plots ----
    if grid_x is not None and last_fields is not None:
        print("Generating final-frame contour plots...", flush=True)
        ext_plot_pts = (ext_p1, ext_p2) if has_ext else None
        for fname in field_names:
            save_contour_plot(
                last_fields[fname], fname, grid_x, grid_y,
                scale, unit, roi, repair_zone, ext_plot_pts, out_dir,
            )

    shutdown_engine()
    print("Done.", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
