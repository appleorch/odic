#!/usr/bin/env python3
"""
Comprehensive 2D Digital Image Correlation (DIC) Analysis Program
Inspired by Vic-2D from Correlated Solutions

Features:
- Subset-based correlation with sub-pixel accuracy
- Full-field displacement (U, V) computation
- Strain tensor computation (Exx, Eyy, Exy, principal strains, von Mises)
- Load-displacement integration from CSV
- Interactive HTML visualization dashboard (Vic-2D iris-style)

Author: DIC Analysis Tool
"""

import numpy as np
import cv2
import os
import json
import csv
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Configuration
# ============================================================
class DICConfig:
    """DIC analysis configuration parameters"""
    def __init__(self):
        # Subset parameters
        self.subset_size = 31        # pixels (odd number)
        self.step_size = 15          # grid spacing in pixels

        # Correlation parameters
        self.search_radius = 50      # initial search window (pixels)
        self.use_optical_flow = True # use optical flow for initial guess
        self.subpixel_search = 3     # +/- pixels around initial guess for fine search
        self.correlation_threshold = 0.5  # minimum ZNCC to accept a match

        # Optical flow parameters (Farneback)
        self.of_levels = 5           # number of pyramid levels
        self.of_winsize = 31         # averaging window size
        self.of_iterations = 5       # iterations at each pyramid level
        self.of_poly_n = 7           # polynomial expansion neighbourhood
        self.of_poly_sigma = 1.5     # Gaussian sigma for polynomial

        # Strain computation
        self.strain_window = 5       # window size for strain computation (in grid points)
        self.strain_type = 'engineering'  # 'engineering' or 'green_lagrange'
        self.strain_min_points = 6   # minimum valid points in window for strain fit

        # Displacement filtering (applied before strain computation)
        self.displacement_filter = 'none'  # 'none', 'gaussian', 'median'
        self.displacement_filter_size = 3  # kernel size for displacement filter

        # Processing
        self.num_threads = 4
        self.downsample = 2          # downsample images by this factor for speed
        self.frame_skip = 1          # process every Nth frame
        self.start_frame = 0         # first frame index
        self.end_frame = None        # last frame index (None = all frames)
        self.reference_frame = 0     # which frame to use as reference (0 = first)

        # Polygon ROI: list of (x, y) vertices in ORIGINAL image pixel coords, or None for full image
        self.roi_polygon = None

    # ── Fidelity presets ──
    @staticmethod
    def from_preset(name):
        """Return a DICConfig initialised from a named preset."""
        cfg = DICConfig()
        if name == 'draft':
            cfg.subset_size = 21
            cfg.step_size = 25
            cfg.downsample = 6
            cfg.strain_window = 3
            cfg.of_levels = 3
            cfg.of_winsize = 21
            cfg.of_iterations = 3
            cfg.of_poly_n = 5
            cfg.of_poly_sigma = 1.1
            cfg.subpixel_search = 2
            cfg.correlation_threshold = 0.4
            cfg.displacement_filter = 'none'
        elif name == 'standard':
            cfg.subset_size = 31
            cfg.step_size = 15
            cfg.downsample = 4
            cfg.strain_window = 3
            cfg.of_levels = 5
            cfg.of_winsize = 31
            cfg.of_iterations = 5
            cfg.of_poly_n = 7
            cfg.of_poly_sigma = 1.5
            cfg.subpixel_search = 3
            cfg.correlation_threshold = 0.5
            cfg.displacement_filter = 'none'
        elif name == 'high':
            cfg.subset_size = 41
            cfg.step_size = 10
            cfg.downsample = 2
            cfg.strain_window = 5
            cfg.of_levels = 6
            cfg.of_winsize = 41
            cfg.of_iterations = 7
            cfg.of_poly_n = 7
            cfg.of_poly_sigma = 1.5
            cfg.subpixel_search = 4
            cfg.correlation_threshold = 0.6
            cfg.displacement_filter = 'gaussian'
            cfg.displacement_filter_size = 3
        elif name == 'ultra':
            cfg.subset_size = 51
            cfg.step_size = 7
            cfg.downsample = 1
            cfg.strain_window = 5
            cfg.of_levels = 7
            cfg.of_winsize = 51
            cfg.of_iterations = 10
            cfg.of_poly_n = 7
            cfg.of_poly_sigma = 1.5
            cfg.subpixel_search = 5
            cfg.correlation_threshold = 0.65
            cfg.displacement_filter = 'gaussian'
            cfg.displacement_filter_size = 3
        return cfg

# ============================================================
# Image Loading
# ============================================================
class ImageLoader:
    """Loads and manages DIC image sequences"""

    def __init__(self, data_dir, config):
        self.data_dir = Path(data_dir)
        self.config = config
        self.camera0_files = []
        self.camera1_files = []
        self._scan_files()

    def _scan_files(self):
        """Scan directory for image pairs"""
        all_tifs = sorted(self.data_dir.glob('*.tif'))
        for f in all_tifs:
            name = f.stem
            if name.endswith('_0'):
                self.camera0_files.append(f)
            elif name.endswith('_1'):
                self.camera1_files.append(f)
        print(f"Found {len(self.camera0_files)} camera 0 images, {len(self.camera1_files)} camera 1 images")

    def load_image(self, filepath):
        """Load a single image with downsampling (full image always loaded;
        polygon ROI is applied at the grid-point level, not here)."""
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot load image: {filepath}")

        if self.config.downsample > 1:
            ds = self.config.downsample
            img = cv2.resize(img, (img.shape[1]//ds, img.shape[0]//ds),
                           interpolation=cv2.INTER_AREA)
        return img

    def get_frame_count(self):
        return len(self.camera0_files)

    def get_frame_pair(self, idx):
        """Get camera 0 and camera 1 images for frame idx"""
        img0 = self.load_image(self.camera0_files[idx])
        img1 = self.load_image(self.camera1_files[idx]) if idx < len(self.camera1_files) else None
        return img0, img1

# ============================================================
# Load Data Integration
# ============================================================
class LoadData:
    """Manages load/displacement data from CSV"""

    def __init__(self, csv_path):
        self.time = []
        self.load_N = []
        self.displacement_mm = []
        self.frame_indices = []
        self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        """Parse the synchronized CSV file"""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.frame_indices.append(int(row['Count']))
                self.time.append(float(row['Time_0_0']))
                self.load_N.append(float(row['Load_(N)']))
                self.displacement_mm.append(float(row['Displacement_(mm)']))

        self.time = np.array(self.time)
        self.load_N = np.array(self.load_N)
        self.displacement_mm = np.array(self.displacement_mm)
        self.frame_indices = np.array(self.frame_indices)

        # Normalize time to start at 0
        self.time -= self.time[0]
        # Normalize displacement to start at 0
        self.displacement_mm -= self.displacement_mm[0]

        print(f"Load data: {len(self.frame_indices)} frames, "
              f"Load range: {self.load_N.min():.1f} to {self.load_N.max():.1f} N, "
              f"Displacement range: {self.displacement_mm.min():.3f} to {self.displacement_mm.max():.3f} mm")

# ============================================================
# DIC Correlation Engine
# ============================================================
class DICEngine:
    """2D Digital Image Correlation Engine"""

    def __init__(self, config):
        self.config = config

    def compute_displacement_field(self, ref_img, def_img, prev_u=None, prev_v=None,
                                     roi_mask=None):
        """
        Compute full-field displacement between reference and deformed images.

        Uses a two-stage approach:
        1. Dense optical flow (Farneback) for initial estimate
        2. Subset-based ZNCC refinement for sub-pixel accuracy

        Args:
            roi_mask: Optional 2D bool array (ny, nx) — True for grid points to process.
                      Points outside the mask are set to NaN / -1.

        Returns: grid_x, grid_y, u_field, v_field, correlation_field
        """
        h, w = ref_img.shape
        ss = self.config.subset_size
        step = self.config.step_size
        half_ss = ss // 2

        # Define grid points
        margin = half_ss + 5
        grid_y_coords = np.arange(margin, h - margin, step)
        grid_x_coords = np.arange(margin, w - margin, step)
        ny, nx = len(grid_y_coords), len(grid_x_coords)

        grid_x, grid_y = np.meshgrid(grid_x_coords, grid_y_coords)

        # Stage 1: Get initial displacement estimate via optical flow
        if self.config.use_optical_flow:
            flow = cv2.calcOpticalFlowFarneback(
                ref_img, def_img,
                None,
                pyr_scale=0.5,
                levels=self.config.of_levels,
                winsize=self.config.of_winsize,
                iterations=self.config.of_iterations,
                poly_n=self.config.of_poly_n,
                poly_sigma=self.config.of_poly_sigma,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
            u_init = flow[:, :, 0]
            v_init = flow[:, :, 1]
        elif prev_u is not None:
            u_init = prev_u
            v_init = prev_v
        else:
            u_init = np.zeros((h, w), dtype=np.float32)
            v_init = np.zeros((h, w), dtype=np.float32)

        # Stage 2: Refine with subset-based ZNCC
        u_field = np.zeros((ny, nx), dtype=np.float64)
        v_field = np.zeros((ny, nx), dtype=np.float64)
        corr_field = np.zeros((ny, nx), dtype=np.float64)

        ref_float = ref_img.astype(np.float64)
        def_float = def_img.astype(np.float64)

        for iy in range(ny):
            for ix in range(nx):
                # Skip points outside the polygon ROI
                if roi_mask is not None and not roi_mask[iy, ix]:
                    u_field[iy, ix] = np.nan
                    v_field[iy, ix] = np.nan
                    corr_field[iy, ix] = -1
                    continue

                cy, cx = grid_y_coords[iy], grid_x_coords[ix]

                # Get initial guess from optical flow
                u0 = u_init[cy, cx]
                v0 = v_init[cy, cx]

                # Extract reference subset
                ref_sub = ref_float[cy-half_ss:cy+half_ss+1, cx-half_ss:cx+half_ss+1]
                ref_mean = ref_sub.mean()
                ref_std = ref_sub.std()
                if ref_std < 1.0:
                    corr_field[iy, ix] = -1  # mark as bad
                    continue
                ref_norm = (ref_sub - ref_mean) / ref_std

                # Sub-pixel refinement around the initial guess
                best_u, best_v, best_corr = self._subpixel_refine(
                    def_float, ref_norm, cx, cy, u0, v0, half_ss
                )

                u_field[iy, ix] = best_u
                v_field[iy, ix] = best_v
                corr_field[iy, ix] = best_corr

        return grid_x, grid_y, u_field, v_field, corr_field

    def _subpixel_refine(self, def_img, ref_norm, cx, cy, u0, v0, half_ss):
        """Sub-pixel refinement using parabolic interpolation of ZNCC"""
        h, w = def_img.shape
        ss = 2 * half_ss + 1
        search = self.config.subpixel_search  # +/- pixels around initial guess for fine search

        best_corr = -2
        best_du, best_dv = 0, 0

        # Integer-pixel search in small window
        for dy in range(-search, search+1):
            for dx in range(-search, search+1):
                ty = int(round(cy + v0)) + dy
                tx = int(round(cx + u0)) + dx

                if (ty - half_ss < 0 or ty + half_ss + 1 > h or
                    tx - half_ss < 0 or tx + half_ss + 1 > w):
                    continue

                def_sub = def_img[ty-half_ss:ty+half_ss+1, tx-half_ss:tx+half_ss+1]
                def_mean = def_sub.mean()
                def_std = def_sub.std()
                if def_std < 1.0:
                    continue
                def_norm = (def_sub - def_mean) / def_std

                zncc = np.sum(ref_norm * def_norm) / (ss * ss)
                if zncc > best_corr:
                    best_corr = zncc
                    best_du = dx
                    best_dv = dy

        # Parabolic sub-pixel refinement
        u_int = int(round(cx + u0)) + best_du - cx
        v_int = int(round(cy + v0)) + best_dv - cy

        # Try to fit parabola in u direction
        sub_u = self._try_parabolic(def_img, ref_norm, cx, cy, u_int, v_int, half_ss, axis='u')
        sub_v = self._try_parabolic(def_img, ref_norm, cx, cy, u_int, v_int, half_ss, axis='v')

        return u_int + sub_u, v_int + sub_v, best_corr

    def _try_parabolic(self, def_img, ref_norm, cx, cy, u_int, v_int, half_ss, axis='u'):
        """Parabolic sub-pixel interpolation along one axis"""
        h, w = def_img.shape
        ss = 2 * half_ss + 1
        corrs = []

        for offset in [-1, 0, 1]:
            if axis == 'u':
                tx = cx + u_int + offset
                ty = cy + v_int
            else:
                tx = cx + u_int
                ty = cy + v_int + offset

            if (ty - half_ss < 0 or ty + half_ss + 1 > h or
                tx - half_ss < 0 or tx + half_ss + 1 > w):
                return 0.0

            def_sub = def_img[ty-half_ss:ty+half_ss+1, tx-half_ss:tx+half_ss+1]
            def_mean = def_sub.mean()
            def_std = def_sub.std()
            if def_std < 1.0:
                return 0.0
            def_norm = (def_sub - def_mean) / def_std
            corrs.append(np.sum(ref_norm * def_norm) / (ss * ss))

        if len(corrs) == 3 and corrs[0] < corrs[1] and corrs[2] < corrs[1]:
            denom = 2.0 * (2 * corrs[1] - corrs[0] - corrs[2])
            if abs(denom) > 1e-10:
                return (corrs[0] - corrs[2]) / denom
        return 0.0

# ============================================================
# Strain Computation
# ============================================================
class StrainComputer:
    """Compute strain fields from displacement data"""

    def __init__(self, config):
        self.config = config

    def compute_strains(self, grid_x, grid_y, u_field, v_field, pixel_scale=1.0):
        """
        Compute strain tensor from displacement fields.

        Uses local least-squares plane fitting over a window.

        Returns dict with: exx, eyy, exy, e1, e2, gamma_max, von_mises, theta_p
        """
        ny, nx = u_field.shape

        # Grid spacing (in pixels or mm)
        if nx > 1 and ny > 1:
            dx = (grid_x[0, 1] - grid_x[0, 0]) * pixel_scale
            dy = (grid_y[1, 0] - grid_y[0, 0]) * pixel_scale
        else:
            dx = dy = 1.0

        # Scale displacements
        u = u_field * pixel_scale
        v = v_field * pixel_scale

        # Compute displacement gradients using local least-squares plane fitting.
        # Uses an adaptive window that handles NaN (out-of-ROI) points gracefully:
        # only valid (non-NaN) neighbours are used for the fit, requiring >= 6 points.
        w = self.config.strain_window

        exx = np.full((ny, nx), np.nan)
        eyy = np.full((ny, nx), np.nan)
        exy = np.full((ny, nx), np.nan)

        # Pre-build local coordinate offsets
        wy_range = np.arange(-w, w + 1)
        wx_range = np.arange(-w, w + 1)
        off_x, off_y = np.meshgrid(wx_range * dx, wy_range * dy)
        off_x_flat = off_x.ravel()
        off_y_flat = off_y.ravel()

        for iy in range(ny):
            for ix in range(nx):
                # Skip if this point itself has no displacement
                if np.isnan(u[iy, ix]) or np.isnan(v[iy, ix]):
                    continue

                # Gather neighbours within the window (clamped to grid bounds)
                iy0 = max(0, iy - w)
                iy1 = min(ny, iy + w + 1)
                ix0 = max(0, ix - w)
                ix1 = min(nx, ix + w + 1)

                u_win = u[iy0:iy1, ix0:ix1]
                v_win = v[iy0:iy1, ix0:ix1]

                # Build local coordinate arrays for the (possibly clipped) window
                local_y = np.arange(iy0 - iy, iy1 - iy) * dy
                local_x = np.arange(ix0 - ix, ix1 - ix) * dx
                lx, ly = np.meshgrid(local_x, local_y)

                # Mask: only use valid (non-NaN) points
                valid = np.isfinite(u_win) & np.isfinite(v_win)
                n_valid = np.sum(valid)
                if n_valid < self.config.strain_min_points:  # need minimum points for a robust plane fit
                    continue

                lx_v = lx[valid]
                ly_v = ly[valid]
                u_v = u_win[valid]
                v_v = v_win[valid]

                A = np.column_stack([np.ones(n_valid), lx_v, ly_v])

                try:
                    coeffs_u = np.linalg.lstsq(A, u_v, rcond=None)[0]
                    coeffs_v = np.linalg.lstsq(A, v_v, rcond=None)[0]

                    du_dx = coeffs_u[1]
                    du_dy = coeffs_u[2]
                    dv_dx = coeffs_v[1]
                    dv_dy = coeffs_v[2]

                    if self.config.strain_type == 'green_lagrange':
                        exx[iy, ix] = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
                        eyy[iy, ix] = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
                        exy[iy, ix] = 0.5 * (du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy)
                    else:
                        exx[iy, ix] = du_dx
                        eyy[iy, ix] = dv_dy
                        exy[iy, ix] = 0.5 * (du_dy + dv_dx)
                except:
                    pass

        # Principal strains
        e1 = np.full((ny, nx), np.nan)
        e2 = np.full((ny, nx), np.nan)
        gamma_max = np.full((ny, nx), np.nan)
        von_mises = np.full((ny, nx), np.nan)
        theta_p = np.full((ny, nx), np.nan)

        valid = ~np.isnan(exx)

        # Principal strains from Mohr's circle
        e_avg = 0.5 * (exx[valid] + eyy[valid])
        R = np.sqrt((0.5 * (exx[valid] - eyy[valid]))**2 + exy[valid]**2)

        e1[valid] = e_avg + R
        e2[valid] = e_avg - R
        gamma_max[valid] = 2 * R

        # Von Mises equivalent strain
        von_mises[valid] = np.sqrt(exx[valid]**2 + eyy[valid]**2 -
                                    exx[valid]*eyy[valid] + 3*exy[valid]**2) * (2.0/3.0)**0.5

        # Principal angle
        denom = exx[valid] - eyy[valid]
        theta_p[valid] = np.where(np.abs(denom) > 1e-15,
                                   0.5 * np.arctan2(2*exy[valid], denom) * 180/np.pi,
                                   0.0)

        return {
            'exx': exx, 'eyy': eyy, 'exy': exy,
            'e1': e1, 'e2': e2,
            'gamma_max': gamma_max, 'von_mises': von_mises,
            'theta_p': theta_p
        }

# ============================================================
# Results Container
# ============================================================
class DICResults:
    """Container for DIC analysis results"""

    def __init__(self):
        self.frames = []
        self.grid_x = None
        self.grid_y = None
        self.load_data = None

    def add_frame(self, frame_idx, u, v, corr, strains=None):
        self.frames.append({
            'frame_idx': frame_idx,
            'u': u,
            'v': v,
            'correlation': corr,
            'strains': strains
        })

    def to_json_for_viz(self, processed_frame_indices):
        """Export results as JSON for the HTML visualization"""
        data = {
            'grid_x': self.grid_x.tolist(),
            'grid_y': self.grid_y.tolist(),
            'num_frames': len(self.frames),
            'frame_indices': processed_frame_indices,
            'frames': []
        }

        if self.load_data is not None:
            data['load'] = {
                'time': self.load_data.time.tolist(),
                'load_N': self.load_data.load_N.tolist(),
                'displacement_mm': self.load_data.displacement_mm.tolist(),
                'frame_indices': self.load_data.frame_indices.tolist()
            }

        for f in self.frames:
            frame_data = {
                'frame_idx': int(f['frame_idx']),
                'u': np.where(np.isnan(f['u']), None, f['u']).tolist(),
                'v': np.where(np.isnan(f['v']), None, f['v']).tolist(),
                'correlation': np.where(np.isnan(f['correlation']), None, f['correlation']).tolist(),
            }
            if f['strains'] is not None:
                for key, val in f['strains'].items():
                    frame_data[key] = np.where(np.isnan(val), None, val).tolist()
            data['frames'].append(frame_data)

        return data

# ============================================================
# Main Analysis Pipeline
# ============================================================
class DICAnalyzer:
    """Main DIC analysis coordinator"""

    def __init__(self, data_dir, config=None):
        self.config = config or DICConfig()
        self.data_dir = Path(data_dir)
        self.loader = ImageLoader(data_dir, self.config)
        self.engine = DICEngine(self.config)
        self.strain_computer = StrainComputer(self.config)
        self.results = DICResults()

        # Load CSV data
        csv_file = list(self.data_dir.glob('*.csv'))
        if csv_file:
            self.load_data = LoadData(csv_file[0])
            self.results.load_data = self.load_data
        else:
            self.load_data = None

    def run(self, start_frame=0, end_frame=None, frame_skip=None,
            progress_callback=None, cancel_event=None):
        """
        Run the full DIC analysis pipeline.

        Args:
            progress_callback: Optional callable(frame_idx, current_step, total_steps,
                               elapsed, frame_data_dict) called after each frame.
            cancel_event: Optional threading.Event — if set, stops processing.
        """
        if frame_skip is None:
            frame_skip = self.config.frame_skip

        n_frames = self.loader.get_frame_count()
        if end_frame is None:
            end_frame = n_frames
        end_frame = min(end_frame, n_frames)

        frame_indices = list(range(start_frame, end_frame, frame_skip))
        print(f"\n{'='*60}")
        print(f"  2D DIC Analysis")
        print(f"  Frames: {start_frame} to {end_frame-1} (step {frame_skip})")
        print(f"  Total frames to process: {len(frame_indices)}")
        print(f"  Subset size: {self.config.subset_size}px, Step: {self.config.step_size}px")
        print(f"  Downsample: {self.config.downsample}x")
        print(f"{'='*60}\n")

        # Load reference image (first frame)
        ref_img, _ = self.loader.get_frame_pair(frame_indices[0])
        print(f"Reference image size: {ref_img.shape[1]}x{ref_img.shape[0]} pixels")

        # ── Build grid and polygon ROI mask ──
        h, w = ref_img.shape
        ss = self.config.subset_size
        step = self.config.step_size
        half_ss = ss // 2
        margin = half_ss + 5
        gy = np.arange(margin, h - margin, step)
        gx = np.arange(margin, w - margin, step)
        grid_x, grid_y = np.meshgrid(gx, gy)
        ny, nx = len(gy), len(gx)

        self.results.grid_x = grid_x
        self.results.grid_y = grid_y

        # Compute ROI mask from polygon (if provided)
        roi_mask = None
        if self.config.roi_polygon is not None and len(self.config.roi_polygon) >= 3:
            ds = self.config.downsample
            # Scale polygon vertices from original image coords to downsampled coords
            poly_ds = np.array(self.config.roi_polygon, dtype=np.float64) / ds
            poly_cv = poly_ds.reshape((-1, 1, 2)).astype(np.float32)

            roi_mask = np.zeros((ny, nx), dtype=bool)
            inside_count = 0
            for iy in range(ny):
                for ix in range(nx):
                    pt = (float(grid_x[iy, ix]), float(grid_y[iy, ix]))
                    dist = cv2.pointPolygonTest(poly_cv, pt, False)
                    if dist >= 0:  # inside or on edge
                        roi_mask[iy, ix] = True
                        inside_count += 1
            print(f"ROI polygon: {len(self.config.roi_polygon)} vertices, "
                  f"{inside_count}/{ny*nx} grid points inside ({inside_count*100//(ny*nx)}%)")
        else:
            print("No ROI polygon — processing full image")

        prev_u = None
        prev_v = None
        processed_indices = []

        for i, fidx in enumerate(frame_indices):
            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                print(f"\nAnalysis cancelled at frame {fidx}.")
                break

            t_start = time.time()

            def_img, _ = self.loader.get_frame_pair(fidx)

            if fidx == frame_indices[0]:
                # Reference frame — zero displacement, masked outside ROI
                u_field = np.full((ny, nx), np.nan)
                v_field = np.full((ny, nx), np.nan)
                corr_field = np.full((ny, nx), -1.0)

                if roi_mask is not None:
                    u_field[roi_mask] = 0.0
                    v_field[roi_mask] = 0.0
                    corr_field[roi_mask] = 1.0
                else:
                    u_field[:] = 0.0
                    v_field[:] = 0.0
                    corr_field[:] = 1.0

                strains = {}
                for key in ('exx', 'eyy', 'exy', 'e1', 'e2', 'gamma_max', 'von_mises', 'theta_p'):
                    arr = np.full((ny, nx), np.nan)
                    if roi_mask is not None:
                        arr[roi_mask] = 0.0
                    else:
                        arr[:] = 0.0
                    strains[key] = arr
            else:
                # Correlate — pass roi_mask so only ROI points are computed
                grid_x, grid_y, u_field, v_field, corr_field = \
                    self.engine.compute_displacement_field(
                        ref_img, def_img, prev_u, prev_v, roi_mask=roi_mask)

                # Mask low-correlation points
                bad = corr_field < self.config.correlation_threshold
                u_field[bad] = np.nan
                v_field[bad] = np.nan

                # Optional displacement field filtering before strain computation
                if self.config.displacement_filter != 'none':
                    u_filtered = self._filter_displacement(u_field)
                    v_filtered = self._filter_displacement(v_field)
                else:
                    u_filtered = u_field
                    v_filtered = v_field

                # Compute strains
                strains = self.strain_computer.compute_strains(
                    grid_x, grid_y, u_filtered, v_filtered, pixel_scale=1.0
                )

                prev_u = u_field
                prev_v = v_field

            self.results.add_frame(fidx, u_field, v_field, corr_field, strains)
            processed_indices.append(int(fidx))

            elapsed = time.time() - t_start

            # Progress bar
            pct = (i + 1) / len(frame_indices) * 100
            bar_len = 30
            filled = int(bar_len * (i + 1) / len(frame_indices))
            bar = '█' * filled + '░' * (bar_len - filled)

            u_rng = f"[{np.nanmin(u_field):.2f}, {np.nanmax(u_field):.2f}]" if not np.all(np.isnan(u_field)) else "[N/A]"
            print(f"\r  [{bar}] {pct:5.1f}% | Frame {fidx:>4d} | U range: {u_rng} | {elapsed:.1f}s", end='', flush=True)

            # Fire progress callback with frame results
            if progress_callback:
                frame_data = {
                    'frame_idx': int(fidx),
                    'u': np.where(np.isnan(u_field), None, u_field).tolist(),
                    'v': np.where(np.isnan(v_field), None, v_field).tolist(),
                    'correlation': np.where(np.isnan(corr_field), None, corr_field).tolist(),
                }
                if strains:
                    for key, val in strains.items():
                        frame_data[key] = np.where(np.isnan(val), None, val).tolist()
                progress_callback(fidx, i + 1, len(frame_indices), elapsed, frame_data)

        print(f"\n\nAnalysis complete! {len(processed_indices)} frames processed.")
        return self.results, processed_indices

    def _filter_displacement(self, field):
        """Apply spatial filter to a displacement field, preserving NaN regions."""
        ks = self.config.displacement_filter_size
        if ks < 2:
            return field

        filtered = field.copy()
        valid = np.isfinite(field)

        if self.config.displacement_filter == 'median':
            # Median filter — only on valid sub-regions, row by row/col by col
            from numpy.lib.stride_tricks import as_strided
            half = ks // 2
            ny, nx = field.shape
            for iy in range(ny):
                for ix in range(nx):
                    if not valid[iy, ix]:
                        continue
                    iy0, iy1 = max(0, iy - half), min(ny, iy + half + 1)
                    ix0, ix1 = max(0, ix - half), min(nx, ix + half + 1)
                    patch = field[iy0:iy1, ix0:ix1]
                    v = patch[np.isfinite(patch)]
                    if len(v) >= 3:
                        filtered[iy, ix] = np.median(v)

        elif self.config.displacement_filter == 'gaussian':
            # Simple weighted average with Gaussian-like weights
            half = ks // 2
            ny, nx = field.shape
            # Build small Gaussian kernel
            ax = np.arange(-half, half + 1)
            kx, ky = np.meshgrid(ax, ax)
            kernel = np.exp(-(kx**2 + ky**2) / (2 * (half * 0.6)**2))

            for iy in range(ny):
                for ix in range(nx):
                    if not valid[iy, ix]:
                        continue
                    iy0, iy1 = max(0, iy - half), min(ny, iy + half + 1)
                    ix0, ix1 = max(0, ix - half), min(nx, ix + half + 1)
                    # Get corresponding kernel slice
                    ky0 = iy0 - (iy - half)
                    ky1 = ky0 + (iy1 - iy0)
                    kx0 = ix0 - (ix - half)
                    kx1 = kx0 + (ix1 - ix0)
                    patch = field[iy0:iy1, ix0:ix1]
                    k = kernel[ky0:ky1, kx0:kx1]
                    mask = np.isfinite(patch)
                    if np.sum(mask) >= 3:
                        filtered[iy, ix] = np.sum(patch[mask] * k[mask]) / np.sum(k[mask])

        return filtered

    def get_setup_info(self):
        """Return folder scan info for the dashboard setup screen"""
        import base64
        info = {
            'folder': str(self.data_dir),
            'camera0_count': len(self.loader.camera0_files),
            'camera1_count': len(self.loader.camera1_files),
            'has_csv': self.load_data is not None,
        }
        if self.load_data:
            info['load_range'] = [float(self.load_data.load_N.min()), float(self.load_data.load_N.max())]
            info['disp_range'] = [float(self.load_data.displacement_mm.min()), float(self.load_data.displacement_mm.max())]
            info['total_frames_csv'] = len(self.load_data.frame_indices)

        # Generate preview thumbnail
        if len(self.loader.camera0_files) > 0:
            img = cv2.imread(str(self.loader.camera0_files[0]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                info['image_width'] = img.shape[1]
                info['image_height'] = img.shape[0]
                thumb = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
                _, buf = cv2.imencode('.png', thumb)
                info['preview_b64'] = base64.b64encode(buf).decode('ascii')
        return info

    def get_ref_thumbnail_b64(self, frame_idx=0):
        """Return a base64 PNG thumbnail of a frame"""
        import base64
        img, _ = self.loader.get_frame_pair(frame_idx)
        if img is not None:
            thumb = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            _, buf = cv2.imencode('.png', thumb)
            return base64.b64encode(buf).decode('ascii')
        return None

    def generate_visualization(self, results, processed_indices, output_dir):
        """Generate the interactive HTML visualization dashboard"""
        output_dir = Path(output_dir)

        # Export result data as JSON
        json_data = results.to_json_for_viz(processed_indices)

        # Generate reference image thumbnail for overlay
        ref_img, _ = self.loader.get_frame_pair(processed_indices[0])
        # Encode as base64
        import base64
        _, buf = cv2.imencode('.png', ref_img)
        ref_b64 = base64.b64encode(buf).decode('ascii')

        # Also generate a few key frame thumbnails
        frame_thumbnails = {}
        thumbnail_indices = np.linspace(0, len(processed_indices)-1, min(10, len(processed_indices)), dtype=int)
        for ti in thumbnail_indices:
            fidx = processed_indices[ti]
            img, _ = self.loader.get_frame_pair(fidx)
            # Resize for thumbnail
            thumb = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            _, buf = cv2.imencode('.png', thumb)
            frame_thumbnails[str(fidx)] = base64.b64encode(buf).decode('ascii')

        json_data['ref_image_b64'] = ref_b64
        json_data['frame_thumbnails'] = frame_thumbnails
        json_data['image_width'] = int(ref_img.shape[1])
        json_data['image_height'] = int(ref_img.shape[0])

        # Write JSON data
        json_path = output_dir / 'dic_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        print(f"Results saved to {json_path} ({json_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Generate HTML dashboard
        html_path = output_dir / 'dic_dashboard.html'
        self._write_html_dashboard(html_path, json_data)
        print(f"Dashboard saved to {html_path}")

        # Also generate static matplotlib plots
        self._generate_static_plots(results, processed_indices, output_dir)

        return html_path

    def _generate_static_plots(self, results, processed_indices, output_dir):
        """Generate static matplotlib summary plots"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib import cm

        output_dir = Path(output_dir)

        # Find a frame with good deformation (near end)
        key_frame_idx = len(results.frames) - 1
        f = results.frames[key_frame_idx]
        gx = results.grid_x
        gy = results.grid_y

        variables = {
            'U (Displacement X)': ('u', 'pixels'),
            'V (Displacement Y)': ('v', 'pixels'),
            'Exx (Axial Strain)': ('exx', 'strain'),
            'Eyy (Transverse Strain)': ('eyy', 'strain'),
            'Exy (Shear Strain)': ('exy', 'strain'),
            'E1 (Max Principal)': ('e1', 'strain'),
            'Von Mises Strain': ('von_mises', 'strain'),
        }

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'DIC Analysis Results — Frame {processed_indices[key_frame_idx]}',
                     fontsize=16, fontweight='bold')

        for ax_idx, (title, (key, unit)) in enumerate(variables.items()):
            row, col = divmod(ax_idx, 3)
            ax = axes[row, col]

            if key in ('u', 'v'):
                data = f[key]
            elif f['strains'] is not None and key in f['strains']:
                data = f['strains'][key]
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                ax.set_title(title)
                continue

            data = np.array(data, dtype=float)
            valid = np.isfinite(data)
            if not np.any(valid):
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center')
                ax.set_title(title)
                continue

            vmin = np.nanpercentile(data[valid], 2)
            vmax = np.nanpercentile(data[valid], 98)

            cmap = 'jet' if 'strain' in unit else 'RdBu_r'
            im = ax.pcolormesh(gx, gy, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            plt.colorbar(im, ax=ax, label=unit, shrink=0.8)
            ax.set_title(title, fontsize=11)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')

        # Load-displacement curve
        ax = axes[2, 1]
        if results.load_data is not None:
            ax.plot(results.load_data.displacement_mm, results.load_data.load_N, 'b-', linewidth=1)
            # Mark processed frames
            for pi in processed_indices:
                if pi < len(results.load_data.displacement_mm):
                    ax.plot(results.load_data.displacement_mm[pi], results.load_data.load_N[pi],
                           'ro', markersize=2, alpha=0.5)
            ax.set_xlabel('Displacement (mm)')
            ax.set_ylabel('Load (N)')
            ax.set_title('Load vs Displacement')
            ax.grid(True, alpha=0.3)

        # Correlation quality
        ax = axes[2, 2]
        corr = np.array(f['correlation'], dtype=float)
        valid_corr = corr[np.isfinite(corr)]
        if len(valid_corr) > 0:
            ax.hist(valid_corr, bins=50, color='steelblue', edgecolor='navy', alpha=0.8)
            ax.set_xlabel('Correlation Coefficient')
            ax.set_ylabel('Count')
            ax.set_title('Correlation Quality')
            ax.axvline(0.9, color='red', linestyle='--', label='Threshold')
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'dic_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Static summary plot saved to {output_dir / 'dic_summary.png'}")

    def _write_html_dashboard(self, filepath, json_data):
        """Write the interactive Vic-2D inspired HTML dashboard"""

        # Serialize the data inline (for portability)
        # For large datasets, we'll reference the JSON file
        data_json_str = json.dumps(json_data)

        html = self._get_dashboard_html(data_json_str)

        with open(filepath, 'w') as f:
            f.write(html)

    def _get_dashboard_html(self, data_json_str):
        """Return the full HTML dashboard code"""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DIC Analysis Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; overflow: hidden; height: 100vh; }}

.app {{ display: grid; grid-template-rows: 48px 1fr 200px; grid-template-columns: 260px 1fr 320px; height: 100vh; gap: 1px; background: #0d0d1a; }}

/* Header */
.header {{ grid-column: 1 / -1; background: linear-gradient(90deg, #16213e 0%, #1a1a2e 50%, #16213e 100%); display: flex; align-items: center; padding: 0 20px; border-bottom: 2px solid #e94560; }}
.header h1 {{ font-size: 18px; font-weight: 600; color: #fff; }}
.header h1 span {{ color: #e94560; font-weight: 700; }}
.header .subtitle {{ margin-left: 20px; font-size: 12px; color: #888; }}
.header .frame-info {{ margin-left: auto; font-size: 13px; color: #aaa; }}

/* Sidebar */
.sidebar {{ background: #16213e; padding: 12px; overflow-y: auto; }}
.sidebar h3 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #e94560; margin: 12px 0 8px 0; }}
.sidebar h3:first-child {{ margin-top: 0; }}

.control-group {{ margin-bottom: 8px; }}
.control-group label {{ display: block; font-size: 11px; color: #888; margin-bottom: 3px; }}
.control-group select, .control-group input {{ width: 100%; padding: 6px 8px; background: #0f3460; border: 1px solid #333; color: #e0e0e0; border-radius: 4px; font-size: 12px; }}
.control-group select:focus, .control-group input:focus {{ border-color: #e94560; outline: none; }}

.color-bar {{ width: 100%; height: 20px; border-radius: 3px; margin: 4px 0; }}
.range-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #888; }}

.btn {{ display: inline-block; padding: 6px 12px; background: #e94560; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; margin: 2px; }}
.btn:hover {{ background: #c73e54; }}
.btn.secondary {{ background: #0f3460; border: 1px solid #333; }}
.btn.secondary:hover {{ background: #1a4a7a; }}
.btn.active {{ background: #e94560; box-shadow: 0 0 8px rgba(233,69,96,0.5); }}

/* Main viewport */
.viewport {{ background: #111; position: relative; overflow: hidden; }}
.viewport canvas {{ position: absolute; top: 0; left: 0; }}
#imageCanvas {{ z-index: 1; }}
#contourCanvas {{ z-index: 2; }}
#overlayCanvas {{ z-index: 3; }}

.viewport .tools {{ position: absolute; top: 10px; left: 10px; z-index: 10; display: flex; gap: 4px; }}
.viewport .coordinates {{ position: absolute; bottom: 10px; left: 10px; z-index: 10; background: rgba(0,0,0,0.8); padding: 4px 8px; border-radius: 3px; font-size: 11px; font-family: monospace; }}
.viewport .colorbar-overlay {{ position: absolute; right: 15px; top: 50%; transform: translateY(-50%); z-index: 10; width: 30px; }}
.colorbar-overlay canvas {{ border-radius: 3px; }}
.colorbar-overlay .cb-label {{ font-size: 9px; text-align: center; margin-top: 2px; }}
.colorbar-overlay .cb-range {{ font-size: 9px; color: #aaa; text-align: right; margin-right: 35px; }}

/* Right panel */
.right-panel {{ background: #16213e; padding: 12px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }}
.panel-section {{ background: #0f3460; border-radius: 6px; padding: 10px; }}
.panel-section h4 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #e94560; margin-bottom: 8px; }}
.panel-section canvas {{ width: 100%; border-radius: 3px; }}

/* Statistics */
.stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
.stat-item {{ background: #16213e; padding: 6px; border-radius: 3px; text-align: center; }}
.stat-item .stat-value {{ font-size: 14px; font-weight: 600; color: #fff; }}
.stat-item .stat-label {{ font-size: 9px; color: #888; text-transform: uppercase; }}

/* Timeline */
.timeline {{ grid-column: 1 / -1; background: #16213e; padding: 10px 20px; display: flex; flex-direction: column; gap: 6px; }}
.timeline-controls {{ display: flex; align-items: center; gap: 10px; }}
.timeline-slider {{ flex: 1; -webkit-appearance: none; height: 6px; background: #0f3460; border-radius: 3px; outline: none; }}
.timeline-slider::-webkit-slider-thumb {{ -webkit-appearance: none; width: 16px; height: 16px; background: #e94560; border-radius: 50%; cursor: pointer; }}
.timeline-charts {{ display: flex; gap: 10px; height: 130px; }}
.timeline-charts canvas {{ flex: 1; background: #0d0d1a; border-radius: 4px; }}

/* Extraction line display */
.extraction-display {{ background: #0d0d1a; border-radius: 4px; padding: 8px; }}
.extraction-display canvas {{ width: 100%; height: 120px; }}

/* Inspector tooltip */
.inspector {{ display: none; position: absolute; z-index: 100; background: rgba(15,52,96,0.95); border: 1px solid #e94560; border-radius: 6px; padding: 8px 12px; font-size: 11px; pointer-events: none; min-width: 180px; }}
.inspector .row {{ display: flex; justify-content: space-between; gap: 15px; margin: 2px 0; }}
.inspector .row .label {{ color: #888; }}
.inspector .row .value {{ color: #fff; font-family: monospace; }}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <h1><span>DIC</span> Analysis Dashboard</h1>
    <span class="subtitle">2D Digital Image Correlation &mdash; Inspired by Vic-2D iris</span>
    <span class="frame-info" id="frameInfo">Frame 0 / 0</span>
  </div>

  <div class="sidebar">
    <h3>Display Variable</h3>
    <div class="control-group">
      <select id="variableSelect">
        <option value="u">U (Displacement X)</option>
        <option value="v">V (Displacement Y)</option>
        <option value="displacement_mag">|U| (Displacement Magnitude)</option>
        <option value="exx" selected>Exx (Axial Strain)</option>
        <option value="eyy">Eyy (Transverse Strain)</option>
        <option value="exy">Exy (Shear Strain)</option>
        <option value="e1">E1 (Max Principal Strain)</option>
        <option value="e2">E2 (Min Principal Strain)</option>
        <option value="gamma_max">Gamma Max (Max Shear)</option>
        <option value="von_mises">Von Mises Strain</option>
        <option value="correlation">Correlation Coefficient</option>
      </select>
    </div>

    <h3>Color Map</h3>
    <div class="control-group">
      <select id="colormapSelect">
        <option value="jet" selected>Jet</option>
        <option value="viridis">Viridis</option>
        <option value="plasma">Plasma</option>
        <option value="inferno">Inferno</option>
        <option value="coolwarm">Cool-Warm</option>
        <option value="rdbu">Red-Blue</option>
        <option value="rainbow">Rainbow</option>
        <option value="hot">Hot</option>
      </select>
      <canvas id="colorbarPreview" height="20" style="margin-top:4px;border-radius:3px;"></canvas>
      <div class="range-labels"><span id="rangeMin">0</span><span id="rangeMax">1</span></div>
    </div>

    <h3>Range</h3>
    <div class="control-group">
      <label>Min</label>
      <input type="number" id="rangeMinInput" step="any">
    </div>
    <div class="control-group">
      <label>Max</label>
      <input type="number" id="rangeMaxInput" step="any">
    </div>
    <button class="btn secondary" onclick="autoRange()">Auto Range</button>
    <button class="btn secondary" onclick="symmetricRange()">Symmetric</button>

    <h3>Display Options</h3>
    <div class="control-group">
      <label><input type="checkbox" id="showImage" checked> Show Image</label>
    </div>
    <div class="control-group">
      <label><input type="checkbox" id="showContour" checked> Show Contour</label>
    </div>
    <div class="control-group">
      <label><input type="checkbox" id="showGrid"> Show Grid</label>
    </div>
    <div class="control-group">
      <label>Opacity</label>
      <input type="range" id="contourOpacity" min="0" max="100" value="70" style="width:100%">
    </div>
    <div class="control-group">
      <label>Contour Levels</label>
      <input type="range" id="contourLevels" min="5" max="50" value="20" style="width:100%">
    </div>

    <h3>Playback</h3>
    <div class="control-group">
      <label>Speed: <span id="speedVal">10</span> FPS</label>
      <input type="range" id="speedRange" min="1" max="60" value="10" style="width:100%">
    </div>

    <h3>Tools</h3>
    <div style="display:flex;flex-wrap:wrap;gap:4px;">
      <button class="btn secondary" id="toolInspect" onclick="setTool('inspect')">🔍 Inspect</button>
      <button class="btn secondary" id="toolLine" onclick="setTool('line')">📏 Line</button>
      <button class="btn secondary" id="toolExtensometer" onclick="setTool('extensometer')">📐 Extensometer</button>
      <button class="btn secondary" id="toolPoint" onclick="setTool('point')">📍 Point</button>
    </div>

    <h3>Export</h3>
    <button class="btn" onclick="exportCSV()">📄 Export CSV</button>
    <button class="btn" onclick="exportImage()">🖼️ Export Image</button>
  </div>

  <div class="viewport" id="viewport">
    <canvas id="imageCanvas"></canvas>
    <canvas id="contourCanvas"></canvas>
    <canvas id="overlayCanvas"></canvas>
    <div class="coordinates" id="coordinates">X: — Y: — Val: —</div>
    <div class="inspector" id="inspector"></div>
  </div>

  <div class="right-panel">
    <div class="panel-section">
      <h4>Statistics</h4>
      <div class="stats-grid" id="statsGrid">
        <div class="stat-item"><div class="stat-value" id="statMin">—</div><div class="stat-label">Min</div></div>
        <div class="stat-item"><div class="stat-value" id="statMax">—</div><div class="stat-label">Max</div></div>
        <div class="stat-item"><div class="stat-value" id="statMean">—</div><div class="stat-label">Mean</div></div>
        <div class="stat-item"><div class="stat-value" id="statStd">—</div><div class="stat-label">Std Dev</div></div>
        <div class="stat-item"><div class="stat-value" id="statPoints">—</div><div class="stat-label">Valid Pts</div></div>
        <div class="stat-item"><div class="stat-value" id="statCorr">—</div><div class="stat-label">Avg Corr</div></div>
      </div>
    </div>

    <div class="panel-section">
      <h4>Histogram</h4>
      <canvas id="histogramCanvas" height="120"></canvas>
    </div>

    <div class="panel-section">
      <h4>Line Extraction</h4>
      <canvas id="lineExtractCanvas" height="120"></canvas>
    </div>

    <div class="panel-section">
      <h4>Point Tracking</h4>
      <canvas id="pointTrackCanvas" height="120"></canvas>
    </div>
  </div>

  <div class="timeline">
    <div class="timeline-controls">
      <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
      <button class="btn secondary" onclick="prevFrame()">◀</button>
      <input type="range" class="timeline-slider" id="frameSlider" min="0" max="0" value="0">
      <button class="btn secondary" onclick="nextFrame()">▶</button>
      <span id="frameLabel" style="font-size:12px;min-width:80px;">Frame 0</span>
      <span id="timeLabel" style="font-size:11px;color:#888;min-width:60px;"></span>
    </div>
    <div class="timeline-charts">
      <canvas id="loadDispChart"></canvas>
      <canvas id="strainTimeChart"></canvas>
    </div>
  </div>
</div>

<script>
// ============================================================
// DIC Dashboard Application
// ============================================================

const DATA = ''' + data_json_str + ''';

// State
let state = {{
  currentFrame: 0,
  variable: 'exx',
  colormap: 'jet',
  rangeMin: 0,
  rangeMax: 1,
  autoRange: true,
  showImage: true,
  showContour: true,
  showGrid: false,
  opacity: 0.7,
  contourLevels: 20, // Discretization levels
  fps: 10,           // Playback speed
  tool: 'inspect',
  playing: false,
  zoom: 1,
  panX: 0,
  panY: 0,
  lineStart: null,
  lineEnd: null,
  trackedPoints: [],
  extensometerPts: [],
}};

// Colormaps
const COLORMAPS = {{
  jet: (t) => {{
    let r,g,b;
    if(t<0.125) {{ r=0; g=0; b=0.5+t*4; }}
    else if(t<0.375) {{ r=0; g=(t-0.125)*4; b=1; }}
    else if(t<0.625) {{ r=(t-0.375)*4; g=1; b=1-(t-0.375)*4; }}
    else if(t<0.875) {{ r=1; g=1-(t-0.625)*4; b=0; }}
    else {{ r=1-(t-0.875)*2; g=0; b=0; }}
    return [Math.round(r*255), Math.round(g*255), Math.round(b*255)];
  }},
  viridis: (t) => {{
    const c = [[68,1,84],[72,35,116],[64,67,135],[52,94,141],[41,120,142],[32,144,140],[34,167,132],[68,190,112],[121,209,81],[189,222,38],[253,231,37]];
    const idx = t * (c.length-1);
    const i = Math.min(Math.floor(idx), c.length-2);
    const f = idx - i;
    return [Math.round(c[i][0]*(1-f)+c[i+1][0]*f), Math.round(c[i][1]*(1-f)+c[i+1][1]*f), Math.round(c[i][2]*(1-f)+c[i+1][2]*f)];
  }},
  plasma: (t) => {{
    const c = [[13,8,135],[75,3,161],[126,3,168],[168,34,150],[203,70,121],[231,109,90],[250,155,60],[253,203,44],[240,249,33]];
    const idx = t * (c.length-1);
    const i = Math.min(Math.floor(idx), c.length-2);
    const f = idx - i;
    return [Math.round(c[i][0]*(1-f)+c[i+1][0]*f), Math.round(c[i][1]*(1-f)+c[i+1][1]*f), Math.round(c[i][2]*(1-f)+c[i+1][2]*f)];
  }},
  inferno: (t) => {{
    const c = [[0,0,4],[22,11,57],[66,10,104],[106,23,110],[147,38,103],[186,54,85],[221,81,58],[246,121,24],[252,172,12],[245,228,56],[252,255,164]];
    const idx = t * (c.length-1);
    const i = Math.min(Math.floor(idx), c.length-2);
    const f = idx - i;
    return [Math.round(c[i][0]*(1-f)+c[i+1][0]*f), Math.round(c[i][1]*(1-f)+c[i+1][1]*f), Math.round(c[i][2]*(1-f)+c[i+1][2]*f)];
  }},
  coolwarm: (t) => {{
    const r = t < 0.5 ? Math.round(59 + t*2*200) : 255;
    const g = t < 0.5 ? Math.round(76 + t*2*180) : Math.round(255 - (t-0.5)*2*180);
    const b = t < 0.5 ? 255 : Math.round(255 - (t-0.5)*2*200);
    return [r,g,b];
  }},
  rdbu: (t) => {{
    if(t < 0.5) return [Math.round(33+t*2*200), Math.round(102+t*2*153), Math.round(172+t*2*83)];
    return [Math.round(255-(t-0.5)*2*55), Math.round(255-(t-0.5)*2*195), Math.round(255-(t-0.5)*2*237)];
  }},
  rainbow: (t) => {{
    const h = t * 300;
    const c = 1, x = c * (1 - Math.abs((h/60)%2-1));
    let r=0,g=0,b=0;
    if(h<60) {{r=c;g=x;}} else if(h<120){{r=x;g=c;}} else if(h<180){{g=c;b=x;}}
    else if(h<240){{g=x;b=c;}} else if(h<300){{r=x;b=c;}} else{{r=c;b=x;}}
    return [Math.round(r*255),Math.round(g*255),Math.round(b*255)];
  }},
  hot: (t) => {{
    const r = Math.min(255, Math.round(t*3*255));
    const g = Math.min(255, Math.max(0, Math.round((t-0.333)*3*255)));
    const b = Math.min(255, Math.max(0, Math.round((t-0.666)*3*255)));
    return [r,g,b];
  }}
}};

function getColor(value, vmin, vmax) {{
  if(value === null || value === undefined || isNaN(value)) return [40,40,40,0];
  let t = (value - vmin) / (vmax - vmin);
  t = Math.max(0, Math.min(1, t));

  // Discretize for contour levels if needed
  if(state.contourLevels < 50) {{
    t = Math.floor(t * state.contourLevels) / state.contourLevels;
  }}

  const fn = COLORMAPS[state.colormap] || COLORMAPS.jet;
  return fn(t);
}}

// Get data for current variable and frame
function getCurrentData() {{
  const frame = DATA.frames[state.currentFrame];
  if(!frame) return null;

  if(state.variable === 'displacement_mag') {{
    const u = frame.u, v = frame.v;
    const ny = u.length, nx = u[0].length;
    const mag = [];
    for(let i=0; i<ny; i++) {{
      mag[i] = [];
      for(let j=0; j<nx; j++) {{
        if(u[i][j] !== null && v[i][j] !== null) {{
          mag[i][j] = Math.sqrt(u[i][j]*u[i][j] + v[i][j]*v[i][j]);
        }} else {{ mag[i][j] = null; }}
      }}
    }}
    return mag;
  }}

  return frame[state.variable] || null;
}}

// Compute statistics
function computeStats(data) {{
  if(!data) return {{}};
  const vals = [];
  for(let i=0; i<data.length; i++)
    for(let j=0; j<data[i].length; j++)
      if(data[i][j] !== null && !isNaN(data[i][j])) vals.push(data[i][j]);

  if(vals.length === 0) return {{ min:0, max:0, mean:0, std:0, count:0 }};
  vals.sort((a,b) => a-b);
  const sum = vals.reduce((a,b)=>a+b, 0);
  const mean = sum / vals.length;
  const variance = vals.reduce((a,b) => a + (b-mean)*(b-mean), 0) / vals.length;
  return {{
    min: vals[0],
    max: vals[vals.length-1],
    mean: mean,
    std: Math.sqrt(variance),
    count: vals.length,
    p2: vals[Math.floor(vals.length*0.02)],
    p98: vals[Math.floor(vals.length*0.98)],
    values: vals
  }};
}}

// ============================================================
// Rendering
// ============================================================
function setupCanvases() {{
  const vp = document.getElementById('viewport');
  const rect = vp.getBoundingClientRect();
  const w = rect.width, h = rect.height;

  ['imageCanvas', 'contourCanvas', 'overlayCanvas'].forEach(id => {{
    const c = document.getElementById(id);
    c.width = w;
    c.height = h;
    c.style.width = w + 'px';
    c.style.height = h + 'px';
  }});
}}

function renderImage() {{
  const canvas = document.getElementById('imageCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if(!state.showImage) return;

  // Draw reference image
  const fidx = DATA.frame_indices[state.currentFrame];
  let b64 = DATA.frame_thumbnails[String(fidx)] || DATA.ref_image_b64;

  const img = new Image();
  img.onload = () => {{
    const scale = Math.min(canvas.width / DATA.image_width, canvas.height / DATA.image_height) * state.zoom;
    const iw = DATA.image_width * scale;
    const ih = DATA.image_height * scale;
    const ox = (canvas.width - iw) / 2 + state.panX;
    const oy = (canvas.height - ih) / 2 + state.panY;
    ctx.drawImage(img, ox, oy, iw, ih);
  }};
  img.src = 'data:image/png;base64,' + b64;
}}

function renderContour() {{
  const canvas = document.getElementById('contourCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if(!state.showContour) return;

  const data = getCurrentData();
  if(!data) return;

  const gx = DATA.grid_x;
  const gy = DATA.grid_y;
  const ny = data.length, nx = data[0].length;

  const scale = Math.min(canvas.width / DATA.image_width, canvas.height / DATA.image_height) * state.zoom;
  const ox = (canvas.width - DATA.image_width * scale) / 2 + state.panX;
  const oy = (canvas.height - DATA.image_height * scale) / 2 + state.panY;

  const vmin = state.rangeMin;
  const vmax = state.rangeMax;

  // Create offscreen canvas for contour
  const offCanvas = document.createElement('canvas');
  offCanvas.width = nx;
  offCanvas.height = ny;
  const offCtx = offCanvas.getContext('2d');
  const imgData = offCtx.createImageData(nx, ny);

  for(let iy=0; iy<ny; iy++) {{
    for(let ix=0; ix<nx; ix++) {{
      const idx = (iy * nx + ix) * 4;
      const val = data[iy][ix];
      if(val === null || isNaN(val)) {{
        imgData.data[idx] = 0;
        imgData.data[idx+1] = 0;
        imgData.data[idx+2] = 0;
        imgData.data[idx+3] = 0;
      }} else {{
        const [r,g,b] = getColor(val, vmin, vmax);
        imgData.data[idx] = r;
        imgData.data[idx+1] = g;
        imgData.data[idx+2] = b;
        imgData.data[idx+3] = Math.round(state.opacity * 255);
      }}
    }}
  }}
  offCtx.putImageData(imgData, 0, 0);

  // Map grid coordinates to screen coordinates
  const x0 = ox + gx[0][0] * scale;
  const y0 = oy + gy[0][0] * scale;
  const x1 = ox + gx[0][nx-1] * scale;
  const y1 = oy + gy[ny-1][0] * scale;

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(offCanvas, x0, y0, x1-x0, y1-y0);

  // Grid overlay
  if(state.showGrid) {{
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 0.5;
    for(let iy=0; iy<ny; iy++) {{
      for(let ix=0; ix<nx; ix++) {{
        const sx = ox + gx[iy][ix] * scale;
        const sy = oy + gy[iy][ix] * scale;
        ctx.strokeRect(sx-1, sy-1, 2, 2);
      }}
    }}
  }}
}}

function renderOverlay() {{
  const canvas = document.getElementById('overlayCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scale = Math.min(canvas.width / DATA.image_width, canvas.height / DATA.image_height) * state.zoom;
  const ox = (canvas.width - DATA.image_width * scale) / 2 + state.panX;
  const oy = (canvas.height - DATA.image_height * scale) / 2 + state.panY;

  // Draw line extraction
  if(state.lineStart && state.lineEnd) {{
    ctx.strokeStyle = '#e94560';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.moveTo(state.lineStart.sx, state.lineStart.sy);
    ctx.lineTo(state.lineEnd.sx, state.lineEnd.sy);
    ctx.stroke();
    ctx.setLineDash([]);

    // End markers
    [state.lineStart, state.lineEnd].forEach(p => {{
      ctx.fillStyle = '#e94560';
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, 5, 0, Math.PI*2);
      ctx.fill();
    }});
  }}

  // Draw tracked points
  state.trackedPoints.forEach((p, i) => {{
    const sx = ox + p.gx * scale;
    const sy = oy + p.gy * scale;
    ctx.strokeStyle = ['#00ff88', '#ff8800', '#00aaff', '#ff00aa'][i % 4];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx-8, sy); ctx.lineTo(sx+8, sy);
    ctx.moveTo(sx, sy-8); ctx.lineTo(sx, sy+8);
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = '10px monospace';
    ctx.fillText('P' + (i+1), sx+10, sy-5);
  }});

  // Extensometer
  if(state.extensometerPts.length === 2) {{
    const p1 = state.extensometerPts[0], p2 = state.extensometerPts[1];
    const sx1 = ox + p1.gx * scale, sy1 = oy + p1.gy * scale;
    const sx2 = ox + p2.gx * scale, sy2 = oy + p2.gy * scale;
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx1, sy1);
    ctx.lineTo(sx2, sy2);
    ctx.stroke();

    // Show gauge length
    const frame = DATA.frames[state.currentFrame];
    if(frame) {{
      const dist0 = Math.sqrt((p2.gx-p1.gx)**2 + (p2.gy-p1.gy)**2);
      let u1=0,v1=0,u2=0,v2=0;
      if(frame.u && p1.iy < frame.u.length && p1.ix < frame.u[0].length) {{
        u1 = frame.u[p1.iy][p1.ix] || 0;
        v1 = frame.v[p1.iy][p1.ix] || 0;
        u2 = frame.u[p2.iy][p2.ix] || 0;
        v2 = frame.v[p2.iy][p2.ix] || 0;
      }}
      const dx = (p2.gx+u2) - (p1.gx+u1);
      const dy = (p2.gy+v2) - (p1.gy+v1);
      const dist = Math.sqrt(dx*dx + dy*dy);
      const strain = (dist - dist0) / dist0;

      ctx.fillStyle = '#00ff88';
      ctx.font = '12px monospace';
      const mx = (sx1+sx2)/2, my = (sy1+sy2)/2;
      ctx.fillText('L=' + dist.toFixed(1) + 'px  e=' + (strain*100).toFixed(3) + '%', mx+10, my-10);
    }}
  }}
}}

function render() {{
  renderImage();
  renderContour();
  renderOverlay();
  updateStats();
  updateHistogram();
  updateColorbarPreview();
  updateFrameInfo();
}}

function updateFrameInfo() {{
  const fidx = DATA.frame_indices[state.currentFrame];
  document.getElementById('frameInfo').textContent = `Frame ${{fidx}} (${{state.currentFrame+1}}/${{DATA.num_frames}})`;
  document.getElementById('frameLabel').textContent = `Frame ${{fidx}}`;
  document.getElementById('frameSlider').value = state.currentFrame;

  if(DATA.load) {{
    const idx = DATA.load.frame_indices.indexOf(fidx);
    if(idx >= 0) {{
      document.getElementById('timeLabel').textContent =
        `${{DATA.load.time[idx].toFixed(1)}}s | ${{DATA.load.load_N[idx].toFixed(0)}}N`;
    }}
  }}
}}

function updateStats() {{
  const data = getCurrentData();
  const stats = computeStats(data);

  const fmt = (v) => {{
    if(v === undefined) return '—';
    if(Math.abs(v) < 0.01 && v !== 0) return v.toExponential(3);
    return v.toFixed(4);
  }};

  document.getElementById('statMin').textContent = fmt(stats.min);
  document.getElementById('statMax').textContent = fmt(stats.max);
  document.getElementById('statMean').textContent = fmt(stats.mean);
  document.getElementById('statStd').textContent = fmt(stats.std);
  document.getElementById('statPoints').textContent = stats.count || '—';

  // Average correlation
  const frame = DATA.frames[state.currentFrame];
  if(frame && frame.correlation) {{
    const corrStats = computeStats(frame.correlation);
    document.getElementById('statCorr').textContent = corrStats.mean ? corrStats.mean.toFixed(4) : '—';
  }}
}}

function updateHistogram() {{
  const canvas = document.getElementById('histogramCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth - 20;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const data = getCurrentData();
  const stats = computeStats(data);
  if(!stats.values || stats.values.length === 0) return;

  const nBins = 40;
  const bins = new Array(nBins).fill(0);
  const range = stats.max - stats.min || 1;

  stats.values.forEach(v => {{
    const bin = Math.min(nBins-1, Math.floor((v - stats.min) / range * nBins));
    bins[bin]++;
  }});

  const maxBin = Math.max(...bins);
  const bw = canvas.width / nBins;
  const pad = 5;

  bins.forEach((count, i) => {{
    const h = (count / maxBin) * (canvas.height - 2*pad);
    const t = i / nBins;
    const [r,g,b] = getColor(stats.min + t * range, state.rangeMin, state.rangeMax);
    ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
    ctx.fillRect(i * bw, canvas.height - pad - h, bw - 1, h);
  }});

  // Axis labels
  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.fillText(stats.min.toFixed(4), 2, canvas.height - 1);
  ctx.textAlign = 'right';
  ctx.fillText(stats.max.toFixed(4), canvas.width - 2, canvas.height - 1);
  ctx.textAlign = 'left';
}}

function updateColorbarPreview() {{
  const canvas = document.getElementById('colorbarPreview');
  canvas.width = canvas.parentElement.clientWidth - 4;
  canvas.height = 20;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;

  for(let x=0; x<w; x++) {{
    const t = x / w;
    const [r,g,b] = getColor(state.rangeMin + t * (state.rangeMax - state.rangeMin), state.rangeMin, state.rangeMax);
    ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
    ctx.fillRect(x, 0, 1, 20);
  }}

  document.getElementById('rangeMin').textContent = state.rangeMin.toFixed(4);
  document.getElementById('rangeMax').textContent = state.rangeMax.toFixed(4);
}}

function updateLoadDispChart() {{
  const canvas = document.getElementById('loadDispChart');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth / 2;
  canvas.height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if(!DATA.load) {{
    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.fillText('No load data', 20, canvas.height/2);
    return;
  }}

  const pad = {{ top: 15, right: 10, bottom: 25, left: 50 }};
  const w = canvas.width - pad.left - pad.right;
  const h = canvas.height - pad.top - pad.bottom;

  const disp = DATA.load.displacement_mm;
  const load = DATA.load.load_N;
  const dMin = Math.min(...disp), dMax = Math.max(...disp);
  const lMin = Math.min(...load), lMax = Math.max(...load);
  const dRange = dMax - dMin || 1, lRange = lMax - lMin || 1;

  // Grid
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 0.5;
  for(let i=0; i<=4; i++) {{
    const y = pad.top + h * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
  }}

  // Curve
  ctx.strokeStyle = '#4a9eff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for(let i=0; i<disp.length; i++) {{
    const x = pad.left + (disp[i] - dMin) / dRange * w;
    const y = pad.top + h - (load[i] - lMin) / lRange * h;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }}
  ctx.stroke();

  // Current frame marker
  const fidx = DATA.frame_indices[state.currentFrame];
  const li = DATA.load.frame_indices.indexOf(fidx);
  if(li >= 0) {{
    const x = pad.left + (disp[li] - dMin) / dRange * w;
    const y = pad.top + h - (load[li] - lMin) / lRange * h;
    ctx.fillStyle = '#e94560';
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI*2); ctx.fill();
  }}

  // Labels
  ctx.fillStyle = '#888';
  ctx.font = '9px sans-serif';
  ctx.fillText('Displacement (mm)', pad.left + w/2 - 40, canvas.height - 3);
  ctx.save();
  ctx.translate(10, pad.top + h/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('Load (N)', -20, 0);
  ctx.restore();

  ctx.fillStyle = '#4a9eff';
  ctx.font = '10px sans-serif';
  ctx.fillText('Load vs Displacement', pad.left, 12);
}}

function updateStrainTimeChart() {{
  const canvas = document.getElementById('strainTimeChart');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth / 2;
  canvas.height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const pad = {{ top: 15, right: 10, bottom: 25, left: 50 }};
  const w = canvas.width - pad.left - pad.right;
  const h = canvas.height - pad.top - pad.bottom;

  // Compute mean value over time for current variable
  const means = [];
  const maxes = [];
  for(let fi=0; fi<DATA.frames.length; fi++) {{
    const frame = DATA.frames[fi];
    let d;
    if(state.variable === 'displacement_mag') {{
      d = [];
      if(frame.u) {{
        for(let i=0;i<frame.u.length;i++)
          for(let j=0;j<frame.u[i].length;j++)
            if(frame.u[i][j]!==null && frame.v[i][j]!==null)
              d.push(Math.sqrt(frame.u[i][j]**2+frame.v[i][j]**2));
      }}
    }} else {{
      d = [];
      const fd = frame[state.variable];
      if(fd) {{
        for(let i=0;i<fd.length;i++)
          for(let j=0;j<fd[i].length;j++)
            if(fd[i][j]!==null && !isNaN(fd[i][j])) d.push(fd[i][j]);
      }}
    }}
    if(d.length > 0) {{
      const m = d.reduce((a,b)=>a+b,0)/d.length;
      means.push(m);
      maxes.push(Math.max(...d));
    }} else {{
      means.push(0);
      maxes.push(0);
    }}
  }}

  if(means.length === 0) return;

  const yMin = Math.min(...means, ...maxes);
  const yMax = Math.max(...means, ...maxes);
  const yRange = yMax - yMin || 1;

  // Grid
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 0.5;
  for(let i=0; i<=4; i++) {{
    const y = pad.top + h * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
  }}

  // Max line
  ctx.strokeStyle = 'rgba(233,69,96,0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for(let i=0; i<maxes.length; i++) {{
    const x = pad.left + i / (maxes.length-1) * w;
    const y = pad.top + h - (maxes[i] - yMin) / yRange * h;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }}
  ctx.stroke();

  // Mean line
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for(let i=0; i<means.length; i++) {{
    const x = pad.left + i / (means.length-1) * w;
    const y = pad.top + h - (means[i] - yMin) / yRange * h;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }}
  ctx.stroke();

  // Current frame marker
  const x = pad.left + state.currentFrame / (means.length-1) * w;
  ctx.strokeStyle = '#e94560';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + h); ctx.stroke();

  ctx.fillStyle = '#00ff88';
  ctx.font = '10px sans-serif';
  ctx.fillText(state.variable + ' over time (mean/max)', pad.left, 12);

  ctx.fillStyle = '#888';
  ctx.font = '9px sans-serif';
  ctx.fillText('Frame', pad.left + w/2 - 15, canvas.height - 3);
}}

function updateLineExtraction() {{
  const canvas = document.getElementById('lineExtractCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth - 20;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if(!state.lineStart || !state.lineEnd) {{
    ctx.fillStyle = '#666';
    ctx.font = '11px sans-serif';
    ctx.fillText('Use Line tool to extract data', 10, canvas.height/2);
    return;
  }}

  // Extract data along line
  const data = getCurrentData();
  if(!data) return;

  const gx = DATA.grid_x, gy = DATA.grid_y;
  const ny = data.length, nx = data[0].length;

  const nSamples = 100;
  const lineData = [];

  for(let s=0; s<nSamples; s++) {{
    const t = s / (nSamples-1);
    const px = state.lineStart.gx + t * (state.lineEnd.gx - state.lineStart.gx);
    const py = state.lineStart.gy + t * (state.lineEnd.gy - state.lineStart.gy);

    // Find nearest grid point
    let bestDist = Infinity, bestVal = null;
    for(let iy=0; iy<ny; iy++) {{
      for(let ix=0; ix<nx; ix++) {{
        const dx = gx[iy][ix] - px;
        const dy = gy[iy][ix] - py;
        const d = dx*dx + dy*dy;
        if(d < bestDist) {{ bestDist = d; bestVal = data[iy][ix]; }}
      }}
    }}

    const dist = Math.sqrt((px-state.lineStart.gx)**2 + (py-state.lineStart.gy)**2);
    if(bestVal !== null && !isNaN(bestVal)) lineData.push({{dist, val: bestVal}});
  }}

  if(lineData.length === 0) return;

  const pad = {{ top: 10, right: 10, bottom: 20, left: 45 }};
  const w = canvas.width - pad.left - pad.right;
  const h = canvas.height - pad.top - pad.bottom;

  const dMax = Math.max(...lineData.map(d=>d.dist));
  const vMin = Math.min(...lineData.map(d=>d.val));
  const vMax = Math.max(...lineData.map(d=>d.val));
  const vRange = vMax - vMin || 1;

  ctx.strokeStyle = '#e94560';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  lineData.forEach((d, i) => {{
    const x = pad.left + d.dist / dMax * w;
    const y = pad.top + h - (d.val - vMin) / vRange * h;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }});
  ctx.stroke();

  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.fillText(vMin.toFixed(4), 2, pad.top + h);
  ctx.fillText(vMax.toFixed(4), 2, pad.top + 8);
  ctx.fillText('Distance (px)', pad.left + w/2 - 30, canvas.height - 2);
}}

function updatePointTracking() {{
  const canvas = document.getElementById('pointTrackCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth - 20;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if(state.trackedPoints.length === 0) {{
    ctx.fillStyle = '#666';
    ctx.font = '11px sans-serif';
    ctx.fillText('Use Point tool to track values', 10, canvas.height/2);
    return;
  }}

  const pad = {{ top: 10, right: 10, bottom: 20, left: 45 }};
  const w = canvas.width - pad.left - pad.right;
  const h = canvas.height - pad.top - pad.bottom;

  const colors = ['#00ff88', '#ff8800', '#00aaff', '#ff00aa'];

  // Extract values over all frames for each tracked point
  let allVals = [];
  const traces = state.trackedPoints.map((pt, pi) => {{
    const vals = [];
    for(let fi=0; fi<DATA.frames.length; fi++) {{
      const fd = DATA.frames[fi][state.variable];
      if(fd && pt.iy < fd.length && pt.ix < fd[0].length && fd[pt.iy][pt.ix] !== null) {{
        vals.push(fd[pt.iy][pt.ix]);
        allVals.push(fd[pt.iy][pt.ix]);
      }} else {{
        vals.push(null);
      }}
    }}
    return vals;
  }});

  if(allVals.length === 0) return;
  const vMin = Math.min(...allVals);
  const vMax = Math.max(...allVals);
  const vRange = vMax - vMin || 1;

  traces.forEach((vals, pi) => {{
    ctx.strokeStyle = colors[pi % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    vals.forEach((v, fi) => {{
      if(v === null) {{ started = false; return; }}
      const x = pad.left + fi / (vals.length-1) * w;
      const y = pad.top + h - (v - vMin) / vRange * h;
      if(!started) {{ ctx.moveTo(x,y); started = true; }} else ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }});

  // Current frame marker
  const x = pad.left + state.currentFrame / (DATA.frames.length-1) * w;
  ctx.strokeStyle = 'rgba(255,255,255,0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + h); ctx.stroke();

  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.fillText(vMin.toFixed(4), 2, pad.top + h);
  ctx.fillText(vMax.toFixed(4), 2, pad.top + 8);
}}

// ============================================================
// Interaction
// ============================================================

function screenToGrid(sx, sy) {{
  const canvas = document.getElementById('contourCanvas');
  const scale = Math.min(canvas.width / DATA.image_width, canvas.height / DATA.image_height) * state.zoom;
  const ox = (canvas.width - DATA.image_width * scale) / 2 + state.panX;
  const oy = (canvas.height - DATA.image_height * scale) / 2 + state.panY;

  const gx = (sx - ox) / scale;
  const gy = (sy - oy) / scale;
  return {{ gx, gy }};
}}

function findNearestGridPoint(gx, gy) {{
  const gridX = DATA.grid_x, gridY = DATA.grid_y;
  const ny = gridX.length, nx = gridX[0].length;

  let bestDist = Infinity, bestIx = 0, bestIy = 0;
  for(let iy=0; iy<ny; iy++) {{
    for(let ix=0; ix<nx; ix++) {{
      const dx = gridX[iy][ix] - gx;
      const dy = gridY[iy][ix] - gy;
      const d = dx*dx + dy*dy;
      if(d < bestDist) {{ bestDist = d; bestIx = ix; bestIy = iy; }}
    }}
  }}
  return {{ ix: bestIx, iy: bestIy, gx: gridX[bestIy][bestIx], gy: gridY[bestIy][bestIx] }};
}}

document.getElementById('overlayCanvas').addEventListener('mousemove', (e) => {{
  const rect = e.target.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const {{ gx, gy }} = screenToGrid(sx, sy);
  const nearest = findNearestGridPoint(gx, gy);

  const data = getCurrentData();
  let val = '—';
  if(data && nearest.iy < data.length && nearest.ix < data[0].length) {{
    const v = data[nearest.iy][nearest.ix];
    if(v !== null && !isNaN(v)) val = v.toFixed(6);
  }}

  document.getElementById('coordinates').textContent =
    `X: ${{nearest.gx.toFixed(1)}}  Y: ${{nearest.gy.toFixed(1)}}  ${{state.variable}}: ${{val}}`;

  // Inspector tooltip
  if(state.tool === 'inspect') {{
    const inspector = document.getElementById('inspector');
    const frame = DATA.frames[state.currentFrame];
    if(frame && nearest.iy < (frame.u||[]).length) {{
      const vars = ['u','v','exx','eyy','exy','e1','e2','von_mises','correlation'];
      let html = '';
      vars.forEach(vr => {{
        const fd = frame[vr];
        if(fd && nearest.iy < fd.length && nearest.ix < fd[0].length) {{
          const v = fd[nearest.iy][nearest.ix];
          html += `<div class="row"><span class="label">${{vr}}</span><span class="value">${{v!==null ? v.toFixed(6) : 'N/A'}}</span></div>`;
        }}
      }});
      inspector.innerHTML = html;
      inspector.style.display = 'block';
      inspector.style.left = (sx + 15) + 'px';
      inspector.style.top = (sy - 10) + 'px';
    }}
  }}
}});

document.getElementById('overlayCanvas').addEventListener('mouseleave', () => {{
  document.getElementById('inspector').style.display = 'none';
}});

document.getElementById('overlayCanvas').addEventListener('click', (e) => {{
  const rect = e.target.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const {{ gx, gy }} = screenToGrid(sx, sy);
  const nearest = findNearestGridPoint(gx, gy);

  if(state.tool === 'line') {{
    if(!state.lineStart) {{
      state.lineStart = {{ ...nearest, sx, sy }};
    }} else {{
      state.lineEnd = {{ ...nearest, sx, sy }};
      renderOverlay();
      updateLineExtraction();
    }}
  }} else if(state.tool === 'point') {{
    if(state.trackedPoints.length < 4) {{
      state.trackedPoints.push(nearest);
      renderOverlay();
      updatePointTracking();
    }}
  }} else if(state.tool === 'extensometer') {{
    if(state.extensometerPts.length < 2) {{
      state.extensometerPts.push(nearest);
      renderOverlay();
    }}
  }}
}});

// Zoom with scroll
document.getElementById('overlayCanvas').addEventListener('wheel', (e) => {{
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  state.zoom *= delta;
  state.zoom = Math.max(0.1, Math.min(10, state.zoom));
  render();
}});

// Pan with middle mouse
let isPanning = false, panStartX, panStartY;
document.getElementById('overlayCanvas').addEventListener('mousedown', (e) => {{
  if(e.button === 1) {{ // middle click
    isPanning = true;
    panStartX = e.clientX - state.panX;
    panStartY = e.clientY - state.panY;
    e.preventDefault();
  }}
}});
document.addEventListener('mousemove', (e) => {{
  if(isPanning) {{
    state.panX = e.clientX - panStartX;
    state.panY = e.clientY - panStartY;
    render();
  }}
}});
document.addEventListener('mouseup', () => {{ isPanning = false; }});

// Controls
document.getElementById('variableSelect').addEventListener('change', (e) => {{
  state.variable = e.target.value;
  if(state.autoRange) autoRange();
  render();
  updateStrainTimeChart();
  updatePointTracking();
  updateLineExtraction();
}});

document.getElementById('colormapSelect').addEventListener('change', (e) => {{
  state.colormap = e.target.value;
  render();
}});

document.getElementById('rangeMinInput').addEventListener('change', (e) => {{
  state.rangeMin = parseFloat(e.target.value);
  state.autoRange = false;
  render();
}});
document.getElementById('rangeMaxInput').addEventListener('change', (e) => {{
  state.rangeMax = parseFloat(e.target.value);
  state.autoRange = false;
  render();
}});

document.getElementById('showImage').addEventListener('change', (e) => {{
  state.showImage = e.target.checked;
  render();
}});
document.getElementById('showContour').addEventListener('change', (e) => {{
  state.showContour = e.target.checked;
  render();
}});
document.getElementById('showGrid').addEventListener('change', (e) => {{
  state.showGrid = e.target.checked;
  render();
}});
document.getElementById('contourOpacity').addEventListener('input', (e) => {{
  state.opacity = e.target.value / 100;
  render();
}});
document.getElementById('contourLevels').addEventListener('input', (e) => {{
  state.contourLevels = parseInt(e.target.value);
  render();
}});

// Speed control
document.getElementById('speedRange').addEventListener('input', (e) => {{
  state.fps = parseInt(e.target.value);
  document.getElementById('speedVal').textContent = state.fps;
}});

document.getElementById('frameSlider').addEventListener('input', (e) => {{
  state.currentFrame = parseInt(e.target.value);
  render();
  updateLoadDispChart();
  updateStrainTimeChart();
  updateLineExtraction();
  updatePointTracking();
}});

function setTool(tool) {{
  state.tool = tool;
  document.querySelectorAll('.sidebar .btn.secondary').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('tool' + tool.charAt(0).toUpperCase() + tool.slice(1));
  if(btn) btn.classList.add('active');

  // Reset line/points for new tool selection
  if(tool === 'line') {{ state.lineStart = null; state.lineEnd = null; }}
  if(tool === 'point') {{ state.trackedPoints = []; }}
  if(tool === 'extensometer') {{ state.extensometerPts = []; }}
  renderOverlay();
}}

function autoRange() {{
  const data = getCurrentData();
  const stats = computeStats(data);
  if(stats.count > 0) {{
    state.rangeMin = stats.p2;
    state.rangeMax = stats.p98;
    state.autoRange = true;
    document.getElementById('rangeMinInput').value = state.rangeMin.toFixed(6);
    document.getElementById('rangeMaxInput').value = state.rangeMax.toFixed(6);
    render();
  }}
}}

function symmetricRange() {{
  const absMax = Math.max(Math.abs(state.rangeMin), Math.abs(state.rangeMax));
  state.rangeMin = -absMax;
  state.rangeMax = absMax;
  document.getElementById('rangeMinInput').value = state.rangeMin.toFixed(6);
  document.getElementById('rangeMaxInput').value = state.rangeMax.toFixed(6);
  render();
}}

function togglePlay() {{
  state.playing = !state.playing;
  document.getElementById('playBtn').textContent = state.playing ? '⏸ Pause' : '▶ Play';
  if(state.playing) playLoop();
}}

function playLoop() {{
  if(!state.playing) return;
  nextFrame();
  requestAnimationFrame(() => setTimeout(playLoop, 1000 / state.fps));
}}

function nextFrame() {{
  state.currentFrame = Math.min(DATA.num_frames - 1, state.currentFrame + 1);
  if(state.autoRange) autoRange(); else render();
  updateLoadDispChart();
  updateStrainTimeChart();
  updateLineExtraction();
  updatePointTracking();
}}

function prevFrame() {{
  state.currentFrame = Math.max(0, state.currentFrame - 1);
  if(state.autoRange) autoRange(); else render();
  updateLoadDispChart();
  updateStrainTimeChart();
  updateLineExtraction();
  updatePointTracking();
}}

function exportCSV() {{
  const data = getCurrentData();
  if(!data) return;
  const gx = DATA.grid_x, gy = DATA.grid_y;
  let csv = 'X,Y,' + state.variable + '\\n';
  for(let iy=0; iy<data.length; iy++)
    for(let ix=0; ix<data[0].length; ix++)
      if(data[iy][ix] !== null)
        csv += gx[iy][ix] + ',' + gy[iy][ix] + ',' + data[iy][ix] + '\\n';

  const blob = new Blob([csv], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'dic_' + state.variable + '_frame' + DATA.frame_indices[state.currentFrame] + '.csv';
  a.click();
}}

function exportImage() {{
  const contourCanvas = document.getElementById('contourCanvas');
  const imageCanvas = document.getElementById('imageCanvas');

  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = contourCanvas.width;
  exportCanvas.height = contourCanvas.height;
  const ctx = exportCanvas.getContext('2d');
  ctx.drawImage(imageCanvas, 0, 0);
  ctx.drawImage(contourCanvas, 0, 0);

  const a = document.createElement('a');
  a.href = exportCanvas.toDataURL('image/png');
  a.download = 'dic_' + state.variable + '_frame' + DATA.frame_indices[state.currentFrame] + '.png';
  a.click();
}}

// ============================================================
// Initialize
// ============================================================
window.addEventListener('load', () => {{
  setupCanvases();
  document.getElementById('frameSlider').max = DATA.num_frames - 1;

  autoRange();
  render();
  updateLoadDispChart();
  updateStrainTimeChart();
}});

window.addEventListener('resize', () => {{
  setupCanvases();
  render();
  updateLoadDispChart();
  updateStrainTimeChart();
}});
</script>
</body>
</html>'''


# ============================================================
# Entry Point
# ============================================================
if __name__ == '__main__':
    DATA_DIR = '/sessions/adoring-vigilant-cori/mnt/M300-F1'
    OUTPUT_DIR = '/sessions/adoring-vigilant-cori/mnt/M300-F1'

    # Configure using presets
    # Uncomment the preset you wish to use:
    config = DICConfig.from_preset('standard')
    # config = DICConfig.from_preset('draft')
    # config = DICConfig.from_preset('high')
    # config = DICConfig.from_preset('ultra')

    # IMPORTANT: The following overrides are commented out to allow the
    # presets above to take effect. Uncomment them only if you wish to
    # manually override specific values from the preset.
    # config.subset_size = 31
    # config.step_size = 15
    # config.downsample = 4
    # config.strain_window = 3

    # Frame skip and optical flow are usually set per-run regardless of fidelity
    config.frame_skip = 10         # Process every 10th frame
    config.use_optical_flow = True

    # Run analysis
    analyzer = DICAnalyzer(DATA_DIR, config)

    # Process a representative set of frames
    total_frames = analyzer.loader.get_frame_count()
    results, processed_indices = analyzer.run(
        start_frame=0,
        end_frame=total_frames,
        frame_skip=max(1, total_frames // 40)  # ~40 frames total
    )

    # Generate visualization
    html_path = analyzer.generate_visualization(results, processed_indices, OUTPUT_DIR)
    print(f"\nDone! Open {html_path} to view the interactive dashboard.")