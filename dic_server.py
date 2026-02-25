#!/usr/bin/env python3
"""
DIC Analysis Server
Serves an interactive dashboard with folder selection, configuration,
background computation with live progress, and full Vic-2D style results.

Usage:
    python dic_server.py [--port 8050]

Then open http://localhost:8050 in your browser.
"""

import http.server
import json
import os
import sys
import threading
import queue
import time
import base64
import urllib.parse
import webbrowser
from pathlib import Path
from io import BytesIO

# Add parent directory so we can import dic_analysis
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Global State
# ============================================================
class ServerState:
    def __init__(self):
        self.analyzer = None
        self.analysis_thread = None
        self.cancel_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.is_running = False
        self.results_json = None
        self.grid_info = None  # {grid_x, grid_y, image_width, image_height}
        self.load_data_json = None
        self.processed_indices = []
        self.ref_thumbnail_b64 = None
        self.frame_thumbnails = {}
        self.lock = threading.Lock()
        # tkinter dialog results
        self.tk_result = None
        self.tk_event = threading.Event()

STATE = ServerState()

# ============================================================
# DIC Analysis Thread
# ============================================================
def run_analysis(data_dir, config_dict):
    """Run DIC analysis in background thread"""
    from dic_analysis import DICConfig, DICAnalyzer
    import numpy as np
    import cv2

    # Use preset if specified, otherwise build from individual params
    preset = config_dict.get('preset', 'custom')
    if preset != 'custom':
        config = DICConfig.from_preset(preset)
    else:
        config = DICConfig()

    config.subset_size = int(config_dict.get('subset_size', config.subset_size))
    config.step_size = int(config_dict.get('step_size', config.step_size))
    config.downsample = int(config_dict.get('downsample', config.downsample))
    config.strain_window = int(config_dict.get('strain_window', config.strain_window))
    config.strain_type = config_dict.get('strain_type', config.strain_type)
    config.use_optical_flow = True

    # Advanced correlation parameters
    config.subpixel_search = int(config_dict.get('subpixel_search', config.subpixel_search))
    config.correlation_threshold = float(config_dict.get('correlation_threshold', config.correlation_threshold))

    # Optical flow parameters
    config.of_levels = int(config_dict.get('of_levels', config.of_levels))
    config.of_winsize = int(config_dict.get('of_winsize', config.of_winsize))
    config.of_iterations = int(config_dict.get('of_iterations', config.of_iterations))
    config.of_poly_n = int(config_dict.get('of_poly_n', config.of_poly_n))
    config.of_poly_sigma = float(config_dict.get('of_poly_sigma', config.of_poly_sigma))

    # Displacement filtering
    config.displacement_filter = config_dict.get('displacement_filter', config.displacement_filter)
    config.displacement_filter_size = int(config_dict.get('displacement_filter_size', config.displacement_filter_size))

    # Strain tuning
    config.strain_min_points = int(config_dict.get('strain_min_points', config.strain_min_points))

    # Polygon ROI — list of [x, y] in original image coordinates
    roi_polygon = config_dict.get('roi_polygon', None)
    if roi_polygon and len(roi_polygon) >= 3:
        config.roi_polygon = [(pt[0], pt[1]) for pt in roi_polygon]
    else:
        config.roi_polygon = None

    frame_skip = int(config_dict.get('frame_skip', 14))
    start_frame = int(config_dict.get('start_frame', 0))
    end_frame_cfg = config_dict.get('end_frame', None)

    STATE.cancel_event.clear()
    STATE.is_running = True
    STATE.results_json = None
    STATE.processed_indices = []

    try:
        analyzer = DICAnalyzer(data_dir, config)
        STATE.analyzer = analyzer

        # Send setup info
        setup_info = analyzer.get_setup_info()
        STATE.progress_queue.put({
            'type': 'setup',
            'data': setup_info
        })

        total_frames = analyzer.loader.get_frame_count()

        # Get reference thumbnail
        ref_b64 = analyzer.get_ref_thumbnail_b64(0)
        STATE.ref_thumbnail_b64 = ref_b64

        def on_progress(frame_idx, step, total, elapsed, frame_data):
            # Add grid info on first frame
            if step == 1 and analyzer.results.grid_x is not None:
                STATE.grid_info = {
                    'grid_x': analyzer.results.grid_x.tolist(),
                    'grid_y': analyzer.results.grid_y.tolist(),
                    'image_width': int(analyzer.loader.camera0_files[0].stat().st_size > 0 and
                                       cv2.imread(str(analyzer.loader.camera0_files[0]),
                                                  cv2.IMREAD_GRAYSCALE).shape[1] //
                                       config.downsample or 750),
                    'image_height': int(cv2.imread(str(analyzer.loader.camera0_files[0]),
                                                    cv2.IMREAD_GRAYSCALE).shape[0] //
                                        config.downsample or 1024),
                }
                # Properly compute image dimensions
                ref_img = cv2.imread(str(analyzer.loader.camera0_files[0]), cv2.IMREAD_GRAYSCALE)
                if ref_img is not None:
                    STATE.grid_info['image_width'] = ref_img.shape[1] // config.downsample
                    STATE.grid_info['image_height'] = ref_img.shape[0] // config.downsample

                STATE.progress_queue.put({
                    'type': 'grid_info',
                    'data': STATE.grid_info
                })

            STATE.processed_indices.append(int(frame_idx))

            # Get load data for this frame if available
            load_info = None
            if analyzer.load_data and frame_idx < len(analyzer.load_data.load_N):
                load_info = {
                    'load_N': float(analyzer.load_data.load_N[frame_idx]),
                    'displacement_mm': float(analyzer.load_data.displacement_mm[frame_idx]),
                    'time': float(analyzer.load_data.time[frame_idx]),
                }

            # Generate thumbnail of THIS frame's actual deformed image
            frame_thumb_b64 = None
            try:
                raw_img = cv2.imread(str(analyzer.loader.camera0_files[frame_idx]),
                                     cv2.IMREAD_GRAYSCALE)
                if raw_img is not None:
                    # Downsample to match the analysis resolution, then halve again for transfer
                    ds = config.downsample
                    thumb = cv2.resize(raw_img,
                                       (raw_img.shape[1] // ds, raw_img.shape[0] // ds),
                                       interpolation=cv2.INTER_AREA)
                    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    frame_thumb_b64 = base64.b64encode(buf).decode('ascii')
            except:
                pass

            STATE.progress_queue.put({
                'type': 'frame_result',
                'frame_idx': int(frame_idx),
                'step': step,
                'total': total,
                'pct': round(step / total * 100, 1),
                'elapsed': round(elapsed, 2),
                'data': frame_data,
                'load': load_info,
                'image_b64': frame_thumb_b64,
            })

        run_end = int(end_frame_cfg) if end_frame_cfg is not None else total_frames
        results, processed_indices = analyzer.run(
            start_frame=start_frame,
            end_frame=run_end,
            frame_skip=frame_skip,
            progress_callback=on_progress,
            cancel_event=STATE.cancel_event,
        )

        # Send load data
        if analyzer.load_data:
            STATE.load_data_json = {
                'time': analyzer.load_data.time.tolist(),
                'load_N': analyzer.load_data.load_N.tolist(),
                'displacement_mm': analyzer.load_data.displacement_mm.tolist(),
                'frame_indices': analyzer.load_data.frame_indices.tolist(),
            }
            STATE.progress_queue.put({
                'type': 'load_data',
                'data': STATE.load_data_json
            })

        # Build final results
        final = {
            'grid_x': results.grid_x.tolist() if results.grid_x is not None else [],
            'grid_y': results.grid_y.tolist() if results.grid_y is not None else [],
            'num_frames': len(results.frames),
            'frame_indices': processed_indices,
            'ref_image_b64': ref_b64,
            'image_width': STATE.grid_info['image_width'] if STATE.grid_info else 750,
            'image_height': STATE.grid_info['image_height'] if STATE.grid_info else 1024,
            'frames': [],
        }
        if analyzer.load_data:
            final['load'] = STATE.load_data_json

        for f in results.frames:
            fd = {
                'frame_idx': int(f['frame_idx']),
                'u': np.where(np.isnan(f['u']), None, f['u']).tolist(),
                'v': np.where(np.isnan(f['v']), None, f['v']).tolist(),
                'correlation': np.where(np.isnan(f['correlation']), None, f['correlation']).tolist(),
            }
            if f['strains']:
                for key, val in f['strains'].items():
                    fd[key] = np.where(np.isnan(val), None, val).tolist()
            final['frames'].append(fd)

        STATE.results_json = final

        STATE.progress_queue.put({'type': 'complete', 'num_frames': len(processed_indices)})

    except Exception as e:
        import traceback
        STATE.progress_queue.put({'type': 'error', 'message': str(e), 'traceback': traceback.format_exc()})
    finally:
        STATE.is_running = False


# ============================================================
# HTTP Request Handler
# ============================================================
class DICRequestHandler(http.server.BaseHTTPRequestHandler):
    """Handle HTTP requests for the DIC dashboard"""

    def log_message(self, format, *args):
        # Quieter logging
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length > 0:
            return json.loads(self.rfile.read(length))
        return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self._send_html(get_dashboard_html())

        elif path == '/api/progress':
            self._handle_sse()

        elif path == '/api/results':
            if STATE.results_json:
                self._send_json(STATE.results_json)
            else:
                self._send_json({'error': 'No results available'}, 404)

        elif path == '/api/status':
            self._send_json({
                'running': STATE.is_running,
                'frames_done': len(STATE.processed_indices),
            })

        elif path.startswith('/api/ref_image'):
            self._handle_ref_image()

        else:
            self.send_error(404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == '/api/browse':
            self._handle_browse()

        elif path == '/api/scan':
            self._handle_scan()

        elif path == '/api/start':
            self._handle_start()

        elif path == '/api/stop':
            self._handle_stop()

        else:
            self.send_error(404)

    def _handle_ref_image(self):
        """Return the reference image (frame 0) as base64 PNG for ROI drawing"""
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        folder = params.get('folder', [''])[0]

        if not folder or not os.path.isdir(folder):
            self._send_json({'error': 'Invalid folder'}, 400)
            return

        import cv2
        tifs = sorted(Path(folder).glob('*_0.tif'))
        if not tifs:
            self._send_json({'error': 'No camera 0 images found'}, 404)
            return

        img = cv2.imread(str(tifs[0]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._send_json({'error': 'Cannot read image'}, 500)
            return

        orig_h, orig_w = img.shape

        # Downsample for display (target ~1500px wide max)
        max_display = 1500
        display_scale = 1.0
        if orig_w > max_display:
            display_scale = max_display / orig_w
            new_w = int(orig_w * display_scale)
            new_h = int(orig_h * display_scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode('.png', img)
        b64 = base64.b64encode(buf).decode('ascii')

        self._send_json({
            'image_b64': b64,
            'orig_width': orig_w,
            'orig_height': orig_h,
            'display_width': img.shape[1],
            'display_height': img.shape[0],
            'display_scale': display_scale,
        })

    def _handle_browse(self):
        """Open native OS folder picker via tkinter"""
        # Signal the main thread to open a dialog
        STATE.tk_result = None
        STATE.tk_event.clear()

        # Post event to tkinter main loop
        def open_dialog():
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            folder = filedialog.askdirectory(title="Select DIC Image Folder")
            root.destroy()
            STATE.tk_result = folder if folder else None
            STATE.tk_event.set()

        t = threading.Thread(target=open_dialog, daemon=True)
        t.start()
        STATE.tk_event.wait(timeout=120)

        if STATE.tk_result:
            self._send_json({'folder': STATE.tk_result})
        else:
            self._send_json({'folder': None, 'cancelled': True})

    def _handle_scan(self):
        """Scan a folder for DIC images"""
        body = self._read_body()
        folder = body.get('folder', '')

        if not folder or not os.path.isdir(folder):
            self._send_json({'error': 'Invalid folder path'}, 400)
            return

        from dic_analysis import DICConfig, DICAnalyzer
        config = DICConfig()
        config.downsample = 1  # don't downsample for scan

        try:
            analyzer = DICAnalyzer(folder, config)
            info = analyzer.get_setup_info()
            self._send_json(info)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    def _handle_start(self):
        """Start DIC analysis in background"""
        if STATE.is_running:
            self._send_json({'error': 'Analysis already running'}, 409)
            return

        body = self._read_body()
        folder = body.get('folder', '')
        config = body.get('config', {})

        if not folder or not os.path.isdir(folder):
            self._send_json({'error': 'Invalid folder'}, 400)
            return

        # Clear previous state
        while not STATE.progress_queue.empty():
            try:
                STATE.progress_queue.get_nowait()
            except queue.Empty:
                break
        STATE.processed_indices = []
        STATE.results_json = None

        STATE.analysis_thread = threading.Thread(
            target=run_analysis, args=(folder, config), daemon=True
        )
        STATE.analysis_thread.start()

        self._send_json({'status': 'started'})

    def _handle_stop(self):
        """Cancel running analysis"""
        if STATE.is_running:
            STATE.cancel_event.set()
            self._send_json({'status': 'cancelling'})
        else:
            self._send_json({'status': 'not_running'})

    def _handle_sse(self):
        """Server-Sent Events stream for live progress"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            while True:
                try:
                    event = STATE.progress_queue.get(timeout=1)
                    data = json.dumps(event)
                    self.wfile.write(f"data: {data}\n\n".encode('utf-8'))
                    self.wfile.flush()

                    if event.get('type') in ('complete', 'error'):
                        break
                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(f": keepalive\n\n".encode('utf-8'))
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


# ============================================================
# Dashboard HTML
# ============================================================
def get_dashboard_html():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DIC Analysis</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; overflow: hidden; }

/* ── SETUP SCREEN ── */
.setup-screen { display:flex; height:100vh; }
.setup-left { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:40px; }
.setup-right { width:380px; background:#16213e; padding:30px; overflow-y:auto; border-left:1px solid #333; }

.logo { font-size:36px; font-weight:700; margin-bottom:8px; }
.logo span { color:#e94560; }
.tagline { color:#888; font-size:14px; margin-bottom:40px; }

.folder-select { text-align:center; width:100%; max-width:500px; }
.folder-btn { padding:14px 32px; background:#e94560; color:white; border:none; border-radius:8px; font-size:16px; cursor:pointer; transition:all 0.2s; }
.folder-btn:hover { background:#c73e54; transform:translateY(-1px); box-shadow:0 4px 15px rgba(233,69,96,0.3); }
.folder-btn:disabled { background:#555; cursor:not-allowed; transform:none; box-shadow:none; }
.folder-path { margin-top:12px; padding:10px 16px; background:#0f3460; border-radius:6px; font-family:monospace; font-size:13px; color:#aaa; word-break:break-all; display:none; }

.scan-info { margin-top:24px; width:100%; max-width:500px; display:none; }
.scan-card { background:#16213e; border-radius:8px; padding:16px; margin-bottom:12px; }
.scan-card h4 { font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#e94560; margin-bottom:8px; }
.scan-stat { display:flex; justify-content:space-between; padding:4px 0; font-size:13px; }
.scan-stat .label { color:#888; }
.scan-stat .value { color:#fff; font-family:monospace; }
.scan-preview { max-width:100%; border-radius:6px; margin-top:8px; opacity:0.8; }

.run-btn { padding:14px 48px; background:#00c853; color:white; border:none; border-radius:8px; font-size:16px; font-weight:600; cursor:pointer; margin-top:20px; transition:all 0.2s; display:none; }
.run-btn:hover { background:#00a844; box-shadow:0 4px 15px rgba(0,200,83,0.3); }
.run-btn:disabled { background:#555; cursor:not-allowed; }

/* ── ROI SCREEN ── */
.roi-screen { display:none; height:100vh; flex-direction:column; }
.roi-header { background:linear-gradient(90deg,#16213e,#1a1a2e,#16213e); padding:10px 20px; display:flex; align-items:center; border-bottom:2px solid #e94560; gap:12px; }
.roi-header h2 { font-size:16px; font-weight:600; }
.roi-header h2 span { color:#e94560; }
.roi-header .roi-instructions { font-size:12px; color:#888; flex:1; }
.roi-toolbar { display:flex; gap:6px; align-items:center; }
.roi-toolbar .btn { padding:8px 16px; }
.roi-confirm-btn { padding:10px 24px !important; background:#00c853 !important; font-weight:600; }
.roi-confirm-btn:hover { background:#00a844 !important; }
.roi-confirm-btn:disabled { background:#555 !important; cursor:not-allowed; }
.roi-viewport { flex:1; position:relative; overflow:hidden; background:#111; }
.roi-viewport canvas { position:absolute; top:0; left:0; cursor:crosshair; }
#roiImageCanvas { z-index:1; }
#roiDrawCanvas { z-index:2; }
.roi-status { position:absolute; bottom:10px; left:10px; z-index:10; background:rgba(0,0,0,0.8); padding:6px 12px; border-radius:4px; font-size:12px; font-family:monospace; }
.roi-zoom-info { position:absolute; top:10px; right:10px; z-index:10; background:rgba(0,0,0,0.7); padding:4px 8px; border-radius:4px; font-size:11px; color:#888; }

/* Config panel */
.config-section { margin-bottom:20px; }
.config-section h3 { font-size:12px; text-transform:uppercase; letter-spacing:1px; color:#e94560; margin-bottom:12px; padding-bottom:6px; border-bottom:1px solid #333; }
.config-row { margin-bottom:12px; }
.config-row label { display:block; font-size:11px; color:#888; margin-bottom:4px; }
.config-row input, .config-row select { width:100%; padding:8px 10px; background:#0f3460; border:1px solid #333; color:#e0e0e0; border-radius:5px; font-size:13px; }
.config-row input:focus, .config-row select:focus { border-color:#e94560; outline:none; }
.config-row .hint { font-size:10px; color:#555; margin-top:2px; }

/* ── ANALYSIS SCREEN ── */
.analysis-screen { display:none; height:100vh; flex-direction:column; }

.analysis-header { background:linear-gradient(90deg,#16213e,#1a1a2e,#16213e); padding:10px 20px; display:flex; align-items:center; border-bottom:2px solid #e94560; }
.analysis-header h1 { font-size:18px; font-weight:600; }
.analysis-header h1 span { color:#e94560; }
.analysis-header .status { margin-left:auto; font-size:13px; color:#888; }

.progress-bar-container { padding:12px 20px; background:#16213e; }
.progress-outer { height:8px; background:#0d0d1a; border-radius:4px; overflow:hidden; }
.progress-inner { height:100%; background:linear-gradient(90deg,#e94560,#ff6b6b); border-radius:4px; transition:width 0.3s; width:0%; }
.progress-info { display:flex; justify-content:space-between; margin-top:6px; font-size:12px; color:#888; }

.stop-btn { padding:6px 16px; background:#e94560; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px; }

/* Main content area */
.main-area { flex:1; display:grid; grid-template-columns:260px 1fr 320px; gap:1px; background:#0d0d1a; overflow:hidden; }

/* Sidebar */
.sidebar { background:#16213e; padding:12px; overflow-y:auto; }
.sidebar h3 { font-size:12px; text-transform:uppercase; letter-spacing:1px; color:#e94560; margin:12px 0 8px; }
.sidebar h3:first-child { margin-top:0; }
.control-group { margin-bottom:8px; }
.control-group label { display:block; font-size:11px; color:#888; margin-bottom:3px; }
.control-group select, .control-group input[type="number"] { width:100%; padding:6px 8px; background:#0f3460; border:1px solid #333; color:#e0e0e0; border-radius:4px; font-size:12px; }
.range-labels { display:flex; justify-content:space-between; font-size:10px; color:#888; }
.btn { display:inline-block; padding:6px 12px; background:#e94560; color:white; border:none; border-radius:4px; cursor:pointer; font-size:12px; margin:2px; }
.btn:hover { background:#c73e54; }
.btn.secondary { background:#0f3460; border:1px solid #333; }
.btn.secondary:hover { background:#1a4a7a; }
.btn.active { background:#e94560; box-shadow:0 0 8px rgba(233,69,96,0.5); }

/* Viewport */
.viewport { background:#111; position:relative; overflow:hidden; }
.viewport canvas { position:absolute; top:0; left:0; }
#imageCanvas { z-index:1; }
#contourCanvas { z-index:2; }
#overlayCanvas { z-index:3; }
.viewport .coordinates { position:absolute; bottom:10px; left:10px; z-index:10; background:rgba(0,0,0,0.8); padding:4px 8px; border-radius:3px; font-size:11px; font-family:monospace; }
.inspector { display:none; position:absolute; z-index:100; background:rgba(15,52,96,0.95); border:1px solid #e94560; border-radius:6px; padding:8px 12px; font-size:11px; pointer-events:none; min-width:180px; }
.inspector .row { display:flex; justify-content:space-between; gap:15px; margin:2px 0; }
.inspector .row .label { color:#888; }
.inspector .row .value { color:#fff; font-family:monospace; }

/* Right panel */
.right-panel { background:#16213e; padding:12px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }
.panel-section { background:#0f3460; border-radius:6px; padding:10px; }
.panel-section h4 { font-size:11px; text-transform:uppercase; letter-spacing:0.5px; color:#e94560; margin-bottom:8px; }
.panel-section canvas { width:100%; border-radius:3px; }
.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:4px; }
.stat-item { background:#16213e; padding:6px; border-radius:3px; text-align:center; }
.stat-item .stat-value { font-size:14px; font-weight:600; color:#fff; }
.stat-item .stat-label { font-size:9px; color:#888; text-transform:uppercase; }

/* Timeline */
.timeline { background:#16213e; padding:10px 20px; display:flex; flex-direction:column; gap:6px; }
.timeline-controls { display:flex; align-items:center; gap:10px; }
.timeline-slider { flex:1; -webkit-appearance:none; height:6px; background:#0f3460; border-radius:3px; outline:none; }
.timeline-slider::-webkit-slider-thumb { -webkit-appearance:none; width:16px; height:16px; background:#e94560; border-radius:50%; cursor:pointer; }
.timeline-charts { display:flex; gap:10px; height:120px; }
.timeline-charts canvas { flex:1; background:#0d0d1a; border-radius:4px; }

/* Loading overlay */
.loading-overlay { position:absolute; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.7); display:flex; align-items:center; justify-content:center; z-index:50; }
.loading-overlay .spinner { width:40px; height:40px; border:3px solid #333; border-top-color:#e94560; border-radius:50%; animation:spin 1s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>

<!-- ═══════════════ SETUP SCREEN ═══════════════ -->
<div class="setup-screen" id="setupScreen">
  <div class="setup-left">
    <div class="logo"><span>DIC</span> Analysis</div>
    <div class="tagline">2D Digital Image Correlation — Select a folder to begin</div>

    <div class="folder-select">
      <button class="folder-btn" id="browseBtn" onclick="browseFolder()">📂 Select Image Folder</button>
      <div class="folder-path" id="folderPath"></div>
    </div>

    <div class="scan-info" id="scanInfo">
      <div class="scan-card">
        <h4>Dataset Summary</h4>
        <div class="scan-stat"><span class="label">Camera 0 images</span><span class="value" id="cam0Count">—</span></div>
        <div class="scan-stat"><span class="label">Camera 1 images</span><span class="value" id="cam1Count">—</span></div>
        <div class="scan-stat"><span class="label">Image dimensions</span><span class="value" id="imgDims">—</span></div>
        <div class="scan-stat"><span class="label">Load/disp CSV</span><span class="value" id="csvStatus">—</span></div>
        <div class="scan-stat" id="loadRangeRow" style="display:none"><span class="label">Load range</span><span class="value" id="loadRange">—</span></div>
        <div class="scan-stat" id="dispRangeRow" style="display:none"><span class="label">Displacement range</span><span class="value" id="dispRange">—</span></div>
        <img id="previewImg" class="scan-preview" style="display:none">
      </div>
      <button class="run-btn" id="runBtn" onclick="startAnalysis()">▶ Run Analysis</button>
    </div>
  </div>

  <div class="setup-right">
    <div class="config-section">
      <h3>Fidelity Preset</h3>
      <div class="config-row">
        <select id="cfgPreset" onchange="applyPreset(this.value)">
          <option value="draft">Draft (fastest)</option>
          <option value="standard" selected>Standard</option>
          <option value="high">High Quality</option>
          <option value="ultra">Ultra (slowest)</option>
          <option value="custom">Custom</option>
        </select>
        <div class="hint">Presets auto-fill all parameters below. Switch to Custom to tweak individually.</div>
      </div>
    </div>

    <div class="config-section">
      <h3>Correlation Parameters</h3>
      <div class="config-row">
        <label>Subset Size (px)</label>
        <input type="number" id="cfgSubsetSize" value="31" min="11" max="101" step="2">
        <div class="hint">Odd number, 21–51 typical. Larger = more robust, slower.</div>
      </div>
      <div class="config-row">
        <label>Step Size (px)</label>
        <input type="number" id="cfgStepSize" value="15" min="3" max="50">
        <div class="hint">Grid spacing. Smaller = denser field, slower.</div>
      </div>
      <div class="config-row">
        <label>Downsample Factor</label>
        <input type="number" id="cfgDownsample" value="4" min="1" max="8">
        <div class="hint">1 = full res, 4 = quarter res. Higher = faster.</div>
      </div>
      <div class="config-row">
        <label>Sub-pixel Search (±px)</label>
        <input type="number" id="cfgSubpixelSearch" value="3" min="1" max="10">
        <div class="hint">Integer-pixel search radius around optical flow estimate.</div>
      </div>
      <div class="config-row">
        <label>Correlation Threshold</label>
        <input type="number" id="cfgCorrThreshold" value="0.5" min="0" max="1" step="0.05">
        <div class="hint">Minimum ZNCC to accept a match. Higher = stricter.</div>
      </div>
    </div>

    <div class="config-section">
      <h3>Optical Flow</h3>
      <div class="config-row">
        <label>Pyramid Levels</label>
        <input type="number" id="cfgOfLevels" value="5" min="1" max="10">
        <div class="hint">More levels = handles larger displacements.</div>
      </div>
      <div class="config-row">
        <label>Window Size</label>
        <input type="number" id="cfgOfWinsize" value="31" min="5" max="101" step="2">
        <div class="hint">Averaging window. Larger = smoother.</div>
      </div>
      <div class="config-row">
        <label>Iterations</label>
        <input type="number" id="cfgOfIterations" value="5" min="1" max="20">
        <div class="hint">Per-level iterations. More = slower but better convergence.</div>
      </div>
    </div>

    <div class="config-section">
      <h3>Strain Computation</h3>
      <div class="config-row">
        <label>Strain Window (grid pts)</label>
        <input type="number" id="cfgStrainWindow" value="3" min="2" max="10">
        <div class="hint">Local window for gradient estimation.</div>
      </div>
      <div class="config-row">
        <label>Strain Type</label>
        <select id="cfgStrainType">
          <option value="engineering" selected>Engineering Strain</option>
          <option value="green_lagrange">Green-Lagrange</option>
        </select>
      </div>
      <div class="config-row">
        <label>Min Points for Fit</label>
        <input type="number" id="cfgStrainMinPts" value="6" min="3" max="20">
        <div class="hint">Minimum valid neighbours needed for least-squares plane fit.</div>
      </div>
    </div>

    <div class="config-section">
      <h3>Displacement Filtering</h3>
      <div class="config-row">
        <label>Filter Type</label>
        <select id="cfgDispFilter">
          <option value="none" selected>None</option>
          <option value="gaussian">Gaussian</option>
          <option value="median">Median</option>
        </select>
        <div class="hint">Smooth displacement before strain computation.</div>
      </div>
      <div class="config-row">
        <label>Kernel Size</label>
        <input type="number" id="cfgDispFilterSize" value="3" min="3" max="11" step="2">
        <div class="hint">Filter kernel size (odd number).</div>
      </div>
    </div>

    <div class="config-section">
      <h3>Processing</h3>
      <div class="config-row">
        <label>Frame Skip</label>
        <input type="number" id="cfgFrameSkip" value="14" min="1" max="100">
        <div class="hint">Process every Nth frame. Lower = more frames, slower.</div>
      </div>
      <div class="config-row">
        <label>Start Frame</label>
        <input type="number" id="cfgStartFrame" value="0" min="0">
        <div class="hint">First frame index to process.</div>
      </div>
      <div class="config-row">
        <label>End Frame</label>
        <input type="number" id="cfgEndFrame" value="" placeholder="(all)">
        <div class="hint">Last frame index. Leave blank for all frames.</div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════ ROI SCREEN ═══════════════ -->
<div class="roi-screen" id="roiScreen">
  <div class="roi-header">
    <h2><span>ROI</span> Selection</h2>
    <span class="roi-instructions">
      <b>Click</b> to place polygon vertices &nbsp;|&nbsp;
      <b>Double-click</b> to close polygon &nbsp;|&nbsp;
      <b>Drag</b> vertices to adjust &nbsp;|&nbsp;
      <b>Scroll</b> to zoom &nbsp;|&nbsp;
      <b>Middle-drag</b> to pan
    </span>
    <div class="roi-toolbar">
      <button class="btn secondary" onclick="roiClear()">✕ Clear</button>
      <button class="btn secondary" onclick="roiUndo()">↩ Undo</button>
      <button class="btn roi-confirm-btn" id="roiConfirmBtn" onclick="roiConfirmAndRun()" disabled>✓ Confirm ROI &amp; Run</button>
    </div>
  </div>
  <div class="roi-viewport" id="roiViewport">
    <canvas id="roiImageCanvas"></canvas>
    <canvas id="roiDrawCanvas"></canvas>
    <div class="roi-status" id="roiStatus">Loading reference image...</div>
    <div class="roi-zoom-info" id="roiZoomInfo">100%</div>
  </div>
</div>

<!-- ═══════════════ ANALYSIS SCREEN ═══════════════ -->
<div class="analysis-screen" id="analysisScreen">
  <div class="analysis-header">
    <h1><span>DIC</span> Analysis Dashboard</h1>
    <span class="status" id="analysisStatus">Initializing...</span>
  </div>

  <div class="progress-bar-container" id="progressContainer">
    <div class="progress-outer"><div class="progress-inner" id="progressBar"></div></div>
    <div class="progress-info">
      <span id="progressText">Starting analysis...</span>
      <span><button class="stop-btn" id="stopBtn" onclick="stopAnalysis()">⏹ Stop</button></span>
    </div>
  </div>

  <div class="main-area">
    <!-- Left sidebar -->
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
          <option value="hot">Hot</option>
        </select>
        <canvas id="colorbarPreview" height="20" style="margin-top:4px;border-radius:3px;"></canvas>
        <div class="range-labels"><span id="rangeMinLabel">0</span><span id="rangeMaxLabel">1</span></div>
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

      <h3>Display</h3>
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

      <h3>Tools</h3>
      <div style="display:flex;flex-wrap:wrap;gap:4px;">
        <button class="btn secondary active" id="toolInspect" onclick="setTool('inspect')">🔍 Inspect</button>
        <button class="btn secondary" id="toolLine" onclick="setTool('line')">📏 Line</button>
        <button class="btn secondary" id="toolExtensometer" onclick="setTool('extensometer')">📐 Extenso.</button>
        <button class="btn secondary" id="toolPoint" onclick="setTool('point')">📍 Point</button>
      </div>

      <h3>Export</h3>
      <button class="btn" onclick="exportCSV()">📄 CSV</button>
      <button class="btn" onclick="exportImage()">🖼 Image</button>
    </div>

    <!-- Main viewport -->
    <div class="viewport" id="viewport">
      <canvas id="imageCanvas"></canvas>
      <canvas id="contourCanvas"></canvas>
      <canvas id="overlayCanvas"></canvas>
      <div class="coordinates" id="coordinates">X: — Y: — Val: —</div>
      <div class="inspector" id="inspector"></div>
      <div class="loading-overlay" id="viewportLoading"><div class="spinner"></div></div>
    </div>

    <!-- Right panel -->
    <div class="right-panel">
      <div class="panel-section">
        <h4>Statistics</h4>
        <div class="stats-grid">
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
  </div>

  <!-- Timeline -->
  <div class="timeline">
    <div class="timeline-controls">
      <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
      <button class="btn secondary" onclick="prevFrame()">◀</button>
      <input type="range" class="timeline-slider" id="frameSlider" min="0" max="0" value="0">
      <button class="btn secondary" onclick="nextFrame()">▶</button>
      <span id="frameLabel" style="font-size:12px;min-width:80px;">Frame 0</span>
      <span id="timeLabel" style="font-size:11px;color:#888;min-width:100px;"></span>
    </div>
    <div class="timeline-charts">
      <canvas id="loadDispChart"></canvas>
      <canvas id="strainTimeChart"></canvas>
    </div>
  </div>
</div>

<script>
// ============================================================
// APPLICATION STATE
// ============================================================
const APP = {
  selectedFolder: null,
  scanInfo: null,
  // Live data that arrives frame-by-frame
  DATA: {
    grid_x: null, grid_y: null,
    frames: [],
    frame_indices: [],
    num_frames: 0,
    image_width: 750, image_height: 1024,
    ref_image_b64: null,
    frame_images: [],  // per-frame base64 thumbnails for background animation
    load: null,
  },
  state: {
    currentFrame: 0,
    variable: 'exx',
    colormap: 'jet',
    rangeMin: 0, rangeMax: 1,
    autoRange: true,
    showImage: true, showContour: true, showGrid: false,
    opacity: 0.7,
    tool: 'inspect',
    playing: false,
    zoom: 1, panX: 0, panY: 0,
    lineStart: null, lineEnd: null,
    trackedPoints: [],
    extensometerPts: [],
  },
  eventSource: null,
};

// ============================================================
// COLORMAPS
// ============================================================
const COLORMAPS = {
  jet: t => {
    let r,g,b;
    if(t<0.125){r=0;g=0;b=0.5+t*4;} else if(t<0.375){r=0;g=(t-0.125)*4;b=1;}
    else if(t<0.625){r=(t-0.375)*4;g=1;b=1-(t-0.375)*4;} else if(t<0.875){r=1;g=1-(t-0.625)*4;b=0;}
    else{r=1-(t-0.875)*2;g=0;b=0;}
    return [Math.round(r*255),Math.round(g*255),Math.round(b*255)];
  },
  viridis: t => {
    const c=[[68,1,84],[72,35,116],[64,67,135],[52,94,141],[41,120,142],[32,144,140],[34,167,132],[68,190,112],[121,209,81],[189,222,38],[253,231,37]];
    const idx=t*(c.length-1),i=Math.min(Math.floor(idx),c.length-2),f=idx-i;
    return [Math.round(c[i][0]*(1-f)+c[i+1][0]*f),Math.round(c[i][1]*(1-f)+c[i+1][1]*f),Math.round(c[i][2]*(1-f)+c[i+1][2]*f)];
  },
  plasma: t => {
    const c=[[13,8,135],[75,3,161],[126,3,168],[168,34,150],[203,70,121],[231,109,90],[250,155,60],[253,203,44],[240,249,33]];
    const idx=t*(c.length-1),i=Math.min(Math.floor(idx),c.length-2),f=idx-i;
    return [Math.round(c[i][0]*(1-f)+c[i+1][0]*f),Math.round(c[i][1]*(1-f)+c[i+1][1]*f),Math.round(c[i][2]*(1-f)+c[i+1][2]*f)];
  },
  inferno: t => {
    const c=[[0,0,4],[22,11,57],[66,10,104],[106,23,110],[147,38,103],[186,54,85],[221,81,58],[246,121,24],[252,172,12],[245,228,56],[252,255,164]];
    const idx=t*(c.length-1),i=Math.min(Math.floor(idx),c.length-2),f=idx-i;
    return [Math.round(c[i][0]*(1-f)+c[i+1][0]*f),Math.round(c[i][1]*(1-f)+c[i+1][1]*f),Math.round(c[i][2]*(1-f)+c[i+1][2]*f)];
  },
  coolwarm: t => {
    const r=t<0.5?Math.round(59+t*2*200):255;
    const g=t<0.5?Math.round(76+t*2*180):Math.round(255-(t-0.5)*2*180);
    const b=t<0.5?255:Math.round(255-(t-0.5)*2*200);
    return [r,g,b];
  },
  rdbu: t => {
    if(t<0.5) return [Math.round(33+t*2*200),Math.round(102+t*2*153),Math.round(172+t*2*83)];
    return [Math.round(255-(t-0.5)*2*55),Math.round(255-(t-0.5)*2*195),Math.round(255-(t-0.5)*2*237)];
  },
  hot: t => {
    return [Math.min(255,Math.round(t*3*255)),Math.min(255,Math.max(0,Math.round((t-0.333)*3*255))),Math.min(255,Math.max(0,Math.round((t-0.666)*3*255)))];
  }
};

function getColor(value, vmin, vmax) {
  if(value===null||value===undefined||isNaN(value)) return [40,40,40,0];
  let t=(value-vmin)/(vmax-vmin); t=Math.max(0,Math.min(1,t));
  return (COLORMAPS[APP.state.colormap]||COLORMAPS.jet)(t);
}

// ============================================================
// SETUP SCREEN
// ============================================================
async function browseFolder() {
  const btn = document.getElementById('browseBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Waiting for folder picker...';

  try {
    const res = await fetch('/api/browse', {method:'POST'});
    const data = await res.json();

    if(data.folder) {
      APP.selectedFolder = data.folder;
      const fp = document.getElementById('folderPath');
      fp.textContent = data.folder;
      fp.style.display = 'block';

      // Scan the folder
      await scanFolder(data.folder);
    }
  } catch(e) {
    console.error('Browse error:', e);
  }

  btn.disabled = false;
  btn.textContent = '📂 Select Image Folder';
}

async function scanFolder(folder) {
  try {
    const res = await fetch('/api/scan', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({folder})
    });
    const info = await res.json();
    APP.scanInfo = info;

    document.getElementById('cam0Count').textContent = info.camera0_count || 0;
    document.getElementById('cam1Count').textContent = info.camera1_count || 0;
    document.getElementById('imgDims').textContent = info.image_width && info.image_height ?
      `${info.image_width} × ${info.image_height}` : '—';
    document.getElementById('csvStatus').textContent = info.has_csv ? '✅ Found' : '❌ Not found';

    if(info.load_range) {
      document.getElementById('loadRangeRow').style.display = 'flex';
      document.getElementById('loadRange').textContent =
        `${info.load_range[0].toFixed(0)} to ${info.load_range[1].toFixed(0)} N`;
    }
    if(info.disp_range) {
      document.getElementById('dispRangeRow').style.display = 'flex';
      document.getElementById('dispRange').textContent =
        `${info.disp_range[0].toFixed(3)} to ${info.disp_range[1].toFixed(3)} mm`;
    }
    if(info.preview_b64) {
      const img = document.getElementById('previewImg');
      img.src = 'data:image/png;base64,' + info.preview_b64;
      img.style.display = 'block';
    }

    // Auto-set frame skip based on image count
    const total = info.camera0_count || 1;
    const skip = Math.max(1, Math.round(total / 40));
    document.getElementById('cfgFrameSkip').value = skip;

    document.getElementById('scanInfo').style.display = 'block';
    document.getElementById('runBtn').style.display = 'inline-block';
  } catch(e) {
    console.error('Scan error:', e);
  }
}

// ============================================================
// START ANALYSIS
// ============================================================
async function startAnalysis() {
  // "Run Analysis" on setup screen -> go to ROI selection screen
  if(!APP.selectedFolder) return;

  // Save config for later
  const endFrameVal = document.getElementById('cfgEndFrame').value.trim();
  APP.savedConfig = {
    preset: document.getElementById('cfgPreset').value,
    subset_size: parseInt(document.getElementById('cfgSubsetSize').value),
    step_size: parseInt(document.getElementById('cfgStepSize').value),
    downsample: parseInt(document.getElementById('cfgDownsample').value),
    subpixel_search: parseInt(document.getElementById('cfgSubpixelSearch').value),
    correlation_threshold: parseFloat(document.getElementById('cfgCorrThreshold').value),
    of_levels: parseInt(document.getElementById('cfgOfLevels').value),
    of_winsize: parseInt(document.getElementById('cfgOfWinsize').value),
    of_iterations: parseInt(document.getElementById('cfgOfIterations').value),
    strain_window: parseInt(document.getElementById('cfgStrainWindow').value),
    strain_type: document.getElementById('cfgStrainType').value,
    strain_min_points: parseInt(document.getElementById('cfgStrainMinPts').value),
    displacement_filter: document.getElementById('cfgDispFilter').value,
    displacement_filter_size: parseInt(document.getElementById('cfgDispFilterSize').value),
    frame_skip: parseInt(document.getElementById('cfgFrameSkip').value),
    start_frame: parseInt(document.getElementById('cfgStartFrame').value) || 0,
    end_frame: endFrameVal ? parseInt(endFrameVal) : null,
  };

  // Switch to ROI screen
  document.getElementById('setupScreen').style.display = 'none';
  document.getElementById('roiScreen').style.display = 'flex';

  // Load reference image for ROI drawing
  await loadRoiImage();
}

// ============================================================
// ROI POLYGON DRAWING
// ============================================================
const ROI = {
  image: null,         // HTMLImageElement
  origW: 0, origH: 0, // original image size
  dispW: 0, dispH: 0, // displayed (possibly downscaled) image size
  dispScale: 1,        // ratio display/original
  zoom: 1, panX: 0, panY: 0,
  vertices: [],        // [{x, y}] in ORIGINAL image pixel coords
  closed: false,
  dragging: -1,        // index of vertex being dragged, or -1
  hoverVertex: -1,
};

async function loadRoiImage() {
  document.getElementById('roiStatus').textContent = 'Loading reference image...';

  const res = await fetch(`/api/ref_image?folder=${encodeURIComponent(APP.selectedFolder)}`);
  const data = await res.json();
  if(data.error) { document.getElementById('roiStatus').textContent = 'Error: '+data.error; return; }

  ROI.origW = data.orig_width;
  ROI.origH = data.orig_height;
  ROI.dispW = data.display_width;
  ROI.dispH = data.display_height;
  ROI.dispScale = data.display_scale;
  ROI.vertices = [];
  ROI.closed = false;

  ROI.image = new Image();
  ROI.image.onload = () => {
    setupRoiCanvases();
    roiFitToView();
    renderRoi();
    document.getElementById('roiStatus').textContent =
      `Image: ${ROI.origW}×${ROI.origH}px — Click to place polygon vertices`;
  };
  ROI.image.src = 'data:image/png;base64,' + data.image_b64;
}

function setupRoiCanvases() {
  const vp = document.getElementById('roiViewport');
  const rect = vp.getBoundingClientRect();
  ['roiImageCanvas','roiDrawCanvas'].forEach(id => {
    const c = document.getElementById(id);
    c.width = rect.width; c.height = rect.height;
    c.style.width = rect.width+'px'; c.style.height = rect.height+'px';
  });
}

function roiFitToView() {
  const vp = document.getElementById('roiViewport').getBoundingClientRect();
  const scaleX = vp.width / ROI.dispW;
  const scaleY = vp.height / ROI.dispH;
  ROI.zoom = Math.min(scaleX, scaleY) * 0.95;
  ROI.panX = (vp.width - ROI.dispW * ROI.zoom) / 2;
  ROI.panY = (vp.height - ROI.dispH * ROI.zoom) / 2;
}

// Convert screen coords to original image coords
function roiScreenToOrig(sx, sy) {
  const ix = (sx - ROI.panX) / ROI.zoom;             // display image coords
  const iy = (sy - ROI.panY) / ROI.zoom;
  return { x: ix / ROI.dispScale, y: iy / ROI.dispScale };  // original image coords
}

// Convert original image coords to screen coords
function roiOrigToScreen(ox, oy) {
  const ix = ox * ROI.dispScale;
  const iy = oy * ROI.dispScale;
  return { sx: ix * ROI.zoom + ROI.panX, sy: iy * ROI.zoom + ROI.panY };
}

function renderRoi() {
  // Draw image
  const imgCanvas = document.getElementById('roiImageCanvas');
  const imgCtx = imgCanvas.getContext('2d');
  imgCtx.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
  if(ROI.image) {
    imgCtx.drawImage(ROI.image, ROI.panX, ROI.panY, ROI.dispW * ROI.zoom, ROI.dispH * ROI.zoom);
  }

  // Draw polygon overlay
  const drawCanvas = document.getElementById('roiDrawCanvas');
  const ctx = drawCanvas.getContext('2d');
  ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);

  if(ROI.vertices.length === 0) return;

  // Convert vertices to screen coords
  const screenPts = ROI.vertices.map(v => roiOrigToScreen(v.x, v.y));

  // Draw filled polygon (semi-transparent)
  if(ROI.closed && screenPts.length >= 3) {
    ctx.fillStyle = 'rgba(233, 69, 96, 0.15)';
    ctx.beginPath();
    ctx.moveTo(screenPts[0].sx, screenPts[0].sy);
    for(let i=1; i<screenPts.length; i++) ctx.lineTo(screenPts[i].sx, screenPts[i].sy);
    ctx.closePath();
    ctx.fill();
  }

  // Draw edges
  ctx.strokeStyle = ROI.closed ? '#e94560' : '#ff8888';
  ctx.lineWidth = 2;
  ctx.setLineDash(ROI.closed ? [] : [6, 4]);
  ctx.beginPath();
  ctx.moveTo(screenPts[0].sx, screenPts[0].sy);
  for(let i=1; i<screenPts.length; i++) ctx.lineTo(screenPts[i].sx, screenPts[i].sy);
  if(ROI.closed) ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw vertices
  screenPts.forEach((p, i) => {
    const isHover = (i === ROI.hoverVertex);
    const radius = isHover ? 7 : 5;
    ctx.fillStyle = isHover ? '#fff' : '#e94560';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(p.sx, p.sy, radius, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();

    // Vertex label
    ctx.fillStyle = '#fff';
    ctx.font = '10px monospace';
    ctx.fillText((i+1), p.sx + 10, p.sy - 8);
  });

  // Update zoom info
  document.getElementById('roiZoomInfo').textContent = Math.round(ROI.zoom * 100 / (ROI.dispScale || 1) * ROI.dispScale) + '%';

  // Update confirm button
  document.getElementById('roiConfirmBtn').disabled = !(ROI.closed && ROI.vertices.length >= 3);
}

// Find vertex near screen position (within threshold)
function roiFindVertex(sx, sy, threshold=12) {
  for(let i=0; i<ROI.vertices.length; i++) {
    const p = roiOrigToScreen(ROI.vertices[i].x, ROI.vertices[i].y);
    if(Math.hypot(p.sx-sx, p.sy-sy) < threshold) return i;
  }
  return -1;
}

// Mouse events on ROI draw canvas
const roiCanvas = () => document.getElementById('roiDrawCanvas');

document.addEventListener('DOMContentLoaded', () => {
  const canvas = roiCanvas();
  if(!canvas) return;

  let isPanning = false, panStartX = 0, panStartY = 0;
  let mouseDownPos = null;

  canvas.addEventListener('mousedown', e => {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

    if(e.button === 1) {
      // Middle click — pan
      isPanning = true;
      panStartX = e.clientX - ROI.panX;
      panStartY = e.clientY - ROI.panY;
      e.preventDefault();
      return;
    }

    if(e.button === 0) {
      // Left click — check if dragging existing vertex
      const vi = roiFindVertex(sx, sy);
      if(vi >= 0 && ROI.closed) {
        ROI.dragging = vi;
        return;
      }
      mouseDownPos = {sx, sy};
    }
  });

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;

    if(isPanning) {
      ROI.panX = e.clientX - panStartX;
      ROI.panY = e.clientY - panStartY;
      renderRoi();
      return;
    }

    if(ROI.dragging >= 0) {
      const orig = roiScreenToOrig(sx, sy);
      ROI.vertices[ROI.dragging] = {x: orig.x, y: orig.y};
      renderRoi();
      return;
    }

    // Hover highlight
    ROI.hoverVertex = roiFindVertex(sx, sy);

    // Update status with coordinates
    const orig = roiScreenToOrig(sx, sy);
    document.getElementById('roiStatus').textContent =
      `X: ${Math.round(orig.x)}  Y: ${Math.round(orig.y)}  |  ${ROI.vertices.length} vertices` +
      (ROI.closed ? ' (closed)' : ' — click to add, double-click to close');

    renderRoi();
  });

  canvas.addEventListener('mouseup', e => {
    if(isPanning) { isPanning = false; return; }
    if(ROI.dragging >= 0) { ROI.dragging = -1; return; }
  });

  canvas.addEventListener('click', e => {
    if(ROI.closed) return; // already closed, use Clear to restart

    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const orig = roiScreenToOrig(sx, sy);

    // Check if clicking near first vertex to close
    if(ROI.vertices.length >= 3) {
      const first = roiOrigToScreen(ROI.vertices[0].x, ROI.vertices[0].y);
      if(Math.hypot(first.sx-sx, first.sy-sy) < 15) {
        ROI.closed = true;
        renderRoi();
        return;
      }
    }

    ROI.vertices.push({x: orig.x, y: orig.y});
    renderRoi();
  });

  canvas.addEventListener('dblclick', e => {
    e.preventDefault();
    if(ROI.vertices.length >= 3 && !ROI.closed) {
      ROI.closed = true;
      renderRoi();
    }
  });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;

    const oldZoom = ROI.zoom;
    ROI.zoom *= e.deltaY > 0 ? 0.9 : 1.1;
    ROI.zoom = Math.max(0.05, Math.min(20, ROI.zoom));

    // Zoom towards mouse position
    ROI.panX = mx - (mx - ROI.panX) * (ROI.zoom / oldZoom);
    ROI.panY = my - (my - ROI.panY) * (ROI.zoom / oldZoom);

    renderRoi();
  });

  canvas.addEventListener('contextmenu', e => {
    e.preventDefault();
    // Right-click to delete nearest vertex
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const vi = roiFindVertex(sx, sy, 15);
    if(vi >= 0) {
      ROI.vertices.splice(vi, 1);
      if(ROI.vertices.length < 3) ROI.closed = false;
      renderRoi();
    }
  });

  // Global mouseup for pan
  document.addEventListener('mouseup', () => { isPanning = false; ROI.dragging = -1; });
  document.addEventListener('mousemove', e => {
    if(isPanning) {
      ROI.panX = e.clientX - panStartX;
      ROI.panY = e.clientY - panStartY;
      renderRoi();
    }
  });
});

function roiClear() {
  ROI.vertices = [];
  ROI.closed = false;
  ROI.dragging = -1;
  renderRoi();
}

function roiUndo() {
  if(ROI.closed) {
    ROI.closed = false;
  } else if(ROI.vertices.length > 0) {
    ROI.vertices.pop();
  }
  renderRoi();
}

async function roiConfirmAndRun() {
  if(!ROI.closed || ROI.vertices.length < 3) return;

  // Add polygon to config (vertices in original image coordinates)
  const config = { ...APP.savedConfig };
  config.roi_polygon = ROI.vertices.map(v => [v.x, v.y]);

  // Switch to analysis screen
  document.getElementById('roiScreen').style.display = 'none';
  document.getElementById('analysisScreen').style.display = 'flex';
  setupCanvases();

  // Start the analysis with ROI
  try {
    await fetch('/api/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({folder: APP.selectedFolder, config})
    });
  } catch(e) {
    console.error('Start error:', e);
    return;
  }

  connectSSE();
}

function connectSSE() {
  if(APP.eventSource) APP.eventSource.close();

  APP.eventSource = new EventSource('/api/progress');

  APP.eventSource.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleProgressEvent(msg);
  };

  APP.eventSource.onerror = () => {
    console.log('SSE connection closed');
    APP.eventSource.close();
  };
}

function handleProgressEvent(msg) {
  switch(msg.type) {
    case 'grid_info':
      APP.DATA.grid_x = msg.data.grid_x;
      APP.DATA.grid_y = msg.data.grid_y;
      APP.DATA.image_width = msg.data.image_width;
      APP.DATA.image_height = msg.data.image_height;
      break;

    case 'frame_result':
      APP.DATA.frames.push(msg.data);
      APP.DATA.frame_indices.push(msg.frame_idx);
      APP.DATA.num_frames = APP.DATA.frames.length;

      // Store per-frame image thumbnail for background animation
      if(msg.image_b64) {
        APP.DATA.frame_images.push(msg.image_b64);
        // Also use first frame as fallback ref
        if(!APP.DATA.ref_image_b64) APP.DATA.ref_image_b64 = msg.image_b64;
      } else {
        APP.DATA.frame_images.push(APP.DATA.ref_image_b64 || null);
      }

      // Update progress bar
      document.getElementById('progressBar').style.width = msg.pct + '%';
      document.getElementById('progressText').textContent =
        `Frame ${msg.frame_idx} — ${msg.step}/${msg.total} (${msg.pct}%) — ${msg.elapsed}s/frame`;

      // Update timeline slider
      document.getElementById('frameSlider').max = APP.DATA.num_frames - 1;

      // Auto-show latest frame
      APP.state.currentFrame = APP.DATA.num_frames - 1;
      if(APP.state.autoRange) autoRange();
      render();
      updateLoadDispChart();
      updateStrainTimeChart();

      // Hide loading overlay after first frame
      document.getElementById('viewportLoading').style.display = 'none';
      break;

    case 'load_data':
      APP.DATA.load = msg.data;
      updateLoadDispChart();
      break;

    case 'complete':
      document.getElementById('progressBar').style.width = '100%';
      document.getElementById('progressText').textContent = `Complete — ${msg.num_frames} frames processed`;
      document.getElementById('stopBtn').style.display = 'none';
      document.getElementById('analysisStatus').textContent = `Analysis complete — ${msg.num_frames} frames`;
      if(APP.eventSource) APP.eventSource.close();
      break;

    case 'error':
      document.getElementById('progressText').textContent = `Error: ${msg.message}`;
      document.getElementById('progressBar').style.background = '#e94560';
      if(APP.eventSource) APP.eventSource.close();
      break;
  }
}

async function stopAnalysis() {
  await fetch('/api/stop', {method:'POST'});
  document.getElementById('progressText').textContent = 'Stopping...';
}

// ============================================================
// DATA HELPERS
// ============================================================
function getCurrentData() {
  const frame = APP.DATA.frames[APP.state.currentFrame];
  if(!frame) return null;

  if(APP.state.variable === 'displacement_mag') {
    const u=frame.u, v=frame.v;
    if(!u||!v) return null;
    const ny=u.length, nx=u[0].length;
    const mag=[];
    for(let i=0;i<ny;i++){mag[i]=[];for(let j=0;j<nx;j++){
      if(u[i][j]!==null&&v[i][j]!==null) mag[i][j]=Math.sqrt(u[i][j]**2+v[i][j]**2);
      else mag[i][j]=null;
    }}
    return mag;
  }
  return frame[APP.state.variable] || null;
}

function computeStats(data) {
  if(!data) return {};
  const vals=[];
  for(let i=0;i<data.length;i++) for(let j=0;j<data[i].length;j++)
    if(data[i][j]!==null&&!isNaN(data[i][j])) vals.push(data[i][j]);
  if(vals.length===0) return {min:0,max:0,mean:0,std:0,count:0};
  vals.sort((a,b)=>a-b);
  const sum=vals.reduce((a,b)=>a+b,0), mean=sum/vals.length;
  const variance=vals.reduce((a,b)=>a+(b-mean)**2,0)/vals.length;
  return {min:vals[0],max:vals[vals.length-1],mean,std:Math.sqrt(variance),count:vals.length,
    p2:vals[Math.floor(vals.length*0.02)],p98:vals[Math.floor(vals.length*0.98)],values:vals};
}

// ============================================================
// RENDERING
// ============================================================
function setupCanvases() {
  const vp = document.getElementById('viewport');
  const rect = vp.getBoundingClientRect();
  ['imageCanvas','contourCanvas','overlayCanvas'].forEach(id => {
    const c=document.getElementById(id);
    c.width=rect.width; c.height=rect.height;
    c.style.width=rect.width+'px'; c.style.height=rect.height+'px';
  });
}

function renderImage() {
  const canvas=document.getElementById('imageCanvas'), ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!APP.state.showImage) return;

  // Use the current frame's image if available, otherwise fallback to ref
  const frameIdx = APP.state.currentFrame;
  const b64 = (APP.DATA.frame_images[frameIdx]) || APP.DATA.ref_image_b64;
  if(!b64) return;

  const img=new Image();
  img.onload=()=>{
    const scale=Math.min(canvas.width/APP.DATA.image_width,canvas.height/APP.DATA.image_height)*APP.state.zoom;
    const iw=APP.DATA.image_width*scale, ih=APP.DATA.image_height*scale;
    const ox=(canvas.width-iw)/2+APP.state.panX, oy=(canvas.height-ih)/2+APP.state.panY;
    ctx.drawImage(img,ox,oy,iw,ih);
  };
  // Use jpeg format since server sends jpg-encoded thumbnails
  const prefix = b64.startsWith('/9j/') ? 'data:image/jpeg;base64,' : 'data:image/png;base64,';
  img.src = prefix + b64;
}

function renderContour() {
  const canvas=document.getElementById('contourCanvas'), ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!APP.state.showContour) return;

  const data=getCurrentData();
  if(!data||!APP.DATA.grid_x) return;

  const gx=APP.DATA.grid_x, gy=APP.DATA.grid_y;
  const ny=data.length, nx=data[0].length;

  const scale=Math.min(canvas.width/APP.DATA.image_width,canvas.height/APP.DATA.image_height)*APP.state.zoom;
  const ox=(canvas.width-APP.DATA.image_width*scale)/2+APP.state.panX;
  const oy=(canvas.height-APP.DATA.image_height*scale)/2+APP.state.panY;

  const off=document.createElement('canvas'); off.width=nx; off.height=ny;
  const offCtx=off.getContext('2d');
  const imgData=offCtx.createImageData(nx,ny);

  for(let iy=0;iy<ny;iy++) for(let ix=0;ix<nx;ix++){
    const idx=(iy*nx+ix)*4, val=data[iy][ix];
    if(val===null||isNaN(val)){imgData.data[idx+3]=0;}
    else{const[r,g,b]=getColor(val,APP.state.rangeMin,APP.state.rangeMax);
      imgData.data[idx]=r;imgData.data[idx+1]=g;imgData.data[idx+2]=b;imgData.data[idx+3]=Math.round(APP.state.opacity*255);}
  }
  offCtx.putImageData(imgData,0,0);

  const x0=ox+gx[0][0]*scale, y0=oy+gy[0][0]*scale;
  const x1=ox+gx[0][nx-1]*scale, y1=oy+gy[ny-1][0]*scale;
  ctx.imageSmoothingEnabled=true;
  ctx.drawImage(off,x0,y0,x1-x0,y1-y0);

  if(APP.state.showGrid){ctx.strokeStyle='rgba(255,255,255,0.15)';ctx.lineWidth=0.5;
    for(let iy=0;iy<ny;iy++) for(let ix=0;ix<nx;ix++){
      ctx.strokeRect(ox+gx[iy][ix]*scale-1,oy+gy[iy][ix]*scale-1,2,2);}}
}

function renderOverlay() {
  const canvas=document.getElementById('overlayCanvas'), ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);

  const scale=Math.min(canvas.width/APP.DATA.image_width,canvas.height/APP.DATA.image_height)*APP.state.zoom;
  const ox=(canvas.width-APP.DATA.image_width*scale)/2+APP.state.panX;
  const oy=(canvas.height-APP.DATA.image_height*scale)/2+APP.state.panY;

  // Line extraction
  if(APP.state.lineStart&&APP.state.lineEnd){
    ctx.strokeStyle='#e94560';ctx.lineWidth=2;ctx.setLineDash([5,3]);
    ctx.beginPath();ctx.moveTo(APP.state.lineStart.sx,APP.state.lineStart.sy);
    ctx.lineTo(APP.state.lineEnd.sx,APP.state.lineEnd.sy);ctx.stroke();ctx.setLineDash([]);
    [APP.state.lineStart,APP.state.lineEnd].forEach(p=>{
      ctx.fillStyle='#e94560';ctx.beginPath();ctx.arc(p.sx,p.sy,5,0,Math.PI*2);ctx.fill();});
  }

  // Tracked points
  APP.state.trackedPoints.forEach((p,i)=>{
    const sx=ox+p.gx*scale,sy=oy+p.gy*scale;
    ctx.strokeStyle=['#00ff88','#ff8800','#00aaff','#ff00aa'][i%4];ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(sx-8,sy);ctx.lineTo(sx+8,sy);ctx.moveTo(sx,sy-8);ctx.lineTo(sx,sy+8);ctx.stroke();
    ctx.fillStyle=ctx.strokeStyle;ctx.font='10px monospace';ctx.fillText('P'+(i+1),sx+10,sy-5);
  });

  // Extensometer
  if(APP.state.extensometerPts.length===2){
    const p1=APP.state.extensometerPts[0],p2=APP.state.extensometerPts[1];
    const sx1=ox+p1.gx*scale,sy1=oy+p1.gy*scale;
    const sx2=ox+p2.gx*scale,sy2=oy+p2.gy*scale;
    ctx.strokeStyle='#00ff88';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx1,sy1);ctx.lineTo(sx2,sy2);ctx.stroke();

    const frame=APP.DATA.frames[APP.state.currentFrame];
    if(frame){
      const dist0=Math.sqrt((p2.gx-p1.gx)**2+(p2.gy-p1.gy)**2);
      let u1=0,v1=0,u2=0,v2=0;
      if(frame.u&&p1.iy<frame.u.length&&p1.ix<frame.u[0].length){
        u1=frame.u[p1.iy][p1.ix]||0;v1=frame.v[p1.iy][p1.ix]||0;
        u2=frame.u[p2.iy][p2.ix]||0;v2=frame.v[p2.iy][p2.ix]||0;
      }
      const dx=(p2.gx+u2)-(p1.gx+u1),dy=(p2.gy+v2)-(p1.gy+v1);
      const dist=Math.sqrt(dx*dx+dy*dy),strain=(dist-dist0)/dist0;
      ctx.fillStyle='#00ff88';ctx.font='12px monospace';
      ctx.fillText('L='+dist.toFixed(1)+'px  e='+(strain*100).toFixed(3)+'%',(sx1+sx2)/2+10,(sy1+sy2)/2-10);
    }
  }
}

function render() {
  renderImage(); renderContour(); renderOverlay();
  updateStats(); updateHistogram(); updateColorbarPreview(); updateFrameInfo();
}

function updateFrameInfo() {
  if(APP.DATA.num_frames===0) return;
  const fidx=APP.DATA.frame_indices[APP.state.currentFrame]||0;
  document.getElementById('frameLabel').textContent=`Frame ${fidx}`;
  document.getElementById('frameSlider').value=APP.state.currentFrame;

  if(APP.DATA.load){
    const li=APP.DATA.load.frame_indices.indexOf(fidx);
    if(li>=0) document.getElementById('timeLabel').textContent=
      `${APP.DATA.load.time[li].toFixed(1)}s | ${APP.DATA.load.load_N[li].toFixed(0)}N`;
  }
}

function updateStats() {
  const data=getCurrentData(),stats=computeStats(data);
  const fmt=v=>{if(v===undefined)return'—';if(Math.abs(v)<0.01&&v!==0)return v.toExponential(3);return v.toFixed(4);};
  document.getElementById('statMin').textContent=fmt(stats.min);
  document.getElementById('statMax').textContent=fmt(stats.max);
  document.getElementById('statMean').textContent=fmt(stats.mean);
  document.getElementById('statStd').textContent=fmt(stats.std);
  document.getElementById('statPoints').textContent=stats.count||'—';

  const frame=APP.DATA.frames[APP.state.currentFrame];
  if(frame&&frame.correlation){
    const cs=computeStats(frame.correlation);
    document.getElementById('statCorr').textContent=cs.mean?cs.mean.toFixed(4):'—';
  }
}

function updateHistogram() {
  const canvas=document.getElementById('histogramCanvas'),ctx=canvas.getContext('2d');
  canvas.width=canvas.parentElement.clientWidth-20;
  ctx.clearRect(0,0,canvas.width,canvas.height);

  const data=getCurrentData(),stats=computeStats(data);
  if(!stats.values||stats.values.length===0) return;

  const nBins=40,bins=new Array(nBins).fill(0);
  const range=stats.max-stats.min||1;
  stats.values.forEach(v=>{bins[Math.min(nBins-1,Math.floor((v-stats.min)/range*nBins))]++;});

  const maxBin=Math.max(...bins),bw=canvas.width/nBins,pad=5;
  bins.forEach((count,i)=>{
    const h=(count/maxBin)*(canvas.height-2*pad),t=i/nBins;
    const[r,g,b]=getColor(stats.min+t*range,APP.state.rangeMin,APP.state.rangeMax);
    ctx.fillStyle=`rgb(${r},${g},${b})`;
    ctx.fillRect(i*bw,canvas.height-pad-h,bw-1,h);
  });
  ctx.fillStyle='#888';ctx.font='9px monospace';
  ctx.fillText(stats.min.toFixed(4),2,canvas.height-1);
  ctx.textAlign='right';ctx.fillText(stats.max.toFixed(4),canvas.width-2,canvas.height-1);ctx.textAlign='left';
}

function updateColorbarPreview() {
  const canvas=document.getElementById('colorbarPreview');
  if(!canvas.parentElement) return;
  canvas.width=canvas.parentElement.clientWidth-4;canvas.height=20;
  const ctx=canvas.getContext('2d'),w=canvas.width;
  for(let x=0;x<w;x++){const t=x/w;
    const[r,g,b]=getColor(APP.state.rangeMin+t*(APP.state.rangeMax-APP.state.rangeMin),APP.state.rangeMin,APP.state.rangeMax);
    ctx.fillStyle=`rgb(${r},${g},${b})`;ctx.fillRect(x,0,1,20);}
  document.getElementById('rangeMinLabel').textContent=APP.state.rangeMin.toFixed(4);
  document.getElementById('rangeMaxLabel').textContent=APP.state.rangeMax.toFixed(4);
}

function updateLoadDispChart() {
  const canvas=document.getElementById('loadDispChart'),ctx=canvas.getContext('2d');
  canvas.width=canvas.parentElement.clientWidth/2;canvas.height=canvas.parentElement.clientHeight;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!APP.DATA.load){ctx.fillStyle='#666';ctx.font='12px sans-serif';ctx.fillText('No load data',20,canvas.height/2);return;}

  const pad={top:15,right:10,bottom:25,left:50},w=canvas.width-pad.left-pad.right,h=canvas.height-pad.top-pad.bottom;
  const disp=APP.DATA.load.displacement_mm,load=APP.DATA.load.load_N;
  const dMin=Math.min(...disp),dMax=Math.max(...disp),lMin=Math.min(...load),lMax=Math.max(...load);
  const dRange=dMax-dMin||1,lRange=lMax-lMin||1;

  ctx.strokeStyle='#333';ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){const y=pad.top+h*i/4;ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(pad.left+w,y);ctx.stroke();}

  ctx.strokeStyle='#4a9eff';ctx.lineWidth=1.5;ctx.beginPath();
  for(let i=0;i<disp.length;i++){const x=pad.left+(disp[i]-dMin)/dRange*w,y=pad.top+h-(load[i]-lMin)/lRange*h;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();

  const fidx=APP.DATA.frame_indices[APP.state.currentFrame];
  const li=APP.DATA.load.frame_indices.indexOf(fidx);
  if(li>=0){const x=pad.left+(disp[li]-dMin)/dRange*w,y=pad.top+h-(load[li]-lMin)/lRange*h;
    ctx.fillStyle='#e94560';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();}

  ctx.fillStyle='#4a9eff';ctx.font='10px sans-serif';ctx.fillText('Load vs Displacement',pad.left,12);
}

function updateStrainTimeChart() {
  const canvas=document.getElementById('strainTimeChart'),ctx=canvas.getContext('2d');
  canvas.width=canvas.parentElement.clientWidth/2;canvas.height=canvas.parentElement.clientHeight;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(APP.DATA.frames.length<2) return;

  const pad={top:15,right:10,bottom:25,left:50},w=canvas.width-pad.left-pad.right,h=canvas.height-pad.top-pad.bottom;
  const means=[],maxes=[];
  for(let fi=0;fi<APP.DATA.frames.length;fi++){
    const frame=APP.DATA.frames[fi];let d=[];
    if(APP.state.variable==='displacement_mag'){
      if(frame.u)for(let i=0;i<frame.u.length;i++)for(let j=0;j<frame.u[i].length;j++)
        if(frame.u[i][j]!==null&&frame.v[i][j]!==null)d.push(Math.sqrt(frame.u[i][j]**2+frame.v[i][j]**2));
    } else {const fd=frame[APP.state.variable];
      if(fd)for(let i=0;i<fd.length;i++)for(let j=0;j<fd[i].length;j++)
        if(fd[i][j]!==null&&!isNaN(fd[i][j]))d.push(fd[i][j]);}
    if(d.length>0){means.push(d.reduce((a,b)=>a+b,0)/d.length);maxes.push(Math.max(...d));}
    else{means.push(0);maxes.push(0);}
  }

  const yMin=Math.min(...means,...maxes),yMax=Math.max(...means,...maxes),yRange=yMax-yMin||1;
  ctx.strokeStyle='#333';ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){const y=pad.top+h*i/4;ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(pad.left+w,y);ctx.stroke();}

  ctx.strokeStyle='rgba(233,69,96,0.5)';ctx.lineWidth=1;ctx.beginPath();
  for(let i=0;i<maxes.length;i++){const x=pad.left+i/(maxes.length-1)*w,y=pad.top+h-(maxes[i]-yMin)/yRange*h;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();

  ctx.strokeStyle='#00ff88';ctx.lineWidth=1.5;ctx.beginPath();
  for(let i=0;i<means.length;i++){const x=pad.left+i/(means.length-1)*w,y=pad.top+h-(means[i]-yMin)/yRange*h;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();

  const x=pad.left+APP.state.currentFrame/(means.length-1)*w;
  ctx.strokeStyle='#e94560';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,pad.top+h);ctx.stroke();

  ctx.fillStyle='#00ff88';ctx.font='10px sans-serif';ctx.fillText(APP.state.variable+' over time',pad.left,12);
}

function updateLineExtraction() {
  const canvas=document.getElementById('lineExtractCanvas'),ctx=canvas.getContext('2d');
  canvas.width=canvas.parentElement.clientWidth-20;ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!APP.state.lineStart||!APP.state.lineEnd){ctx.fillStyle='#666';ctx.font='11px sans-serif';ctx.fillText('Use Line tool',10,canvas.height/2);return;}
  const data=getCurrentData();if(!data||!APP.DATA.grid_x) return;
  const gx=APP.DATA.grid_x,gy=APP.DATA.grid_y,ny=data.length,nx=data[0].length;
  const nSamples=100,lineData=[];
  for(let s=0;s<nSamples;s++){
    const t=s/(nSamples-1);
    const px=APP.state.lineStart.gx+t*(APP.state.lineEnd.gx-APP.state.lineStart.gx);
    const py=APP.state.lineStart.gy+t*(APP.state.lineEnd.gy-APP.state.lineStart.gy);
    let bestDist=Infinity,bestVal=null;
    for(let iy=0;iy<ny;iy++)for(let ix=0;ix<nx;ix++){
      const d=(gx[iy][ix]-px)**2+(gy[iy][ix]-py)**2;
      if(d<bestDist){bestDist=d;bestVal=data[iy][ix];}}
    const dist=Math.sqrt((px-APP.state.lineStart.gx)**2+(py-APP.state.lineStart.gy)**2);
    if(bestVal!==null&&!isNaN(bestVal)) lineData.push({dist,val:bestVal});
  }
  if(lineData.length===0) return;
  const pad2={top:10,right:10,bottom:20,left:45},w=canvas.width-pad2.left-pad2.right,h=canvas.height-pad2.top-pad2.bottom;
  const dMax=Math.max(...lineData.map(d=>d.dist)),vMin=Math.min(...lineData.map(d=>d.val)),vMax=Math.max(...lineData.map(d=>d.val)),vRange=vMax-vMin||1;
  ctx.strokeStyle='#e94560';ctx.lineWidth=1.5;ctx.beginPath();
  lineData.forEach((d,i)=>{const x=pad2.left+d.dist/dMax*w,y=pad2.top+h-(d.val-vMin)/vRange*h;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
}

function updatePointTracking() {
  const canvas=document.getElementById('pointTrackCanvas'),ctx=canvas.getContext('2d');
  canvas.width=canvas.parentElement.clientWidth-20;ctx.clearRect(0,0,canvas.width,canvas.height);
  if(APP.state.trackedPoints.length===0){ctx.fillStyle='#666';ctx.font='11px sans-serif';ctx.fillText('Use Point tool',10,canvas.height/2);return;}
  const colors=['#00ff88','#ff8800','#00aaff','#ff00aa'];
  const pad2={top:10,right:10,bottom:20,left:45},w=canvas.width-pad2.left-pad2.right,h=canvas.height-pad2.top-pad2.bottom;
  let allVals=[];
  const traces=APP.state.trackedPoints.map(pt=>{
    const vals=[];
    for(let fi=0;fi<APP.DATA.frames.length;fi++){
      const fd=APP.DATA.frames[fi][APP.state.variable];
      if(fd&&pt.iy<fd.length&&pt.ix<fd[0].length&&fd[pt.iy][pt.ix]!==null){vals.push(fd[pt.iy][pt.ix]);allVals.push(fd[pt.iy][pt.ix]);}
      else vals.push(null);}
    return vals;});
  if(allVals.length===0) return;
  const vMin=Math.min(...allVals),vMax=Math.max(...allVals),vRange=vMax-vMin||1;
  traces.forEach((vals,pi)=>{ctx.strokeStyle=colors[pi%colors.length];ctx.lineWidth=1.5;ctx.beginPath();let started=false;
    vals.forEach((v,fi)=>{if(v===null){started=false;return;}const x=pad2.left+fi/(vals.length-1)*w,y=pad2.top+h-(v-vMin)/vRange*h;
      if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);});ctx.stroke();});
}

// ============================================================
// INTERACTION
// ============================================================
function screenToGrid(sx,sy) {
  const canvas=document.getElementById('contourCanvas');
  const scale=Math.min(canvas.width/APP.DATA.image_width,canvas.height/APP.DATA.image_height)*APP.state.zoom;
  const ox=(canvas.width-APP.DATA.image_width*scale)/2+APP.state.panX;
  const oy=(canvas.height-APP.DATA.image_height*scale)/2+APP.state.panY;
  return {gx:(sx-ox)/scale, gy:(sy-oy)/scale};
}

function findNearestGridPoint(gx,gy) {
  const gridX=APP.DATA.grid_x,gridY=APP.DATA.grid_y;
  if(!gridX) return null;
  const ny=gridX.length,nx=gridX[0].length;
  let bestDist=Infinity,bestIx=0,bestIy=0;
  for(let iy=0;iy<ny;iy++)for(let ix=0;ix<nx;ix++){
    const d=(gridX[iy][ix]-gx)**2+(gridY[iy][ix]-gy)**2;
    if(d<bestDist){bestDist=d;bestIx=ix;bestIy=iy;}}
  return {ix:bestIx,iy:bestIy,gx:gridX[bestIy][bestIx],gy:gridY[bestIy][bestIx]};
}

document.getElementById('overlayCanvas').addEventListener('mousemove', e => {
  const rect=e.target.getBoundingClientRect(),sx=e.clientX-rect.left,sy=e.clientY-rect.top;
  const {gx,gy}=screenToGrid(sx,sy);
  const nearest=findNearestGridPoint(gx,gy);
  if(!nearest) return;
  const data=getCurrentData();
  let val='—';
  if(data&&nearest.iy<data.length&&nearest.ix<data[0].length){const v=data[nearest.iy][nearest.ix];if(v!==null&&!isNaN(v))val=v.toFixed(6);}
  document.getElementById('coordinates').textContent=`X: ${nearest.gx.toFixed(1)}  Y: ${nearest.gy.toFixed(1)}  ${APP.state.variable}: ${val}`;

  if(APP.state.tool==='inspect'){
    const inspector=document.getElementById('inspector'),frame=APP.DATA.frames[APP.state.currentFrame];
    if(frame){
      const vars=['u','v','exx','eyy','exy','e1','e2','von_mises','correlation'];
      let html='';
      vars.forEach(vr=>{const fd=frame[vr];
        if(fd&&nearest.iy<fd.length&&nearest.ix<fd[0].length){const v=fd[nearest.iy][nearest.ix];
          html+=`<div class="row"><span class="label">${vr}</span><span class="value">${v!==null?v.toFixed(6):'N/A'}</span></div>`;}});
      inspector.innerHTML=html;inspector.style.display='block';
      inspector.style.left=(sx+15)+'px';inspector.style.top=(sy-10)+'px';
    }
  }
});

document.getElementById('overlayCanvas').addEventListener('mouseleave',()=>{document.getElementById('inspector').style.display='none';});

document.getElementById('overlayCanvas').addEventListener('click', e => {
  const rect=e.target.getBoundingClientRect(),sx=e.clientX-rect.left,sy=e.clientY-rect.top;
  const {gx,gy}=screenToGrid(sx,sy);
  const nearest=findNearestGridPoint(gx,gy);
  if(!nearest) return;

  if(APP.state.tool==='line'){
    if(!APP.state.lineStart) APP.state.lineStart={...nearest,sx,sy};
    else{APP.state.lineEnd={...nearest,sx,sy};renderOverlay();updateLineExtraction();}
  } else if(APP.state.tool==='point'){
    if(APP.state.trackedPoints.length<4){APP.state.trackedPoints.push(nearest);renderOverlay();updatePointTracking();}
  } else if(APP.state.tool==='extensometer'){
    if(APP.state.extensometerPts.length<2){APP.state.extensometerPts.push(nearest);renderOverlay();}
  }
});

document.getElementById('overlayCanvas').addEventListener('wheel', e => {
  e.preventDefault();
  APP.state.zoom*=e.deltaY>0?0.9:1.1;
  APP.state.zoom=Math.max(0.1,Math.min(10,APP.state.zoom));
  render();
});

let isPanning=false,panStartX,panStartY;
document.getElementById('overlayCanvas').addEventListener('mousedown',e=>{
  if(e.button===1){isPanning=true;panStartX=e.clientX-APP.state.panX;panStartY=e.clientY-APP.state.panY;e.preventDefault();}
});
document.addEventListener('mousemove',e=>{if(isPanning){APP.state.panX=e.clientX-panStartX;APP.state.panY=e.clientY-panStartY;render();}});
document.addEventListener('mouseup',()=>{isPanning=false;});

// Controls
document.getElementById('variableSelect').addEventListener('change',e=>{APP.state.variable=e.target.value;if(APP.state.autoRange)autoRange();render();updateStrainTimeChart();updatePointTracking();updateLineExtraction();});
document.getElementById('colormapSelect').addEventListener('change',e=>{APP.state.colormap=e.target.value;render();});
document.getElementById('rangeMinInput').addEventListener('change',e=>{APP.state.rangeMin=parseFloat(e.target.value);APP.state.autoRange=false;render();});
document.getElementById('rangeMaxInput').addEventListener('change',e=>{APP.state.rangeMax=parseFloat(e.target.value);APP.state.autoRange=false;render();});
document.getElementById('showImage').addEventListener('change',e=>{APP.state.showImage=e.target.checked;render();});
document.getElementById('showContour').addEventListener('change',e=>{APP.state.showContour=e.target.checked;render();});
document.getElementById('showGrid').addEventListener('change',e=>{APP.state.showGrid=e.target.checked;render();});
document.getElementById('contourOpacity').addEventListener('input',e=>{APP.state.opacity=e.target.value/100;render();});
document.getElementById('frameSlider').addEventListener('input',e=>{
  APP.state.currentFrame=parseInt(e.target.value);
  if(APP.state.autoRange)autoRange();else render();
  updateLoadDispChart();updateStrainTimeChart();updateLineExtraction();updatePointTracking();
});

function setTool(tool) {
  APP.state.tool=tool;
  document.querySelectorAll('.sidebar .btn.secondary').forEach(b=>b.classList.remove('active'));
  const btn=document.getElementById('tool'+tool.charAt(0).toUpperCase()+tool.slice(1));
  if(btn)btn.classList.add('active');
  if(tool==='line'){APP.state.lineStart=null;APP.state.lineEnd=null;}
  if(tool==='point') APP.state.trackedPoints=[];
  if(tool==='extensometer') APP.state.extensometerPts=[];
  renderOverlay();
}

function autoRange() {
  const data=getCurrentData(),stats=computeStats(data);
  if(stats.count>0){
    APP.state.rangeMin=stats.p2;APP.state.rangeMax=stats.p98;APP.state.autoRange=true;
    document.getElementById('rangeMinInput').value=APP.state.rangeMin.toFixed(6);
    document.getElementById('rangeMaxInput').value=APP.state.rangeMax.toFixed(6);
    render();
  }
}

function symmetricRange() {
  const absMax=Math.max(Math.abs(APP.state.rangeMin),Math.abs(APP.state.rangeMax));
  APP.state.rangeMin=-absMax;APP.state.rangeMax=absMax;
  document.getElementById('rangeMinInput').value=APP.state.rangeMin.toFixed(6);
  document.getElementById('rangeMaxInput').value=APP.state.rangeMax.toFixed(6);
  render();
}

function togglePlay(){APP.state.playing=!APP.state.playing;document.getElementById('playBtn').textContent=APP.state.playing?'⏸ Pause':'▶ Play';if(APP.state.playing)playLoop();}
function playLoop(){if(!APP.state.playing)return;nextFrame();requestAnimationFrame(()=>setTimeout(playLoop,150));}
function nextFrame(){if(APP.DATA.num_frames===0)return;APP.state.currentFrame=Math.min(APP.DATA.num_frames-1,APP.state.currentFrame+1);if(APP.state.autoRange)autoRange();else render();updateLoadDispChart();updateStrainTimeChart();updateLineExtraction();updatePointTracking();}
function prevFrame(){APP.state.currentFrame=Math.max(0,APP.state.currentFrame-1);if(APP.state.autoRange)autoRange();else render();updateLoadDispChart();updateStrainTimeChart();updateLineExtraction();updatePointTracking();}

function exportCSV() {
  const data=getCurrentData();if(!data||!APP.DATA.grid_x) return;
  const gx=APP.DATA.grid_x,gy=APP.DATA.grid_y;
  let csv='X,Y,'+APP.state.variable+'\\n';
  for(let iy=0;iy<data.length;iy++)for(let ix=0;ix<data[0].length;ix++)
    if(data[iy][ix]!==null) csv+=gx[iy][ix]+','+gy[iy][ix]+','+data[iy][ix]+'\\n';
  const blob=new Blob([csv],{type:'text/csv'}),a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='dic_'+APP.state.variable+'_frame'+APP.DATA.frame_indices[APP.state.currentFrame]+'.csv';a.click();
}

function exportImage() {
  const contourCanvas=document.getElementById('contourCanvas'),imageCanvas=document.getElementById('imageCanvas');
  const exp=document.createElement('canvas');exp.width=contourCanvas.width;exp.height=contourCanvas.height;
  const ctx=exp.getContext('2d');ctx.drawImage(imageCanvas,0,0);ctx.drawImage(contourCanvas,0,0);
  const a=document.createElement('a');a.href=exp.toDataURL('image/png');
  a.download='dic_'+APP.state.variable+'_frame'+APP.DATA.frame_indices[APP.state.currentFrame]+'.png';a.click();
}

window.addEventListener('resize',()=>{
  if(document.getElementById('roiScreen').style.display==='flex'){setupRoiCanvases();renderRoi();}
  if(document.getElementById('analysisScreen').style.display==='flex'){setupCanvases();render();updateLoadDispChart();updateStrainTimeChart();}
});
</script>
</body>
</html>'''


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='DIC Analysis Server')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    server = http.server.HTTPServer(('0.0.0.0', args.port), DICRequestHandler)
    print(f"\n{'='*50}")
    print(f"  DIC Analysis Server")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*50}\n")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{args.port}')).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
