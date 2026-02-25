# 2-D Digital Image Correlation (DIC) Analysis Tool

A Python-based 2D DIC tool for analyzing tensile test specimens imaged with a stereo camera system. Processes images from a single camera, computes displacement and strain fields using normalized cross-correlation, and produces publication-quality outputs.

## Features

- **Stereo camera support**: Filter images by camera index (`_0.tif` / `_1.tif`)
- **Natural sort**: Timestamps in filenames are sorted numerically, not lexicographically
- **Interactive calibration**: Draw a line over a known feature to set the physical scale
- **ROI & repair zone**: Define rectangular analysis and repair regions interactively
- **Virtual extensometer**: Track two user-selected points for engineering strain
- **NCC-based tracking**: Normalized cross-correlation with configurable subset/step size
- **Strain fields**: Green-Lagrange strains (εxx, εyy, εxy) and von Mises equivalent strain
- **Correlation masking**: Points below the NCC threshold appear as transparent/blank
- **Strain localization detection**: Automatic detection of the first frame exceeding a von Mises threshold
- **DAQ synchronization**: Optional Instron/MTS DAQ file for load/stress pairing
- **Rich outputs**: Per-frame colormapped PNGs, MP4 videos, thesis-quality contour plots, and CSV summaries

## Requirements

- **Python**: 3.9 – 3.11 recommended (tested with 3.10)
- **PyInstaller**: 5.13+ (for .exe packaging)
- **OS**: Windows 10/11 for .exe builds; the Python tool runs on any platform with OpenCV GUI support

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --folder path/to/images --camera 0
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--folder` | *(required)* | Path to folder of `.tif` stereo image pairs |
| `--camera` | `0` | Camera index (0 = left, 1 = right) |
| `--subset` | `29` | Subset (template) size in pixels |
| `--step` | `10` | Step size in pixels |
| `--ncc-threshold` | `0.7` | Minimum NCC score to accept a subset match |
| `--strain-threshold` | `0.02` | Von Mises strain threshold for localization detection |
| `--area` | *None* | Cross-sectional area (physical units²) for stress calculation |
| `--daq` | *None* | Path to DAQ `.txt` file (tab-separated) |

### Interactive Steps

1. **Calibration**: Click two points on a known feature, enter the physical length and unit
2. **ROI selection**: Draw a rectangle around the analysis region (or press C to use full image)
3. **Repair zone**: Optionally define a repair zone for separate statistics
4. **Virtual extensometer**: Optionally click two points for extensometer strain tracking

### Output Structure

```
output/
├── frames/          # Per-frame colormapped PNGs for each field
├── videos/          # MP4 videos (10 fps, mp4v codec) for each field
├── plots/           # High-resolution contour plots (final frame)
├── summary.csv      # Per-frame statistics, load, stress, localization flag
├── extensometer_strain.csv
└── stress_strain.csv
```

## Building the Windows .exe

```bash
pyinstaller dic_tool.spec
```

Or double-click `build.bat` on Windows. The executable will be in `dist/dic_tool.exe`.

### Packaging Notes

- **Python version**: 3.10.x recommended for best OpenCV + PyInstaller compatibility
- **PyInstaller version**: 5.13 or 6.x
- The spec file explicitly includes `matplotlib/mpl-data` and `cv2` data files as hidden imports/datas to prevent runtime crashes
- If the .exe crashes on first run, check that your Python environment matches the versions above

## Architecture

| Module | Responsibility |
|---|---|
| `main.py` | Entry point, CLI, orchestration |
| `calibration.py` | Interactive scale calibration via line drawing |
| `roi.py` | ROI and repair zone selection |
| `extensometer.py` | Virtual extensometer point tracking |
| `dic_engine.py` | NCC correlation and displacement computation |
| `strain.py` | Green-Lagrange strain fields and NCC masking |
| `output.py` | File writing, video generation, plot export |
| `sync.py` | DAQ file parsing and frame-to-load matching |
