# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building the DIC tool as a single-file Windows .exe."""

import os
import sys

block_cipher = None

# Locate site-packages for hidden data files.
import matplotlib
import cv2

mpl_data = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data')
cv2_dir = os.path.dirname(cv2.__file__)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        (mpl_data, 'matplotlib/mpl-data'),
        (cv2_dir, 'cv2'),
    ],
    hiddenimports=[
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        'scipy',
        'scipy.signal',
        'scipy.signal._signaltools',
        'scipy.ndimage',
        'matplotlib',
        'matplotlib.backends.backend_agg',
        'matplotlib.pyplot',
        'cv2',
        'natsort',
        'calibration',
        'roi',
        'extensometer',
        'dic_engine',
        'strain',
        'output',
        'sync',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='dic_tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
