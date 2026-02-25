"""DAQ file parsing and frame-to-load matching for Instron/MTS output."""

import numpy as np


def parse_daq(filepath):
    """Parse a tab-separated DAQ file.

    Expected format (Instron / MTS style):
    - A header block of metadata lines (non-numeric first column)
    - Followed by a row of column names
    - Then numeric data rows with columns including time, load, and displacement

    Returns
    -------
    daq : dict with keys
        'time'         : 1-D ndarray (s)
        'load'         : 1-D ndarray (N or kN depending on source)
        'displacement' : 1-D ndarray (mm or inches depending on source)
        'columns'      : list of column name strings
        'data'         : full 2-D ndarray of the numeric block
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Walk past the header block: find the first line whose first token is
    # numeric (after optional whitespace).  The line immediately *before*
    # that is the column-name row.
    header_end = 0
    for i, line in enumerate(lines):
        token = line.strip().split("\t")[0].strip()
        try:
            float(token)
            header_end = i
            break
        except ValueError:
            continue

    # Column names are on the line just before numeric data starts.
    col_line = lines[header_end - 1] if header_end > 0 else ""
    columns = [c.strip() for c in col_line.split("\t") if c.strip()]

    # Parse the numeric block.
    data_lines = lines[header_end:]
    rows = []
    for line in data_lines:
        parts = line.strip().split("\t")
        try:
            rows.append([float(p) for p in parts if p.strip()])
        except ValueError:
            continue
    data = np.array(rows)

    # Try to identify time / load / displacement columns by name heuristics.
    def _find_col(keywords):
        for kw in keywords:
            for idx, c in enumerate(columns):
                if kw in c.lower():
                    return idx
        return None

    t_idx = _find_col(["time"])
    l_idx = _find_col(["load", "force"])
    d_idx = _find_col(["disp", "position", "extension", "stroke"])

    time_col = data[:, t_idx] if t_idx is not None else np.arange(len(data), dtype=float)
    load_col = data[:, l_idx] if l_idx is not None else np.zeros(len(data))
    disp_col = data[:, d_idx] if d_idx is not None else np.zeros(len(data))

    return {
        "time": time_col,
        "load": load_col,
        "displacement": disp_col,
        "columns": columns,
        "data": data,
    }


def match_frames_to_daq(n_frames, daq):
    """Map each image frame index to the closest DAQ row by evenly
    distributing frames across the DAQ time range.

    Returns
    -------
    indices : list of int, length *n_frames*
        DAQ row index for each image frame.
    """
    daq_time = daq["time"]
    # Assume images span the full test duration, evenly spaced.
    frame_times = np.linspace(daq_time[0], daq_time[-1], n_frames)
    indices = [int(np.argmin(np.abs(daq_time - ft))) for ft in frame_times]
    return indices
