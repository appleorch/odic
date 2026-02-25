"""Strain field computation (Green-Lagrange) and NCC-based masking."""

import numpy as np


def compute_strain_fields(u_phys, v_phys, step_phys):
    """Compute Green-Lagrange strain from displacement grids.

    Parameters
    ----------
    u_phys, v_phys : 2-D arrays (physical units, NaN where masked)
    step_phys : float
        Grid spacing in physical units (step_size * scale).

    Returns
    -------
    exx, eyy, exy, von_mises : 2-D arrays (same shape, NaN where invalid)
    """
    # Central finite differences; np.gradient handles NaN propagation.
    # axis 1 → x direction, axis 0 → y direction.
    du_dx = np.gradient(u_phys, step_phys, axis=1)
    du_dy = np.gradient(u_phys, step_phys, axis=0)
    dv_dx = np.gradient(v_phys, step_phys, axis=1)
    dv_dy = np.gradient(v_phys, step_phys, axis=0)

    # Green-Lagrange: E = 0.5*(F^T F - I)
    #   Exx = du/dx + 0.5*(du/dx^2 + dv/dx^2)
    #   Eyy = dv/dy + 0.5*(du/dy^2 + dv/dy^2)
    #   Exy = 0.5*(du/dy + dv/dx) + 0.5*(du/dx*du/dy + dv/dx*dv/dy)
    exx = du_dx + 0.5 * (du_dx ** 2 + dv_dx ** 2)
    eyy = dv_dy + 0.5 * (du_dy ** 2 + dv_dy ** 2)
    exy = 0.5 * (du_dy + dv_dx) + 0.5 * (du_dx * du_dy + dv_dx * dv_dy)

    # Von Mises equivalent strain (plane-stress form).
    von_mises = np.sqrt(exx ** 2 + eyy ** 2 - exx * eyy + 3 * exy ** 2)

    return exx, eyy, exy, von_mises


def mask_fields(fields, corr, ncc_threshold):
    """Set every grid point with peak NCC < threshold to NaN across all fields.

    Parameters
    ----------
    fields : dict mapping name → 2-D array
    corr : 2-D array of peak NCC values
    ncc_threshold : float

    Returns
    -------
    masked : dict with same keys, NaN applied where corr < threshold
    """
    bad = (corr < ncc_threshold) | np.isnan(corr)
    masked = {}
    for name, arr in fields.items():
        a = arr.copy()
        a[bad] = np.nan
        masked[name] = a
    return masked
