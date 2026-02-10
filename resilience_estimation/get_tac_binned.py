"""
Compute 2D binned mean of TAC as a function of the relative importance of climatic
control on GPP variability and the corresponding meteorological trend.
"""

import numpy as np


def compute_binned_mean(
    ctl: np.ndarray,
    met: np.ndarray,
    tgt: np.ndarray,
    bins: int = 50,
):
    """
    Bin (ctl, met) into a 2D grid and compute mean(tgt) per bin.

    Parameters
    ----------
    ctl : np.ndarray
        Relative importance of climatic control on GPP variability for a given
        meteorological variable, 2D array (lat, lon).
    met : np.ndarray
        Trend of the corresponding meteorological variable, 2D array (lat, lon).
    tgt : np.ndarray
        Target variable (TAC), 2D array (lat, lon).
    bins : int, optional
        Number of bins along each dimension.

    Returns
    -------
    bin_mean : np.ndarray
        Mean TAC in each (met, ctl) bin, shape (bins, bins).
        bin_mean[i, j] corresponds to met_bins[i]..met_bins[i+1] and
        ctl_bins[j]..ctl_bins[j+1].
    """
    # Flatten spatial grids
    flat_ctl = ctl.ravel()
    flat_met = met.ravel()
    flat_tgt = tgt.ravel()

    # Retain grid cells with valid values for all variables
    valid = (~np.isnan(flat_ctl)) & (~np.isnan(flat_met)) & (~np.isnan(flat_tgt))
    fc = flat_ctl[valid]
    fm = flat_met[valid]
    ft = flat_tgt[valid]

    if fc.size == 0:
        raise ValueError("No valid samples available for binning.")

    # Define equally spaced bin edges
    ctl_bins = np.linspace(fc.min(), fc.max(), bins + 1)
    met_bins = np.linspace(fm.min(), fm.max(), bins + 1)

    # Allocate output array
    bin_mean = np.full((bins, bins), np.nan, dtype=float)

    # Assign samples to 2D bins
    for i in range(bins):  # meteorological axis
        m0, m1 = met_bins[i], met_bins[i + 1]
        met_sel = (fm >= m0) & (fm < m1)

        for j in range(bins):  # climatic control axis
            c0, c1 = ctl_bins[j], ctl_bins[j + 1]
            sel = met_sel & (fc >= c0) & (fc < c1)

            if np.any(sel):
                bin_mean[i, j] = np.mean(ft[sel])

    return bin_mean


if __name__ == "__main__":
    # Input arrays must share identical spatial dimensions (lat, lon)
    ctl_arr = np.load("<PATH_TO_CLIMATIC_CONTROL_IMPORTANCE_ARRAY>")
    met_arr = np.load("<PATH_TO_METEOROLOGICAL_TREND_ARRAY>")
    tgt_arr = np.load("<PATH_TO_TAC_ARRAY>")

    tac_bin = compute_binned_mean(ctl_arr, met_arr, tgt_arr, bins=50)
