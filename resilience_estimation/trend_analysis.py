"""
Compute pixel-wise linear trends from gridded time series.

Two use cases are supported
---------------------------
1) Meteorological-type variables:
   - Input: monthly GeoTIFF files
   - Workflow: load data -> moving-window mean -> trend estimation

2) Derived metrics (resilience / TAC):
   - Input: precomputed TAC array
   - Workflow: trend estimation
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
from scipy import stats


# --------------------------------------------------
# Time configuration
# --------------------------------------------------
DATE_RANGE = (
    pd.date_range(start="19830101", end="20231231", freq="MS")
    .strftime("%Y%m%d")
    .tolist()
)

WINDOW_LENGTH = 60  # months


# --------------------------------------------------
# Core functions
# --------------------------------------------------
def load_monthly_stack(date_range, data_dir):
    """
    Load monthly GeoTIFF files into a 3D array (time, lat, lon).
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError("Input data directory does not exist.")

    arrs = []
    for date in date_range:
        ds = gdal.Open(os.path.join(data_dir, f"{date}.tif"))
        if ds is None:
            raise FileNotFoundError(f"Missing file: {date}.tif")
        arrs.append(ds.ReadAsArray())

    return np.asarray(arrs, dtype=float)


def moving_window_mean(arr, window):
    """
    Compute moving-window means along the time dimension.
    """
    means = []
    for i in tqdm(range(0, arr.shape[0] - window + 1), desc="Moving window"):
        means.append(np.mean(arr[i:i + window], axis=0))
    return np.asarray(means, dtype=float)


def pixelwise_trend(arr):
    """
    Estimate linear trend and p-value for each grid cell.
    """
    time_years = np.arange(arr.shape[0]) / 12.0

    trend = np.full(arr.shape[1:], np.nan, dtype=float)
    pvalue = np.full(arr.shape[1:], np.nan, dtype=float)

    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            y = arr[:, i, j]
            valid = ~np.isnan(y)

            if np.sum(valid) > 59:
                slope, _, _, p, _ = stats.linregress(
                    time_years[valid], y[valid]
                )
                trend[i, j] = slope
                pvalue[i, j] = p

    return trend, pvalue


def compute_window_trend(date_range, data_dir, window):
    """
    Full workflow for meteorological-type variables:
    load data -> window mean -> trend.
    """
    stack = load_monthly_stack(date_range, data_dir)
    window_means = moving_window_mean(stack, window)
    trend, pvalue = pixelwise_trend(window_means)
    return trend, pvalue, window_means


# --------------------------------------------------
# Main workflow
# --------------------------------------------------
if __name__ == "__main__":

    # ==================================================
    # Case 1: Meteorological-type variables
    # (monthly GeoTIFFs)
    # ==================================================
    MET_DATA_DIR = "<PATH_TO_MONTHLY_TIFFS>"

    met_trend, met_p, met_window = compute_window_trend(
        date_range=DATE_RANGE,
        data_dir=MET_DATA_DIR,
        window=WINDOW_LENGTH,
    )

    # ==================================================
    # Case 2: Derived metrics (e.g., TAC / resilience)
    # (computed TAC array, trend only)
    # ==================================================
    TAC_ARRAY_PATH = "<PATH_TO_TACT_ARRAY_NPY>"
    tac_arr = np.load(TAC_ARRAY_PATH)
    tac_trend, tac_trend_p = pixelwise_trend(tac_arr)
