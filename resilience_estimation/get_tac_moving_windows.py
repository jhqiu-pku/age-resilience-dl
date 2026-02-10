"""
Compute STL-based anomalies from monthly GeoTIFF data and estimate
resilience metrics in moving windows.
"""

import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm
from scipy import stats
from statsmodels.tsa.seasonal import STL

no_veg_mask = np.load("no_veg_mask.npy")
data_dir = 'TL_GPP'
flux_range = (
    pd.date_range(start="19830101", end="20231231", freq="MS")
    .strftime("%Y%m%d")
    .tolist()
)


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------
def stl_flux_ano(arr: np.ndarray) -> np.ndarray:
    """
    Apply STL decomposition to each grid cell and return the residual component
    as the anomaly time series.

    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (time, lat, lon).

    Returns
    -------
    np.ndarray
        Residual anomalies with the same shape as input.
    """
    # Initialize anomaly array
    arr_ano = np.full(arr.shape, np.nan, dtype=float)
    # Loop over spatial grid
    for i in tqdm(range(arr.shape[1]), desc="STL loop"):  # latitude
        for j in range(arr.shape[2]):  # longitude
            y = arr[:, i, j]  # time series at one grid cell
            valid = ~np.isnan(y)
            # Require enough samples for stable decomposition
            if np.sum(valid) > 59:
                stl = STL(y, period=12, trend=61, seasonal=7)
                res = stl.fit()
                # Use residuals as anomalies
                arr_ano[:, i, j] = res.resid

    return arr_ano


def get_data(date_range, data_dir: str, ano_name: str) -> np.ndarray:
    """
    Read GeoTIFF stacks, apply vegetation mask, compute STL anomalies, and save to disk.

    Parameters
    ----------
    date_range : list[str]
        Date strings (YYYYMMDD) used to locate GeoTIFFs.
    data_dir : str
        Directory containing GeoTIFFs named {YYYYMMDD}.tif.
    ano_name : str
        Output stem name to save anomalies into paper_output/{ano_name}.npy.

    Returns
    -------
    np.ndarray
        Anomaly array with shape (time, lat, lon).
    """
    # Load full time stack
    arrs = []
    for date in date_range:
        ds = gdal.Open(f"{data_dir}/{date}.tif")
        if ds is None:
            raise FileNotFoundError(f"Cannot open: {data_dir}/{date}.tif")
        arrs.append(ds.ReadAsArray())
    arrs = np.asarray(arrs, dtype=float)
    arrs[:, ~no_veg_mask] = np.nan
    # Compute anomalies and save
    anomaly = stl_flux_ano(arrs)
    np.save(f"paper_output/{ano_name}.npy", anomaly)

    return anomaly


def get_resilience(date_range, anomaly: np.ndarray, stride: int):
    """
    Estimate lag-1 regression slopes and p-values using moving windows along the time dimension.

    Parameters
    ----------
    date_range : list[str]
        Full time index (used for window indexing length).
    anomaly : np.ndarray
        Anomaly array (time, lat, lon).
    stride : int
        Window length.

    Returns
    -------
    np.ndarray
        Slopes per window, shape (n_windows, lat, lon).
    np.ndarray
        Two-sided p-values per window, shape (n_windows, lat, lon).
    """
    china_tac, china_p = [], []
    _, lat_dim, lon_dim = anomaly.shape

    # Slide windows across time
    for i in tqdm(range(0, len(date_range) - stride + 1), desc="Moving window"):
        # Extract current window: (stride, lat, lon)
        win_ano = anomaly[i:i + stride]

        # Allocate outputs for this window
        yr_tac = np.full((lat_dim, lon_dim), np.nan, dtype=float)
        yr_p = np.full((lat_dim, lon_dim), np.nan, dtype=float)

        # Construct lagged pairs
        y = win_ano[1:]   # current time step (y_t)
        x = win_ano[:-1]  # previous time step (y_{t-1})
        valid_mask = ~np.isnan(y).any(axis=0) & ~np.isnan(x).any(axis=0)
        # Window means (used for regression)
        x_mean = np.nanmean(x, axis=0)
        y_mean = np.nanmean(y, axis=0)

        # Slope computed by covariance / variance form
        sum_xy = np.nansum((x - x_mean) * (y - y_mean), axis=0)
        sum_xx = np.nansum((x - x_mean) ** 2, axis=0)
        b = sum_xy / sum_xx
        a = y_mean - b * x_mean
        residuals = y - (a + b * x)

        # Sum of squared residuals
        sse = np.nansum(residuals ** 2, axis=0)

        # Degrees of freedom
        df = stride - 3

        # Standard error, t-statistic, and two-sided p-value
        var_b = sse / df / sum_xx
        se_b = np.sqrt(var_b)
        t_stat = b / se_b
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

        # Store results only for valid cells
        yr_tac[valid_mask] = b[valid_mask]
        yr_p[valid_mask] = p_values[valid_mask]

        china_tac.append(yr_tac)
        china_p.append(yr_p)

    return np.asarray(china_tac), np.asarray(china_p)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    gpp_ano = get_data(
        date_range=flux_range,
        data_dir=data_dir,
        ano_name="gpp_ano",
    )
    # Compute moving-window metrics
    gpp_tac, gpp_p = get_resilience(
        date_range=flux_range,
        anomaly=gpp_ano,
        stride=60,
    )
